from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kimi_agent_sdk import (
    BriefDisplayBlock,
    StepInterrupted,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
    TurnBegin,
    TurnEnd,
)

from kimix_tui.history import (
    HistoryAccumulator,
    HistoryBlock,
    _scan_wire_history_index,
    blocks_from_wire_messages,
    load_session_history,
    load_wire_history_page,
    take_last_turns,
)


class SteerInput:
    def __init__(self, user_input: str) -> None:
        self.user_input = user_input


def test_blocks_merge_stream_and_keep_user_turns() -> None:
    blocks = blocks_from_wire_messages(
        [
            TurnBegin(user_input="fix login"),
            ThinkPart(think="looking "),
            ThinkPart(think="at auth"),
            TextPart(text="Check "),
            TextPart(text="the redirect."),
            TurnEnd(),
            SteerInput("also cookies"),
        ]
    )

    assert blocks == [
        HistoryBlock("user", "fix login"),
        HistoryBlock("thinking", "looking at auth"),
        HistoryBlock("assistant", "Check the redirect."),
        HistoryBlock("user", "also cookies"),
    ]


def test_history_keeps_full_user_and_assistant_messages() -> None:
    user_text = "user " + ("u" * 4_500)
    assistant_text = "assistant " + ("a" * 4_500)

    blocks = blocks_from_wire_messages(
        [
            TurnBegin(user_input=user_text),
            TextPart(text=assistant_text[:2_000]),
            TextPart(text=assistant_text[2_000:]),
        ]
    )

    assert blocks == [
        HistoryBlock("user", user_text),
        HistoryBlock("assistant", assistant_text),
    ]


def test_take_last_turns_keeps_tail_and_counts_omitted() -> None:
    blocks = [
        HistoryBlock("user", "one"),
        HistoryBlock("assistant", "a1"),
        HistoryBlock("user", "two"),
        HistoryBlock("assistant", "a2"),
        HistoryBlock("user", "three"),
        HistoryBlock("assistant", "a3"),
        HistoryBlock("user", "four"),
        HistoryBlock("assistant", "a4"),
    ]

    history = take_last_turns(blocks, max_turns=2)

    assert history.omitted_turns == 2
    assert [block.text for block in history.blocks] == ["three", "a3", "four", "a4"]


async def test_load_session_history_uses_injected_messages(tmp_path: Path) -> None:
    messages = [
        TurnBegin(user_input="older"),
        TextPart(text="old reply"),
        TurnBegin(user_input="newest"),
        TextPart(text="new reply"),
    ]

    history = await load_session_history(
        tmp_path,
        "sess-1",
        max_turns=1,
        messages=messages,
    )

    assert history.omitted_turns == 1
    assert history.blocks == [
        HistoryBlock("user", "newest"),
        HistoryBlock("assistant", "new reply"),
    ]


async def test_load_session_history_empty_when_session_missing(tmp_path: Path) -> None:
    history = await load_session_history(tmp_path, "missing-session")

    assert history.blocks == []
    assert history.omitted_turns == 0


async def test_load_session_history_keeps_all_turns_by_default(tmp_path: Path) -> None:
    messages: list[object] = []
    for index in range(6):
        messages.append(TurnBegin(user_input=f"q{index}"))
        messages.append(TextPart(text=f"a{index}"))

    history = await load_session_history(tmp_path, "sess-all", messages=messages)

    assert history.omitted_turns == 0
    assert [block.text for block in history.blocks if block.kind == "user"] == [
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
    ]


async def test_load_session_history_reads_only_tail_turns_from_wire_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    from kaos.path import KaosPath
    from kimi_cli.session import Session as CliSession

    work_dir = tmp_path / "project"
    kaos_dir = KaosPath.unsafe_from_local_path(work_dir.resolve()).canonical()
    session = await CliSession.create(kaos_dir, "tail-session")
    await session.wire_file.open()
    try:
        for index in range(5):
            await session.wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            await session.wire_file.append_message(
                TextPart(text=f'a{index} includes \\"message\\":{{\\"type\\":\\"TurnBegin\\"}}')
            )
            await session.wire_file.append_message(TurnEnd())
    finally:
        await session.wire_file.close()

    history = await load_session_history(work_dir, session.id, max_turns=2, max_blocks=10)

    assert history.omitted_turns == 3
    assert [block.text for block in history.blocks if block.kind == "user"] == ["q3", "q4"]
    assert [block.text for block in history.blocks if block.kind == "assistant"] == [
        'a3 includes \\"message\\":{\\"type\\":\\"TurnBegin\\"}',
        'a4 includes \\"message\\":{\\"type\\":\\"TurnBegin\\"}',
    ]


async def test_indexed_history_pages_read_only_requested_turns(tmp_path: Path) -> None:
    from kimi_cli.wire.file import WireFile

    wire_file = WireFile(tmp_path / "wire.jsonl")
    await wire_file.open()
    try:
        for index in range(6):
            await wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            await wire_file.append_message(TextPart(text=f"a{index}"))
            await wire_file.append_message(TurnEnd())
    finally:
        await wire_file.close()

    wire_index = _scan_wire_history_index(wire_file.path)
    assert wire_index.total_turns == 6

    latest = await load_wire_history_page(wire_index, end_turn=6, page_turns=2)
    assert latest.start_turn == 4
    assert latest.end_turn == 6
    assert latest.has_older is True
    assert [block.text for block in latest.blocks if block.kind == "user"] == ["q4", "q5"]

    older = await load_wire_history_page(wire_index, end_turn=4, page_turns=2)
    assert older.start_turn == 2
    assert older.end_turn == 4
    assert [block.text for block in older.blocks if block.kind == "user"] == ["q2", "q3"]


async def test_indexed_history_disables_json_string_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pydantic_core
    from kimi_cli.wire.file import WireFile

    wire_file = WireFile(tmp_path / "wire.jsonl")
    await wire_file.open()
    try:
        for index in range(3):
            await wire_file.append_message(TurnBegin(user_input=f"q-{index}"))
            await wire_file.append_message(TextPart(text=f"a-{index}"))
            await wire_file.append_message(TurnEnd())
    finally:
        await wire_file.close()

    cache_settings: list[object] = []
    real_from_json = pydantic_core.from_json

    def record_cache_setting(data: object, *args: object, **kwargs: object) -> object:
        cache_settings.append(kwargs.get("cache_strings"))
        return real_from_json(data, *args, **kwargs)

    monkeypatch.setattr(pydantic_core, "from_json", record_cache_setting)
    wire_index = _scan_wire_history_index(wire_file.path)
    history = await load_wire_history_page(wire_index, end_turn=3, page_turns=3)

    assert [block.text for block in history.blocks if block.kind == "user"] == [
        "q-0",
        "q-1",
        "q-2",
    ]
    assert cache_settings
    assert set(cache_settings) == {False}


def test_unknown_objects_are_ignored() -> None:
    assert blocks_from_wire_messages([SimpleNamespace(noise=True), TurnEnd()]) == []


def test_history_keeps_streamed_tool_arguments_and_detailed_result() -> None:
    blocks = blocks_from_wire_messages(
        [
            TurnBegin(user_input="read it"),
            ToolCall(
                id="call-1",
                function=ToolCall.FunctionBody(name="read", arguments=""),
            ),
            ToolCallPart(arguments_part='{"path":'),
            ToolCallPart(arguments_part='"a.py"}'),
            ToolResult(
                tool_call_id="call-1",
                return_value=ToolReturnValue(
                    is_error=False,
                    output="contents",
                    message="success",
                    display=[BriefDisplayBlock(text="read a.py")],
                    extras=None,
                ),
            ),
        ]
    )

    assert [block.kind for block in blocks] == ["user", "tool", "tool_result"]
    assert "Arguments:" in blocks[1].text
    assert '"path": "a.py"' in blocks[1].text
    assert "read · succeeded" in blocks[2].text
    assert "Display:\nread a.py" in blocks[2].text
    assert "Output:\ncontents" in blocks[2].text


def test_accumulator_discards_turns_outside_the_window() -> None:
    accumulator = HistoryAccumulator(max_turns=2, max_blocks=50)
    for index in range(5):
        accumulator.feed(TurnBegin(user_input=f"q{index}"))
        accumulator.feed(TextPart(text=f"a{index}"))

    history = accumulator.finish()

    assert history.omitted_turns == 3
    assert [block.text for block in history.blocks] == ["q3", "a3", "q4", "a4"]


def test_accumulator_caps_retained_blocks() -> None:
    accumulator = HistoryAccumulator(max_turns=10, max_blocks=3)
    accumulator.feed(TurnBegin(user_input="question"))
    for _ in range(8):
        accumulator.feed(StepInterrupted())

    history = accumulator.finish()

    assert len(history.blocks) == 3
    assert all(block.kind == "error" for block in history.blocks)

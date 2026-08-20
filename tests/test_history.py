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
    Timeline,
    _scan_wire_history_index,
    blocks_from_wire_messages,
    create_timeline,
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


async def test_indexed_history_page_keeps_user_turns_when_block_cap_hits(
    tmp_path: Path,
) -> None:
    from kimi_cli.wire.file import WireFile

    wire_file = WireFile(tmp_path / "verbose-wire.jsonl")
    await wire_file.open()
    try:
        for index in range(4):
            await wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            for _ in range(20):
                await wire_file.append_message(StepInterrupted())
            await wire_file.append_message(TextPart(text=f"a{index}"))
            await wire_file.append_message(TurnEnd())
    finally:
        await wire_file.close()

    wire_index = _scan_wire_history_index(wire_file.path)
    page = await load_wire_history_page(
        wire_index,
        end_turn=4,
        page_turns=4,
        max_blocks=8,
    )

    assert page.start_turn == 0
    assert page.end_turn == 4
    assert page.has_older is False
    assert [block.text for block in page.blocks if block.kind == "user"] == [
        "q0",
        "q1",
        "q2",
        "q3",
    ]
    assert [block.text for block in page.blocks if block.kind == "assistant"] == [
        "a0",
        "a1",
        "a2",
        "a3",
    ]


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
    assert blocks[1].text.startswith("read  a.py")
    assert "Arguments:" not in blocks[1].text
    assert "a.py" in blocks[2].text
    assert "contents" in blocks[2].text
    assert "Display:" not in blocks[2].text
    assert "Output:" not in blocks[2].text


def test_accumulator_discards_turns_outside_the_window() -> None:
    accumulator = HistoryAccumulator(max_turns=2, max_blocks=50)
    for index in range(5):
        accumulator.feed(TurnBegin(user_input=f"q{index}"))
        accumulator.feed(TextPart(text=f"a{index}"))

    history = accumulator.finish()

    assert history.omitted_turns == 3
    assert [block.text for block in history.blocks] == ["q3", "a3", "q4", "a4"]


def test_accumulator_caps_auxiliary_blocks_but_keeps_dialogue() -> None:
    accumulator = HistoryAccumulator(max_turns=10, max_blocks=3)
    accumulator.feed(TurnBegin(user_input="question"))
    for _ in range(8):
        accumulator.feed(StepInterrupted())

    history = accumulator.finish()

    assert [block.kind for block in history.blocks] == ["user", "error", "error"]
    assert history.blocks[0].text == "question"


def test_accumulator_block_cap_keeps_user_turns_on_verbose_pages() -> None:
    accumulator = HistoryAccumulator(max_turns=0, max_blocks=8)
    for index in range(4):
        accumulator.feed(TurnBegin(user_input=f"q{index}"))
        for _ in range(20):
            accumulator.feed(StepInterrupted())
        accumulator.feed(TextPart(text=f"a{index}"))

    history = accumulator.finish()

    assert [block.text for block in history.blocks if block.kind == "user"] == [
        "q0",
        "q1",
        "q2",
        "q3",
    ]
    assert [block.text for block in history.blocks if block.kind == "assistant"] == [
        "a0",
        "a1",
        "a2",
        "a3",
    ]
    assert [block.kind for block in history.blocks] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def _turn_blocks(count: int) -> list[list[HistoryBlock]]:
    return [
        [HistoryBlock("user", f"q{index}"), HistoryBlock("assistant", f"a{index}")]
        for index in range(count)
    ]


def test_timeline_from_turn_blocks_exposes_display_items() -> None:
    timeline = Timeline.from_turn_blocks(_turn_blocks(3))

    assert timeline.total_turns == 3
    assert timeline.materialized_turn_count == 3
    assert timeline.display_items() == [
        ("user", "q0", 0),
        ("assistant", "a0", 0),
        ("user", "q1", 1),
        ("assistant", "a1", 1),
        ("user", "q2", 2),
        ("assistant", "a2", 2),
    ]
    assert timeline.turn_at_line(0) == 0
    assert timeline.turn_at_line(timeline.first_line_of_turn(2)) == 2
    assert timeline.virtual_lines() > 0


async def test_timeline_unload_distant_drops_turns_outside_window() -> None:
    timeline = Timeline.from_turn_blocks(_turn_blocks(8))

    timeline.unload_distant(keep_turn=7, radius=1)

    assert timeline.stubs_for_turn(0) is None
    assert timeline.stubs_for_turn(5) is None
    assert timeline.stubs_for_turn(6)[0].body == "q6"
    assert timeline.stubs_for_turn(7)[0].body == "q7"
    assert [text for kind, text, _turn in timeline.display_items() if kind == "user"] == ["q6", "q7"]

    await timeline.materialize_turns(0, 1, hydrate=True)

    assert timeline.stubs_for_turn(0)[0].body == "q0"
    assert timeline.stubs_for_turn(0)[1].body == "a0"


async def test_timeline_slide_to_keeps_only_window_and_skips_the_gap() -> None:
    timeline = Timeline.from_turn_blocks(_turn_blocks(20))

    await timeline.slide_to(0)

    assert timeline.first_materialized_turn() == 0
    assert timeline.last_materialized_turn() == 3
    assert timeline.materialized_turn_count == 4
    assert timeline.stubs_for_turn(19) is None
    users = [text for kind, text, _turn in timeline.display_items() if kind == "user"]
    assert users == ["q0", "q1", "q2", "q3"]
    assert all(stub.body is not None for stub in timeline.iter_materialized_stubs())

    await timeline.slide_to(19)

    assert timeline.stubs_for_turn(0) is None
    assert timeline.first_materialized_turn() == 16
    assert timeline.last_materialized_turn() == 19
    assert timeline.stubs_for_turn(19)[0].body == "q19"
    users = [text for kind, text, _turn in timeline.display_items() if kind == "user"]
    assert users == ["q16", "q17", "q18", "q19"]


async def test_timeline_ensure_turn_materializes_neighbors_and_drops_far_turns() -> None:
    timeline = Timeline.from_turn_blocks(_turn_blocks(10))
    timeline.unload_distant(keep_turn=9, radius=0)
    assert timeline.stubs_for_turn(0) is None

    await timeline.ensure_turn(0, radius=1)

    assert timeline.stubs_for_turn(0)[0].body == "q0"
    assert timeline.stubs_for_turn(1)[0].body == "q1"
    assert timeline.stubs_for_turn(9) is None


async def test_timeline_open_eager_hydrates_all_small_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    from kaos.path import KaosPath
    from kimi_cli.session import Session as CliSession

    work_dir = tmp_path / "project"
    kaos_dir = KaosPath.unsafe_from_local_path(work_dir.resolve()).canonical()
    session = await CliSession.create(kaos_dir, "tl-small")
    await session.wire_file.open()
    try:
        for index in range(5):
            await session.wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            await session.wire_file.append_message(TextPart(text=f"a{index}"))
            await session.wire_file.append_message(TurnEnd())
    finally:
        await session.wire_file.close()

    timeline = await create_timeline(work_dir, session.id)

    assert timeline is not None
    assert timeline.total_turns == 5
    assert timeline.materialized_turn_count == 5
    assert [text for kind, text, _turn in timeline.display_items() if kind == "user"] == [
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
    ]


async def test_timeline_open_hydrates_last_turns_for_large_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import kimix_tui.history as history_module

    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    monkeypatch.setattr(history_module, "EAGER_FILE_SIZE", 0)
    from kaos.path import KaosPath
    from kimi_cli.session import Session as CliSession

    work_dir = tmp_path / "project"
    kaos_dir = KaosPath.unsafe_from_local_path(work_dir.resolve()).canonical()
    session = await CliSession.create(kaos_dir, "tl-large")
    await session.wire_file.open()
    try:
        for index in range(6):
            await session.wire_file.append_message(TurnBegin(user_input=f"q{index}"))
            await session.wire_file.append_message(TextPart(text=f"a{index}"))
            await session.wire_file.append_message(TurnEnd())
    finally:
        await session.wire_file.close()

    timeline = await create_timeline(work_dir, session.id)

    assert timeline is not None
    assert timeline.total_turns == 6
    assert timeline.materialized_turn_count == history_module.INITIAL_HYDRATE_TURNS
    assert timeline.first_materialized_turn() == 3
    assert timeline.stubs_for_turn(0) is None
    assert [stub.text for stub in timeline.stubs_for_turn(5) or [] if stub.kind == "user"] == [
        "q5"
    ]

    await timeline.ensure_turn(0)

    assert timeline.stubs_for_turn(0) is not None
    assert timeline.stubs_for_turn(0)[0].body == "q0"
    assert timeline.stubs_for_turn(5) is None
    assert timeline.last_materialized_turn() == 3
    users = [text for kind, text, _turn in timeline.display_items() if kind == "user"]
    assert users == ["q0", "q1", "q2", "q3"]


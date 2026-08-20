from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from kimix_tui.widgets import PromptInput, Transcript


class TranscriptHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")


class PaddedTranscriptHarness(App[None]):
    CSS = """
    #transcript {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")


class BoundedTranscriptHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript", max_chars=32)


class PromptHarness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield PromptInput(placeholder="Ask", id="prompt")

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self.submitted.append(event.value)


def _visible_text(transcript: Transcript) -> str:
    return "\n".join(
        transcript.render_line(y).text for y in range(max(0, transcript.size.height))
    )


@pytest.mark.asyncio
async def test_transcript_keeps_records_for_scrollback() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_block("user", "one")
        await transcript.append_block("user", "two")
        await transcript.append_block("user", "three")
        await transcript.append_block("user", "four")

        assert [record.text for record in transcript.records] == [
            "one",
            "two",
            "three",
            "four",
        ]
        assert transcript.virtual_size.height >= 4
        assert len(list(transcript.children)) == 0


@pytest.mark.asyncio
async def test_transcript_keeps_full_dialogue_text_when_streaming() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        first = "a" * 100
        second = "b" * 4_500
        await transcript.append_stream("assistant", first)
        await transcript.append_stream("assistant", second)

        assert len(transcript.records) == 1
        assert transcript.records[0].text == first + second

        user_text = "question " + ("x" * 4_500)
        await transcript.append_block("user", user_text)
        assert transcript.records[-1].text == user_text


@pytest.mark.asyncio
async def test_transcript_keeps_full_auxiliary_text() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        tool_text = "x" * 4_500
        await transcript.append_block("tool", tool_text)

        assert transcript.records[0].text == tool_text

        stream_first = "a" * 100
        stream_second = "b" * 4_500
        await transcript.append_stream("thinking", stream_first)
        await transcript.append_stream("thinking", stream_second)
        assert transcript.records[-1].text == stream_first + stream_second


@pytest.mark.asyncio
async def test_transcript_paints_records_lazily_and_bounds_memory() -> None:
    app = BoundedTranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [("user", f"message-{index}-{'x' * 20}") for index in range(10)]
        )

        assert transcript.omitted_records > 0
        assert len(transcript._strip_cache) == 0
        transcript.render_line(0)
        assert len(transcript._strip_cache) == 1
        assert len(transcript._strip_cache) <= 32


@pytest.mark.asyncio
async def test_transcript_append_blocks_keeps_full_history() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [
                ("user", "a"),
                ("assistant", "b"),
                ("user", "c"),
            ]
        )

        assert [record.text for record in transcript.records] == ["a", "b", "c"]
        assert transcript.virtual_size.height >= 3


@pytest.mark.asyncio
async def test_non_dialogue_records_take_one_transcript_line() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(40, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_block("tool", "Read file\nPath: example.py\nArguments: {}")

        assert transcript._record_line_counts == [1]
        assert "▸ Read" in transcript.render_line(0).text


@pytest.mark.asyncio
async def test_only_explicit_copy_action_copies_dialogue_message() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [
                ("system", "Session: demo"),
                ("user", "First question"),
                ("assistant", "First answer"),
                ("tool", "Read file\nPath: example.py"),
                ("tool_result", "Success\n10 lines"),
                ("user", "Second question"),
                ("assistant", "Second answer"),
            ]
        )
        await pilot.pause()

        app.copy_to_clipboard("unchanged")
        first_response_header = sum(transcript._record_line_counts[:2])
        await pilot.click(transcript, offset=(4, first_response_header + 1))
        assert app._clipboard == "unchanged"

        copy_x = transcript._content_width() - 3
        await pilot.click(transcript, offset=(copy_x, first_response_header))
        assert app._clipboard == "First answer"

        tool_line = sum(transcript._record_line_counts[:3])
        await pilot.click(transcript, offset=(4, tool_line), button=3)

        assert app._clipboard == "Read file\nPath: example.py"


@pytest.mark.asyncio
async def test_clicking_copy_action_on_system_record_copies_only_that_record() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [
                ("system", "Session: demo"),
                ("user", "Question"),
                ("assistant", "Answer"),
            ]
        )
        await pilot.pause()

        await pilot.click(transcript, offset=(transcript._content_width() - 3, 0))

        assert app._clipboard == "Session: demo"


@pytest.mark.asyncio
async def test_auxiliary_record_expands_collapses_and_copies_from_action() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(40, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_block("tool", "Read file\nPath: example.py")

        await pilot.click(transcript, offset=(8, 0))
        assert transcript.records[0].expanded is True
        assert transcript._record_line_counts[0] > 1
        assert "Path: example.py" in _visible_text(transcript)

        app.copy_to_clipboard("unchanged")
        await pilot.click(transcript, offset=(8, 1))
        assert app._clipboard == "unchanged"
        assert transcript.records[0].expanded is True

        await pilot.click(transcript, offset=(transcript._content_width() - 3, 0))
        assert app._clipboard == "Read file\nPath: example.py"

        await pilot.click(transcript, offset=(8, 0))
        assert transcript.records[0].expanded is False
        assert transcript._record_line_counts == [1]


@pytest.mark.asyncio
async def test_thinking_records_start_expanded_in_italic() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(48, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_block("thinking", "considering options")
        await pilot.pause()

        visible = _visible_text(transcript)
        assert transcript.records[0].expanded is True
        assert transcript._record_line_counts[0] > 1
        assert "▾ Think" in visible
        assert "considering options" in visible
        header = transcript.render_line(0).text
        assert "considering options" not in header


@pytest.mark.asyncio
async def test_clicking_scrolled_message_body_does_not_copy() -> None:
    app = PaddedTranscriptHarness()
    async with app.run_test(size=(80, 12)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [
                item
                for turn in range(12)
                for item in (
                    ("user", f"Question {turn}"),
                    ("assistant", f"Answer {turn}"),
                )
            ]
        )
        await pilot.pause()

        target_record = 10
        target_line = sum(transcript._record_line_counts[:target_record])
        transcript.scroll_to(
            y=target_line,
            animate=False,
            immediate=True,
            force=True,
        )
        await pilot.pause()
        app.copy_to_clipboard("unchanged")
        await pilot.click(transcript, offset=(3, 1))

        assert app._clipboard == "unchanged"


@pytest.mark.asyncio
async def test_transcript_render_line_paints_colored_markdown() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [
                ("user", "please use **bold**"),
                ("assistant", "## Title\n\nUse **bold** and `code`."),
            ]
        )
        for y in range(transcript.size.height):
            strip = transcript.render_line(y)
            assert all(segment.style is not None for segment in strip._segments)


@pytest.mark.asyncio
async def test_transcript_opens_scrolled_to_latest_history() -> None:
    app = PaddedTranscriptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_block("system", "Session: demo")
        await transcript.append_blocks(
            [("user", f"history-first-{i:03d}") for i in range(50)]
            + [("assistant", "history-latest-reply")]
        )
        await pilot.pause()

        assert transcript.max_scroll_y > 0
        assert transcript.scroll_offset.y == transcript.max_scroll_y
        assert transcript.vertical_scrollbar.position == transcript.scroll_y

        visible = _visible_text(transcript)
        assert "history-first-000" not in visible
        assert "history-latest-reply" in visible


@pytest.mark.asyncio
async def test_transcript_stays_put_when_user_scrolls_up() -> None:
    app = PaddedTranscriptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [("user", f"turn-{i:03d}") for i in range(40)]
        )
        await pilot.pause()
        transcript.scroll_to(y=0, animate=False, immediate=True, force=True)
        await pilot.pause()

        assert transcript.scroll_offset.y == 0
        await transcript.append_block("user", "new-after-scroll")
        await pilot.pause()

        assert transcript.scroll_offset.y == 0
        visible = _visible_text(transcript)
        assert "turn-000" in visible
        assert "new-after-scroll" not in visible


@pytest.mark.asyncio
async def test_prepend_history_preserves_the_visible_scroll_anchor() -> None:
    app = PaddedTranscriptHarness()
    async with app.run_test(size=(80, 12)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [("user", f"current-{index:02d}") for index in range(30)]
        )
        transcript.mark_history_window(0, len(transcript.records))
        transcript.scroll_to(y=24, animate=False, immediate=True, force=True)
        await pilot.pause()
        old_scroll = transcript.scroll_offset.y

        added_lines = await transcript.prepend_history_blocks(
            [("user", "older-00"), ("user", "older-01")]
        )
        await pilot.pause()

        assert added_lines == 6
        assert transcript.scroll_offset.y == old_scroll + added_lines
        assert [record.text for record in transcript.records[:2]] == [
            "older-00",
            "older-01",
        ]
        assert "current-08" in _visible_text(transcript)


@pytest.mark.asyncio
async def test_replacing_history_window_keeps_live_rows() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_blocks(
            [("system", "Session: demo"), ("user", "old-history"), ("assistant", "live")]
        )
        transcript.mark_history_window(1, 2)

        await transcript.replace_history_blocks(
            [("user", "page-history"), ("assistant", "page-reply")]
        )

        assert [(record.kind, record.text) for record in transcript.records] == [
            ("system", "Session: demo"),
            ("user", "page-history"),
            ("assistant", "page-reply"),
            ("assistant", "live"),
        ]
        assert transcript.history_window == (1, 3)


@pytest.mark.asyncio
async def test_jump_to_turn_unpins_and_scrolls_immediately() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 12)):
        transcript = app.query_one(Transcript)
        await transcript.replace_history_blocks(
            [
                ("user", f"q{index}", index)
                for index in range(20)
            ]
        )
        transcript.jump_to_latest()
        assert transcript.pinned_to_latest is True

        transcript.jump_to_turn(0)

        assert transcript.pinned_to_latest is False
        assert transcript.viewport_turn() == 0
        assert "q0" in _visible_text(transcript)
        assert "q19" not in _visible_text(transcript)


@pytest.mark.asyncio
async def test_jump_to_latest_pins_without_leaving_older_rows() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 12)):
        transcript = app.query_one(Transcript)
        await transcript.replace_history_blocks(
            [("user", f"q{index}", index) for index in range(20)]
        )
        transcript.jump_to_turn(0)
        assert transcript.pinned_to_latest is False

        transcript.jump_to_latest()

        assert transcript.pinned_to_latest is True
        assert transcript.scroll_offset.y == transcript.max_scroll_y
        assert "q19" in _visible_text(transcript)


@pytest.mark.asyncio
async def test_replace_history_keeps_in_flight_stream() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_block("system", "Session: demo")
        transcript.mark_history_window()
        await transcript.replace_history_blocks([("user", "q0", 0)])
        await transcript.append_stream("assistant", "hel")
        await transcript.append_stream("assistant", "lo")

        await transcript.replace_history_blocks([("user", "q0", 0), ("assistant", "old", 0)])
        await transcript.append_stream("assistant", "!")

        assert [record.text for record in transcript.records] == [
            "Session: demo",
            "q0",
            "old",
            "hello!",
        ]


@pytest.mark.asyncio
async def test_history_window_skips_fifo_trim() -> None:
    app = BoundedTranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        transcript.mark_history_window()
        await transcript.append_blocks(
            [("user", f"message-{index}-{'x' * 20}") for index in range(10)]
        )

        assert transcript.omitted_records == 0
        assert len(transcript.records) == 10


@pytest.mark.asyncio
async def test_prompt_enter_submits_without_inserting_newline() -> None:
    app = PromptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press("h", "i", "enter")
        assert app.submitted == ["hi"]
        assert prompt.text == "hi"


@pytest.mark.asyncio
async def test_prompt_ctrl_enter_inserts_newline_then_enter_sends() -> None:
    app = PromptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press("h", "i", "ctrl+enter", "t", "h", "e", "r", "e")
        assert prompt.text == "hi\nthere"
        assert app.submitted == []
        await pilot.press("enter")
        assert app.submitted == ["hi\nthere"]


@pytest.mark.asyncio
async def test_prompt_shift_enter_inserts_newline() -> None:
    app = PromptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press("a", "shift+enter", "b")
        assert prompt.text == "a\nb"
        assert app.submitted == []


@pytest.mark.asyncio
async def test_prompt_grows_with_lines_and_caps_height() -> None:
    app = PromptHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.pause()
        assert prompt.outer_size.height == PromptInput.MIN_HEIGHT
        await pilot.press(*(["ctrl+enter"] * 8))
        await pilot.pause()
        assert prompt.outer_size.height == PromptInput.MAX_HEIGHT
        assert prompt.text.count("\n") == 8


from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from kimix_tui.rendering import DISPLAY_CHAR_LIMIT
from kimix_tui.widgets import Transcript


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
async def test_transcript_stream_stops_growing_past_display_limit() -> None:
    app = TranscriptHarness()
    async with app.run_test(size=(80, 24)):
        transcript = app.query_one(Transcript)
        await transcript.append_stream("assistant", "a" * 100)
        await transcript.append_stream("assistant", "b" * (DISPLAY_CHAR_LIMIT + 500))

        assert len(transcript.records) == 1
        assert len(transcript.records[0].text) < DISPLAY_CHAR_LIMIT + 80
        assert "truncated" in transcript.records[0].text


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
async def test_clicking_response_copies_only_that_message() -> None:
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

        first_response_line = sum(transcript._record_line_counts[:2]) + 1
        await pilot.click(transcript, offset=(4, first_response_line))

        assert app._clipboard == "First answer"

        tool_line = sum(transcript._record_line_counts[:3])
        await pilot.click(transcript, offset=(4, tool_line))

        assert app._clipboard == "Read file\nPath: example.py"


@pytest.mark.asyncio
async def test_clicking_leading_system_record_copies_only_that_record() -> None:
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

        await pilot.click(transcript, offset=(4, 0))

        assert app._clipboard == "Session: demo"


@pytest.mark.asyncio
async def test_clicking_scrolled_history_uses_the_visible_record() -> None:
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
        await pilot.click(transcript, offset=(3, 1))

        assert app._clipboard == "Question 5"


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

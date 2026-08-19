"""Virtual transcript: all compact records stay in memory, only visible lines paint."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from kimix_tui.rendering import bounded_concat, truncate_display
from kimix_tui.transcript_paint import render_record_strips, sanitize_strip


@dataclass(slots=True)
class TranscriptRecord:
    kind: str
    text: str


class Transcript(ScrollView, can_focus=True):
    """Scrollable chat log that virtualizes painting.

    Records are kept as truncated strings so the user can scroll back through
    the session. Only the lines inside the viewport are turned into strips —
    there is no per-message widget tree.
    """

    DEFAULT_CSS = """
    Transcript {
        background: $surface;
        overflow-x: hidden;
        overflow-y: scroll;
        scrollbar-size: 1 1;
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[TranscriptRecord] = []
        self._strips: list[Strip] = []
        self._record_line_counts: list[int] = []
        self._wrap_width = 0
        self._stream_kind: str | None = None
        self._stream_record: TranscriptRecord | None = None
        self._stick_to_bottom = True

    def on_mount(self) -> None:
        self.anchor()

    def _content_width(self) -> int:
        width = self.size.width
        if width <= 0:
            return 80
        return max(8, width)

    def _wrap_record(self, record: TranscriptRecord) -> list[Strip]:
        return render_record_strips(
            record.kind,
            record.text,
            width=self._content_width(),
            console=self.app.console,
        )

    def _sync_virtual_size(self) -> None:
        self.virtual_size = Size(self._content_width(), len(self._strips))

    def _append_record(self, record: TranscriptRecord) -> None:
        strips = self._wrap_record(record)
        self.records.append(record)
        self._record_line_counts.append(len(strips))
        self._strips.extend(strips)
        self._sync_virtual_size()

    def _replace_last_strips(self) -> None:
        if not self.records:
            return
        old_count = self._record_line_counts[-1]
        new_strips = self._wrap_record(self.records[-1])
        if old_count:
            del self._strips[-old_count:]
        self._strips.extend(new_strips)
        self._record_line_counts[-1] = len(new_strips)
        self._sync_virtual_size()

    def _rewrap_all(self) -> None:
        self._strips.clear()
        self._record_line_counts.clear()
        for record in self.records:
            strips = self._wrap_record(record)
            self._record_line_counts.append(len(strips))
            self._strips.extend(strips)
        self._sync_virtual_size()

    def _maybe_scroll_end(self) -> None:
        if not self._stick_to_bottom:
            return
        self._anchored = True
        self._anchor_released = False
        # force=True: overflow-y auto/scroll may still have no bar until layout,
        # and ScrollView then treats vertical scrolling as prohibited.
        self.scroll_end(animate=False, immediate=True, force=True, x_axis=False)
        self.call_after_refresh(self._finish_scroll_end)

    def _finish_scroll_end(self) -> None:
        if not self._stick_to_bottom:
            return
        self.scroll_end(animate=False, immediate=True, force=True, x_axis=False)
        if self.show_vertical_scrollbar:
            self.vertical_scrollbar.position = self.scroll_y
        self.refresh()

    def on_resize(self) -> None:
        width = self._content_width()
        if width == self._wrap_width:
            return
        self._wrap_width = width
        if self.records:
            self._rewrap_all()
            self._maybe_scroll_end()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if self._anchored and self._anchor_released:
            self._check_anchor()
        self._stick_to_bottom = self.is_vertical_scroll_end or (
            self._anchored and not self._anchor_released
        )
        if round(old_value) != round(new_value):
            # ScrollView refreshes a padding-translated region; clear the
            # line cache so painted rows match the new offset.
            self.refresh()

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        index = scroll_y + y
        width = self.size.width
        style = self.rich_style
        if width <= 0:
            return sanitize_strip(Strip.blank(0, style), style)
        if index < 0 or index >= len(self._strips):
            return sanitize_strip(Strip.blank(width, style), style)
        line = self._strips[index]
        return sanitize_strip(
            line.crop_extend(scroll_x, scroll_x + width, style),
            style,
        )

    async def append_block(self, kind: str, text: str) -> None:
        self.finish_stream()
        self._append_record(TranscriptRecord(kind, truncate_display(text)))
        self.refresh()
        self._maybe_scroll_end()

    async def append_blocks(self, items: Sequence[tuple[str, str]]) -> None:
        self.finish_stream()
        for kind, text in items:
            self._append_record(TranscriptRecord(kind, truncate_display(text)))
        self.refresh()
        self._maybe_scroll_end()

    async def append_stream(self, kind: str, fragment: str) -> None:
        if not fragment:
            return
        if self._stream_kind != kind or self._stream_record is None:
            self.finish_stream()
            record = TranscriptRecord(kind, bounded_concat("", fragment))
            self._append_record(record)
            self._stream_kind = kind
            self._stream_record = record
        else:
            clipped = bounded_concat(self._stream_record.text, fragment)
            if clipped == self._stream_record.text:
                return
            self._stream_record.text = clipped
            self._replace_last_strips()
        self.refresh()
        self._maybe_scroll_end()

    def finish_stream(self) -> None:
        self._stream_kind = None
        self._stream_record = None

    async def clear_messages(self) -> None:
        self.finish_stream()
        self.records.clear()
        self._strips.clear()
        self._record_line_counts.clear()
        self._sync_virtual_size()
        self._stick_to_bottom = True
        self._anchored = True
        self._anchor_released = False
        self.refresh()

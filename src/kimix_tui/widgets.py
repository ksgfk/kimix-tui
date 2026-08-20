"""Chat widgets: multiline prompt input and virtual transcript."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.cells import cell_len
from textual import events, on
from textual.binding import Binding
from textual.geometry import Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import TextArea

from kimix_tui.transcript_paint import (
    copy_hit_start,
    default_expanded,
    is_compact_record,
    is_dialogue_record,
    record_label,
    render_record_strips,
    sanitize_strip,
)


class PromptInput(TextArea):
    """Chat composer: Enter sends, Ctrl+Enter / Shift+Enter insert a newline."""

    MIN_HEIGHT = 3
    MAX_HEIGHT = 8
    BORDER_CHROME = 2

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("ctrl+enter,ctrl+j,shift+enter", "newline", "Newline", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    PromptInput {
        width: 100%;
        height: 3;
        min-height: 3;
        max-height: 8;
        padding: 0 1;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    """

    @dataclass
    class Submitted(Message):
        """Posted when Enter is pressed to send the composer contents."""

        prompt: PromptInput
        value: str

        @property
        def control(self) -> PromptInput:
            return self.prompt

    def __init__(
        self,
        placeholder: str = "",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        tooltip: str | None = None,
    ) -> None:
        super().__init__(
            "",
            soft_wrap=True,
            tab_behavior="focus",
            show_line_numbers=False,
            highlight_cursor_line=False,
            placeholder=placeholder,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            tooltip=tooltip,
        )
        self._fitted_height = self.MIN_HEIGHT
        self.styles.height = self.MIN_HEIGHT

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def action_newline(self) -> None:
        if self.read_only or self.disabled:
            return
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)

    async def _on_key(self, event: events.Key) -> None:
        if not self.read_only and not self.disabled:
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                self.action_submit()
                return
            if event.key in {"ctrl+enter", "ctrl+j", "shift+enter"}:
                event.stop()
                event.prevent_default()
                self.action_newline()
                return
        await super()._on_key(event)

    def on_mount(self) -> None:
        self.call_after_refresh(self._sync_height)

    def on_resize(self, _event: events.Resize) -> None:
        self._sync_height()

    @on(TextArea.Changed)
    def _on_prompt_changed(self, _event: TextArea.Changed) -> None:
        self._sync_height()

    def _content_line_count(self) -> int:
        wrapped_height = self.wrapped_document.height
        if wrapped_height > 0:
            return wrapped_height
        return self.text.count("\n") + 1

    def _sync_height(self) -> None:
        target = min(
            self.MAX_HEIGHT,
            max(self.MIN_HEIGHT, self._content_line_count() + self.BORDER_CHROME),
        )
        if target == self._fitted_height:
            return
        self._fitted_height = target
        self.styles.height = target


@dataclass(slots=True)
class TranscriptRecord:
    kind: str
    text: str
    expanded: bool = False
    turn: int | None = None


MAX_TRANSCRIPT_CHARS = 64 * 1024 * 1024
_TRIM_TARGET_RATIO = 0.9
_LINE_BLOCK_SIZE = 256
_STRIP_CACHE_SIZE = 32


def _new_record(kind: str, text: str, *, turn: int | None = None) -> TranscriptRecord:
    return TranscriptRecord(kind, text, expanded=default_expanded(kind), turn=turn)


def _record_from_history_item(
    item: tuple[str, str] | tuple[str, str, int],
) -> TranscriptRecord:
    kind, text = item[0], item[1]
    turn = item[2] if len(item) > 2 else None
    return _new_record(kind, text, turn=turn)


class Transcript(ScrollView, can_focus=True):
    """Scrollable chat log that virtualizes painting.

    Record text is kept in full. Line heights are estimated cheaply and Rich
    strips are generated lazily near the viewport.
    """

    class ReachedTop(Message):
        """Emitted once when the user reaches the oldest loaded row."""

    class ReachedBottom(Message):
        """Emitted once when the user reaches the newest loaded row."""

    class ViewportTurn(Message):
        """Emitted when the history turn at the top of the viewport changes."""

        def __init__(self, turn: int | None) -> None:
            super().__init__()
            self.turn = turn

    DEFAULT_CSS = """
    Transcript {
        background: $surface;
        overflow-x: hidden;
        overflow-y: scroll;
        scrollbar-size: 1 1;
    }
    """
    ALLOW_SELECT = False

    def __init__(
        self,
        *args: object,
        max_chars: int = MAX_TRANSCRIPT_CHARS,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[TranscriptRecord] = []
        # Kept as an empty compatibility attribute for callers that inspected
        # the old eager cache. Actual strips live in the bounded LRU below.
        self._strips: list[Strip] = []
        self._record_line_counts: list[int] = []
        self._line_block_sums: list[int] = []
        self._strip_cache: OrderedDict[int, list[Strip]] = OrderedDict()
        self._wrap_width = 0
        self._stream_kind: str | None = None
        self._stream_record: TranscriptRecord | None = None
        self._stick_to_bottom = True
        self._max_chars = max(0, max_chars)
        self._record_chars = 0
        self._omitted_records = 0
        self._top_event_armed = True
        self._bottom_event_armed = True
        self._history_start: int | None = None
        self._history_end: int | None = None
        self._last_viewport_turn: int | None = None

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
            expanded=record.expanded,
        )

    @staticmethod
    def _wrapped_lines(text: str, width: int) -> int:
        lines = text.split("\n")
        return sum(max(1, (cell_len(line) + width - 1) // width) for line in lines)

    def _estimate_record_lines(self, record: TranscriptRecord) -> int:
        """Return a conservative line count without parsing Markdown."""

        if is_compact_record(record.kind, expanded=record.expanded):
            return 1

        width = max(8, self._content_width() - 2)
        if not is_dialogue_record(record.kind):
            return 1 + self._wrapped_lines(record.text or "(no details)", width) + 1

        body_lines = self._wrapped_lines(record.text, width)
        return 1 + body_lines + 1

    def _rebuild_line_blocks(self) -> None:
        self._line_block_sums = [
            sum(self._record_line_counts[start : start + _LINE_BLOCK_SIZE])
            for start in range(0, len(self._record_line_counts), _LINE_BLOCK_SIZE)
        ]

    def _set_line_count(self, index: int, count: int) -> None:
        count = max(1, count)
        old_count = self._record_line_counts[index]
        if old_count == count:
            return
        self._record_line_counts[index] = count
        self._line_block_sums[index // _LINE_BLOCK_SIZE] += count - old_count

    def _cache_record_strips(self, index: int) -> list[Strip]:
        strips = self._strip_cache.pop(index, None)
        if strips is None:
            strips = self._wrap_record(self.records[index])
            self._set_line_count(index, len(strips))
        self._strip_cache[index] = strips
        while len(self._strip_cache) > _STRIP_CACHE_SIZE:
            self._strip_cache.popitem(last=False)
        return strips

    def _sync_virtual_size(self) -> None:
        self.virtual_size = Size(self._content_width(), sum(self._line_block_sums))

    def _insert_records(
        self,
        index: int,
        records: Sequence[TranscriptRecord],
        *,
        sync: bool = True,
        trim: bool = False,
    ) -> int:
        if not records:
            return 0
        index = max(0, min(index, len(self.records)))
        line_counts = [self._estimate_record_lines(record) for record in records]
        self.records[index:index] = records
        self._record_line_counts[index:index] = line_counts
        self._record_chars += sum(len(record.text) for record in records)
        self._strip_cache.clear()
        self._rebuild_line_blocks()
        if trim:
            self._trim_records()
        if sync:
            self._sync_virtual_size()
        return sum(line_counts)

    def _append_record(self, record: TranscriptRecord, *, sync: bool = True) -> None:
        self._insert_records(
            len(self.records),
            [record],
            sync=sync,
            trim=True,
        )

    def _replace_last_strips(self, *, sync: bool = True) -> None:
        if not self.records:
            return
        self._strip_cache.pop(len(self.records) - 1, None)
        self._set_line_count(len(self.records) - 1, self._estimate_record_lines(self.records[-1]))
        if sync:
            self._sync_virtual_size()

    def _rewrap_all(self) -> None:
        self._strip_cache.clear()
        self._record_line_counts = [self._estimate_record_lines(record) for record in self.records]
        self._rebuild_line_blocks()
        self._sync_virtual_size()

    def _trim_records(self) -> None:
        if self._history_start is not None:
            # Timeline unloads offscreen bodies; do not FIFO-drop history or the live tail.
            return
        if self._max_chars <= 0 or self._record_chars <= self._max_chars:
            return
        target = max(1, int(self._max_chars * _TRIM_TARGET_RATIO))
        remove_count = 0
        removed_chars = 0
        while (
            remove_count < len(self.records) - 1
            and self._record_chars - removed_chars > target
        ):
            removed_chars += len(self.records[remove_count].text)
            remove_count += 1
        if not remove_count:
            return
        removed_lines = sum(self._record_line_counts[:remove_count])
        del self.records[:remove_count]
        del self._record_line_counts[:remove_count]
        self._record_chars -= removed_chars
        self._omitted_records += remove_count
        if self._history_start is not None:
            self._history_start = max(0, self._history_start - remove_count)
            self._history_end = max(0, (self._history_end or 0) - remove_count)
            if self._history_end <= self._history_start:
                self._history_start = None
                self._history_end = None
        self._strip_cache.clear()
        self._rebuild_line_blocks()
        if not self._stick_to_bottom and self.scroll_y > 0:
            self.scroll_to(
                y=max(0, self.scroll_y - removed_lines),
                animate=False,
                immediate=True,
                force=True,
            )

    @property
    def omitted_records(self) -> int:
        """Number of oldest records removed after exceeding the memory budget."""

        return self._omitted_records

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

    def jump_to_latest(self) -> None:
        """Return to the newest row and restore automatic bottom anchoring."""

        self._stick_to_bottom = True
        self._anchored = True
        self._anchor_released = False
        self._top_event_armed = True
        self._bottom_event_armed = False
        self._maybe_scroll_end()

    @property
    def pinned_to_latest(self) -> bool:
        return self._stick_to_bottom

    def viewport_turn(self) -> int | None:
        """0-based history turn at the top of the viewport, if any."""

        record_index, _local = self._record_at_line(round(self.scroll_y))
        if record_index is None or not (0 <= record_index < len(self.records)):
            return None
        return self.records[record_index].turn

    def jump_to_turn(self, turn: int) -> None:
        """Scroll the first record of ``turn`` to the top without animation."""

        start = self._history_start or 0
        end = self._history_end or len(self.records)
        target_index = None
        for index in range(start, end):
            if self.records[index].turn == turn:
                target_index = index
                break
        if target_index is None:
            return
        target_y = sum(self._record_line_counts[:target_index])
        self._stick_to_bottom = False
        self._anchored = False
        self._anchor_released = True
        self._top_event_armed = False
        self._bottom_event_armed = False
        self.scroll_to(
            y=target_y,
            animate=False,
            immediate=True,
            force=True,
        )
        # A short window can clamp this seek to the visual end; that is not pin-to-latest.
        self._stick_to_bottom = False
        if not self.is_vertical_scroll_end:
            self._bottom_event_armed = True
        self.refresh()

    def jump_to_history_start(self) -> None:
        """Place the first record in the current history window at the top."""

        if self._history_start is None:
            return
        target_y = sum(self._record_line_counts[: self._history_start])
        self._stick_to_bottom = False
        self._anchored = False
        self._anchor_released = True
        self._top_event_armed = False
        self._bottom_event_armed = False
        self.scroll_to(
            y=target_y,
            animate=False,
            immediate=True,
            force=True,
        )
        self.refresh()

    def jump_to_history_end(self) -> None:
        """Place the last record in the current history window at the bottom."""

        if self._history_end is None:
            return
        end_line = sum(self._record_line_counts[: self._history_end])
        target_y = max(0, end_line - max(1, self.size.height))
        self._stick_to_bottom = False
        self._anchored = False
        self._anchor_released = True
        self._top_event_armed = True
        self._bottom_event_armed = False
        self.scroll_to(
            y=target_y,
            animate=False,
            immediate=True,
            force=True,
        )
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
        at_top = new_value <= 0.5
        at_bottom = self.is_vertical_scroll_end
        if self._anchored and not self._anchor_released:
            self._stick_to_bottom = True
        elif at_bottom and new_value > old_value:
            # User scrolled down to the end of the current view.
            self._stick_to_bottom = True
        elif not at_bottom:
            self._stick_to_bottom = False
        if at_top:
            # Short transcripts sit at top and bottom at once; that is not a
            # user scroll to older history, so do not auto-page.
            if self._top_event_armed and self.is_attached and not at_bottom:
                self._top_event_armed = False
                self.post_message(self.ReachedTop())
        else:
            self._top_event_armed = True
        if at_bottom:
            if self._bottom_event_armed and self.is_attached and not at_top:
                self._bottom_event_armed = False
                self.post_message(self.ReachedBottom())
        else:
            self._bottom_event_armed = True
        viewport_turn = self.viewport_turn()
        if viewport_turn != self._last_viewport_turn:
            self._last_viewport_turn = viewport_turn
            if self.is_attached:
                self.post_message(self.ViewportTurn(viewport_turn))
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
        if index < 0 or index >= sum(self._line_block_sums):
            return sanitize_strip(Strip.blank(width, style), style)
        record_index, local_line = self._record_at_line(index)
        if record_index is None:
            return sanitize_strip(Strip.blank(width, style), style)
        strips = self._cache_record_strips(record_index)
        # Estimates are intentionally conservative. If Rich produced fewer
        # rows, adjust the index and retry so following records stay aligned.
        if local_line >= len(strips):
            self._set_line_count(record_index, len(strips))
            self._sync_virtual_size()
            record_index, local_line = self._record_at_line(index)
            if record_index is None:
                return sanitize_strip(Strip.blank(width, style), style)
            strips = self._cache_record_strips(record_index)
        if local_line >= len(strips):
            return sanitize_strip(Strip.blank(width, style), style)
        line = strips[local_line]
        return sanitize_strip(
            line.crop_extend(scroll_x, scroll_x + width, style),
            style,
        )

    def _record_index_at_line(self, line: int) -> int | None:
        record_index, _local_line = self._record_at_line(line)
        return record_index

    def _record_at_line(self, line: int) -> tuple[int | None, int]:
        if line < 0:
            return None, 0
        remaining = line
        for block_index, block_lines in enumerate(self._line_block_sums):
            if remaining >= block_lines:
                remaining -= block_lines
                continue
            start = block_index * _LINE_BLOCK_SIZE
            stop = min(len(self._record_line_counts), start + _LINE_BLOCK_SIZE)
            for index in range(start, stop):
                line_count = self._record_line_counts[index]
                if remaining < line_count:
                    return index, remaining
                remaining -= line_count
            break
        return None, 0

    def _copy_record(self, record: TranscriptRecord) -> None:
        self.app.copy_to_clipboard(record.text)
        self.notify(f"{record_label(record.kind, record.text)} message copied")

    def _toggle_record(self, index: int) -> None:
        record = self.records[index]
        record.expanded = not record.expanded
        self._strip_cache.pop(index, None)
        self._set_line_count(index, self._estimate_record_lines(record))
        self._sync_virtual_size()
        self.refresh()
        self._maybe_scroll_end()

    def on_click(self, event: events.Click) -> None:
        if event.widget is not self or event.button not in (1, 3):
            return
        content_offset = event.get_content_offset(self)
        if content_offset is None:
            return
        record_index, local_line = self._record_at_line(
            self.scroll_offset.y + content_offset.y
        )
        if record_index is None:
            return
        record = self.records[record_index]
        copy_clicked = content_offset.x >= copy_hit_start(self._content_width())
        if event.button == 3 or (event.button == 1 and local_line == 0 and copy_clicked):
            self._copy_record(record)
        elif event.button == 1 and not is_dialogue_record(record.kind) and (
            not record.expanded or local_line == 0
        ):
            self._toggle_record(record_index)

    def mark_history_window(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        """Mark the persisted-record slice used by the history pager."""

        position = len(self.records) if start is None else max(0, min(start, len(self.records)))
        self._history_start = position
        self._history_end = (
            position
            if end is None
            else max(position, min(end, len(self.records)))
        )

    @property
    def history_window(self) -> tuple[int, int] | None:
        """Return the current persisted-record slice, if one was marked."""

        if self._history_start is None or self._history_end is None:
            return None
        return self._history_start, self._history_end

    async def prepend_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> int:
        """Insert older records while keeping the first visible row anchored."""

        records = [_record_from_history_item(item) for item in items]
        if not records:
            return 0
        added_chars = sum(len(record.text) for record in records)
        if self._max_chars > 0 and self._record_chars + added_chars > self._max_chars:
            return 0
        insert_at = self._history_start if self._history_start is not None else 0
        old_scroll_y = round(self.scroll_y)
        was_at_bottom = self._stick_to_bottom and old_scroll_y > 0
        added_lines = self._insert_records(insert_at, records, sync=False)
        if self._history_start is None:
            self._history_start = insert_at
            self._history_end = insert_at + len(records)
        else:
            self._history_end = (self._history_end or self._history_start) + len(records)
        self._sync_virtual_size()
        self._strip_cache.clear()
        self._stick_to_bottom = False
        self._anchored = False
        self._anchor_released = True
        if was_at_bottom:
            self._stick_to_bottom = True
            self._maybe_scroll_end()
        else:
            target_y = old_scroll_y + added_lines
            self.scroll_to(
                y=target_y,
                animate=False,
                immediate=True,
                force=True,
            )
            self.call_after_refresh(self._restore_prepend_position, target_y)
        self.refresh()
        return added_lines

    def _restore_prepend_position(self, target_y: int) -> None:
        if self._stick_to_bottom:
            return
        self.scroll_to(
            y=target_y,
            animate=False,
            immediate=True,
            force=True,
        )

    async def replace_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> None:
        """Replace the bounded history window while leaving live rows intact."""

        if self._history_start is None:
            self.mark_history_window()
        assert self._history_start is not None
        start = self._history_start
        end = self._history_end or start
        removed = self.records[start:end]
        del self.records[start:end]
        del self._record_line_counts[start:end]
        self._record_chars -= sum(len(record.text) for record in removed)
        replacement = [_record_from_history_item(item) for item in items]
        if self._max_chars > 0:
            available = max(0, self._max_chars - self._record_chars)
            replacement_chars = sum(len(record.text) for record in replacement)
            while replacement and replacement_chars > available:
                removable = next(
                    (
                        index
                        for index, record in enumerate(replacement)
                        if not is_dialogue_record(record.kind)
                    ),
                    None,
                )
                if removable is None:
                    break
                replacement_chars -= len(replacement[removable].text)
                del replacement[removable]
        self._strip_cache.clear()
        self._rebuild_line_blocks()
        self._insert_records(start, replacement, sync=False)
        self._history_start = start
        self._history_end = start + len(replacement)
        self._sync_virtual_size()
        self._stick_to_bottom = False
        self._anchored = False
        self._anchor_released = True
        self._top_event_armed = False
        self._bottom_event_armed = False
        self.scroll_to(y=0, animate=False, immediate=True, force=True)
        self.refresh()

    async def append_block(self, kind: str, text: str) -> None:
        self.finish_stream()
        self._append_record(_new_record(kind, text))
        self.refresh()
        self._maybe_scroll_end()

    async def append_blocks(self, items: Sequence[tuple[str, str]]) -> None:
        self.finish_stream()
        for kind, text in items:
            self._append_record(_new_record(kind, text), sync=False)
        self._sync_virtual_size()
        self.refresh()
        self._maybe_scroll_end()

    async def append_stream(
        self,
        kind: str,
        fragment: str,
        *,
        replace: bool = False,
    ) -> None:
        if not fragment:
            return
        if self._stream_kind != kind or self._stream_record is None:
            self.finish_stream()
            record = _new_record(kind, fragment)
            self._append_record(record)
            self._stream_kind = kind
            self._stream_record = record
        else:
            updated = fragment if replace else self._stream_record.text + fragment
            if updated == self._stream_record.text:
                return
            self._record_chars += len(updated) - len(self._stream_record.text)
            self._stream_record.text = updated
            self._replace_last_strips(sync=False)
            self._trim_records()
            self._sync_virtual_size()
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
        self._line_block_sums.clear()
        self._strip_cache.clear()
        self._record_chars = 0
        self._history_start = None
        self._history_end = None
        self._top_event_armed = True
        self._bottom_event_armed = True
        self._sync_virtual_size()
        self._stick_to_bottom = True
        self._anchored = True
        self._anchor_released = False
        self.refresh()

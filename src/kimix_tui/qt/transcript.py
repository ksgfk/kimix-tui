"""Virtualized chat transcript: list model + custom delegate, no per-row widgets."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import (
    QAbstractListModel,
    QEvent,
    QModelIndex,
    QPoint,
    QPointF,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QListView,
    QMenu,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from kimix_tui.qt.paint import RecordLayout, layout_record, qcolor
from kimix_tui.qt.theme import COLORS
from kimix_tui.transcript_layout import (
    COPY_SUFFIX,
    cell_len,
    copy_hit_start,
    default_expanded,
    is_compact_record,
    is_dialogue_record,
    record_label,
)

MAX_TRANSCRIPT_CHARS = 64 * 1024 * 1024
_TRIM_TARGET_RATIO = 0.9
_DOCUMENT_CACHE_SIZE = 32
_PAD_X = 12
_PAD_Y = 8
_BAR_WIDTH = 3
_HEADER_HEIGHT = 22
_LINE_HEIGHT = 18
_COPY_WIDTH = 36


@dataclass(slots=True)
class TranscriptRecord:
    kind: str
    text: str
    expanded: bool = False
    turn: int | None = None


def _new_record(kind: str, text: str, *, turn: int | None = None) -> TranscriptRecord:
    return TranscriptRecord(kind, text, expanded=default_expanded(kind), turn=turn)


@dataclass(slots=True)
class BodySelection:
    """Character range inside one transcript body, used for mouse copy."""

    row: int
    anchor: int
    position: int

    @property
    def is_empty(self) -> bool:
        return self.anchor == self.position

    def normalized(self) -> tuple[int, int]:
        if self.anchor <= self.position:
            return self.anchor, self.position
        return self.position, self.anchor


def _record_from_history_item(
    item: tuple[str, str] | tuple[str, str, int],
) -> TranscriptRecord:
    kind, text = item[0], item[1]
    turn = item[2] if len(item) > 2 else None
    return _new_record(kind, text, turn=turn)


class TranscriptModel(QAbstractListModel):
    """Owns transcript rows. Painting is delegated; this only stores records."""

    def __init__(self, parent: QWidget | None = None, *, max_chars: int = MAX_TRANSCRIPT_CHARS) -> None:
        super().__init__(parent)
        self.records: list[TranscriptRecord] = []
        self._max_chars = max(0, max_chars)
        self._record_chars = 0
        self._omitted_records = 0
        self._history_start: int | None = None
        self._history_end: int | None = None
        self._stream_kind: str | None = None
        self._stream_record: TranscriptRecord | None = None

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self.records)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < len(self.records)):
            return None
        record = self.records[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return record.text
        if role == Qt.ItemDataRole.UserRole:
            return record
        return None

    @property
    def omitted_records(self) -> int:
        return self._omitted_records

    @property
    def history_window(self) -> tuple[int, int] | None:
        if self._history_start is None or self._history_end is None:
            return None
        return self._history_start, self._history_end

    def mark_history_window(self, start: int | None = None, end: int | None = None) -> None:
        position = len(self.records) if start is None else max(0, min(start, len(self.records)))
        self._history_start = position
        self._history_end = (
            position if end is None else max(position, min(end, len(self.records)))
        )

    def finish_stream(self) -> None:
        self._stream_kind = None
        self._stream_record = None

    def append_block(self, kind: str, text: str) -> None:
        self.finish_stream()
        self._insert_records(len(self.records), [_new_record(kind, text)], trim=True)

    def append_blocks(self, items: Sequence[tuple[str, str]]) -> None:
        self.finish_stream()
        self._insert_records(
            len(self.records),
            [_new_record(kind, text) for kind, text in items],
            trim=True,
        )

    def append_stream(self, kind: str, fragment: str, *, replace: bool = False) -> None:
        if not fragment:
            return
        if self._stream_kind != kind or self._stream_record is None:
            self.finish_stream()
            record = _new_record(kind, fragment)
            self._insert_records(len(self.records), [record], trim=True)
            self._stream_kind = kind
            self._stream_record = record
            return
        updated = fragment if replace else self._stream_record.text + fragment
        if updated == self._stream_record.text:
            return
        self._record_chars += len(updated) - len(self._stream_record.text)
        self._stream_record.text = updated
        self._trim_records()
        row = len(self.records) - 1
        index = self.index(row)
        self.dataChanged.emit(index, index)

    def prepend_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> int:
        records = [_record_from_history_item(item) for item in items]
        if not records:
            return 0
        added_chars = sum(len(record.text) for record in records)
        if self._max_chars > 0 and self._record_chars + added_chars > self._max_chars:
            return 0
        insert_at = self._history_start if self._history_start is not None else 0
        added_lines = sum(_estimate_lines(record, 80) for record in records)
        self._insert_records(insert_at, records, trim=False)
        if self._history_start is None:
            self._history_start = insert_at
            self._history_end = insert_at + len(records)
        else:
            self._history_end = (self._history_end or self._history_start) + len(records)
        return added_lines

    def replace_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> None:
        if self._history_start is None:
            self.mark_history_window()
        assert self._history_start is not None
        start = self._history_start
        end = self._history_end or start
        self.beginResetModel()
        removed = self.records[start:end]
        del self.records[start:end]
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
        self.records[start:start] = replacement
        self._record_chars += sum(len(record.text) for record in replacement)
        self._history_start = start
        self._history_end = start + len(replacement)
        self.endResetModel()

    def clear_messages(self) -> None:
        self.beginResetModel()
        self.finish_stream()
        self.records.clear()
        self._record_chars = 0
        self._history_start = None
        self._history_end = None
        self.endResetModel()

    def toggle_expanded(self, row: int) -> None:
        if not (0 <= row < len(self.records)):
            return
        record = self.records[row]
        if is_dialogue_record(record.kind):
            return
        record.expanded = not record.expanded
        index = self.index(row)
        self.dataChanged.emit(index, index)

    def _insert_records(
        self,
        index: int,
        records: Sequence[TranscriptRecord],
        *,
        trim: bool,
    ) -> None:
        if not records:
            return
        index = max(0, min(index, len(self.records)))
        self.beginInsertRows(QModelIndex(), index, index + len(records) - 1)
        self.records[index:index] = list(records)
        self._record_chars += sum(len(record.text) for record in records)
        self.endInsertRows()
        if trim:
            self._trim_records()

    def _trim_records(self) -> None:
        if self._history_start is not None:
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
        self.beginRemoveRows(QModelIndex(), 0, remove_count - 1)
        del self.records[:remove_count]
        self.endRemoveRows()
        self._record_chars -= removed_chars
        self._omitted_records += remove_count
        if self._stream_record is not None and self._stream_record not in self.records:
            self.finish_stream()


def _estimate_lines(record: TranscriptRecord, width: int) -> int:
    if is_compact_record(record.kind, expanded=record.expanded):
        return 1
    wrap_width = max(8, width - 4)
    lines = record.text.split("\n") if record.text else [""]
    body_lines = 0
    for line in lines:
        cells = max(1, len(line))
        body_lines += max(1, (cells + wrap_width - 1) // wrap_width)
    return 1 + body_lines + 1


def _layout_for(record: TranscriptRecord, width: int) -> RecordLayout:
    return layout_record(
        record.kind,
        record.text,
        width=max(24, width // 8),
        expanded=record.expanded,
    )


def _body_rect(card: QRect) -> QRect:
    return QRect(
        card.left() + _PAD_X,
        card.top() + _HEADER_HEIGHT,
        card.width() - _PAD_X * 2,
        card.height() - _HEADER_HEIGHT - _PAD_Y,
    )


class TranscriptDelegate(QStyledItemDelegate):
    """Paints one message card. Documents are created lazily and LRU-cached."""

    def __init__(self, view: Transcript) -> None:
        super().__init__(view)
        self._view = view
        self._documents: OrderedDict[tuple[object, ...], QTextDocument] = OrderedDict()

    def invalidate(self) -> None:
        self._documents.clear()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return QSize(option.rect.width(), _HEADER_HEIGHT)
        width = max(120, option.rect.width() or self._view.viewport().width())
        if is_compact_record(record.kind, expanded=record.expanded):
            return QSize(width, _HEADER_HEIGHT + _PAD_Y)
        body_height = self._body_height(record, width)
        return QSize(width, _HEADER_HEIGHT + body_height + _PAD_Y * 2)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return
        rect = option.rect.adjusted(8, 4, -8, -4)
        layout = layout_record(
            record.kind,
            record.text,
            width=max(24, rect.width() // 8),
            expanded=record.expanded,
        )
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bar = QRect(rect.left(), rect.top() + 6, _BAR_WIDTH, rect.height() - 12)
        painter.fillRect(bar, qcolor(layout.bar_color))
        header_rect = QRect(
            rect.left() + _PAD_X,
            rect.top() + 4,
            rect.width() - _PAD_X - _COPY_WIDTH,
            _HEADER_HEIGHT,
        )
        painter.setPen(qcolor(layout.bar_color if layout.bar_color != "muted" else "cyan"))
        if layout.bar_color == "muted":
            painter.setPen(QColor(COLORS["muted"]))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            header_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            layout.header,
        )
        copy_rect = QRect(rect.right() - _COPY_WIDTH, rect.top() + 4, _COPY_WIDTH, _HEADER_HEIGHT)
        painter.setPen(QColor(COLORS["muted"]))
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(copy_rect, Qt.AlignmentFlag.AlignCenter, "⧉")
        if not layout.compact and layout.body:
            body_rect = _body_rect(rect)
            document = self._document(index.row(), record, body_rect.width(), layout)
            painter.translate(body_rect.topLeft())
            painter.setClipRect(QRect(0, 0, body_rect.width(), body_rect.height()))
            ctx = QAbstractTextDocumentLayout.PaintContext()
            color = QColor(
                COLORS["muted"] if not is_dialogue_record(record.kind) else COLORS["text"]
            )
            ctx.palette.setColor(QPalette.ColorRole.Text, color)
            ctx.palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
            ctx.palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 0))
            selection = self._view._selection
            if (
                selection is not None
                and selection.row == index.row()
                and not selection.is_empty
            ):
                extra = QAbstractTextDocumentLayout.Selection()
                cursor = QTextCursor(document)
                start, end = selection.normalized()
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                extra.cursor = cursor
                fmt = QTextCharFormat()
                fmt.setBackground(QColor(COLORS["accent"]))
                fmt.setForeground(QColor("#042f2e"))
                extra.format = fmt
                ctx.selections = [extra]
            document.documentLayout().draw(painter, ctx)
        painter.restore()

    def copy_hit(self, option_rect: QRect, pos: QPoint) -> bool:
        rect = option_rect.adjusted(8, 4, -8, -4)
        copy_rect = QRect(rect.right() - _COPY_WIDTH, rect.top() + 4, _COPY_WIDTH, _HEADER_HEIGHT)
        return copy_rect.contains(pos)

    def header_hit(self, option_rect: QRect, pos: QPoint) -> bool:
        rect = option_rect.adjusted(8, 4, -8, -4)
        header_rect = QRect(rect.left(), rect.top(), rect.width(), _HEADER_HEIGHT + 4)
        return header_rect.contains(pos) and not self.copy_hit(option_rect, pos)

    def body_hit(
        self,
        option_rect: QRect,
        pos: QPoint,
        index: QModelIndex,
        *,
        clamp: bool = False,
    ) -> int | None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return None
        layout = _layout_for(record, option_rect.width())
        if layout.compact or not layout.body:
            return None
        body_rect = _body_rect(option_rect.adjusted(8, 4, -8, -4))
        if not body_rect.contains(pos):
            if not clamp or not body_rect.isValid() or body_rect.width() <= 0 or body_rect.height() <= 0:
                return None
            pos = QPoint(
                min(max(pos.x(), body_rect.left()), body_rect.right() - 1),
                min(max(pos.y(), body_rect.top()), body_rect.bottom() - 1),
            )
        document = self._document(index.row(), record, body_rect.width(), layout)
        local = QPointF(pos.x() - body_rect.left(), pos.y() - body_rect.top())
        hit = document.documentLayout().hitTest(local, Qt.HitTestAccuracy.FuzzyHit)
        if hit < 0:
            return None
        return min(hit, max(0, document.characterCount() - 1))

    def document_for(self, option_rect: QRect, index: QModelIndex) -> QTextDocument | None:
        record = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record, TranscriptRecord):
            return None
        layout = _layout_for(record, option_rect.width())
        if layout.compact or not layout.body:
            return None
        body_rect = _body_rect(option_rect.adjusted(8, 4, -8, -4))
        return self._document(index.row(), record, body_rect.width(), layout)

    def _body_height(self, record: TranscriptRecord, width: int) -> int:
        inner = max(40, width - _PAD_X * 2 - 16)
        layout = _layout_for(record, width)
        if layout.compact or not layout.body:
            return 0
        document = self._document(-1, record, inner, layout)
        return max(_LINE_HEIGHT, int(document.size().height()))

    def _document(
        self,
        row: int,
        record: TranscriptRecord,
        width: int,
        layout: RecordLayout,
    ) -> QTextDocument:
        key = (row, record.expanded, width, layout.body, layout.markdown, layout.italic_body)
        document = self._documents.pop(key, None)
        if document is None:
            document = QTextDocument()
            font = QApplication.font() if QApplication.instance() else self._view.font()
            font.setItalic(layout.italic_body)
            document.setDefaultFont(font)
            document.setDefaultStyleSheet(
                "body, p, pre, code { background-color: transparent; }"
            )
            if layout.markdown:
                document.setMarkdown(layout.body)
            else:
                document.setPlainText(layout.body)
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            document.setDefaultTextOption(option)
            document.setDocumentMargin(0)
            document.setTextWidth(max(40, width))
        self._documents[key] = document
        while len(self._documents) > _DOCUMENT_CACHE_SIZE:
            self._documents.popitem(last=False)
        return document


class Transcript(QListView):
    """Scrollable chat log that virtualizes painting and bounds memory."""

    reached_top = Signal()
    reached_bottom = Signal()
    viewport_turn_changed = Signal(object)
    record_copied = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        max_chars: int = MAX_TRANSCRIPT_CHARS,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("transcript")
        self._model = TranscriptModel(self, max_chars=max_chars)
        self._delegate = TranscriptDelegate(self)
        self.setModel(self._model)
        self.setItemDelegate(self._delegate)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSpacing(2)
        self.setUniformItemSizes(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._stick_to_bottom = True
        self._top_event_armed = True
        self._bottom_event_armed = True
        self._last_viewport_turn: int | None = None
        self._wrap_width = 0
        self._selection: BodySelection | None = None
        self._selecting = False
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.viewport().installEventFilter(self)

    @property
    def records(self) -> list[TranscriptRecord]:
        return self._model.records

    @property
    def omitted_records(self) -> int:
        return self._model.omitted_records

    @property
    def history_window(self) -> tuple[int, int] | None:
        return self._model.history_window

    @property
    def pinned_to_latest(self) -> bool:
        return self._stick_to_bottom

    @property
    def _record_line_counts(self) -> list[int]:
        width = max(8, self.viewport().width() // 8 or 80)
        return [_estimate_lines(record, width) for record in self.records]

    @property
    def _strip_cache(self) -> OrderedDict[int, object]:
        return self._delegate._documents  # type: ignore[return-value]

    def _content_width(self) -> int:
        return max(8, self.viewport().width() // 8 or 80)

    def mark_history_window(self, start: int | None = None, end: int | None = None) -> None:
        self._model.mark_history_window(start, end)

    def finish_stream(self) -> None:
        self._model.finish_stream()

    def append_block(self, kind: str, text: str) -> None:
        self._model.append_block(kind, text)
        self._maybe_scroll_end()
        self._notify_viewport_turn()

    def append_blocks(self, items: Sequence[tuple[str, str]]) -> None:
        self._model.append_blocks(items)
        self._maybe_scroll_end()
        self._notify_viewport_turn()

    def append_stream(
        self,
        kind: str,
        fragment: str,
        *,
        replace: bool = False,
    ) -> None:
        self._model.append_stream(kind, fragment, replace=replace)
        self._maybe_scroll_end()

    def prepend_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> int:
        old_scroll = self.verticalScrollBar().value()
        was_at_bottom = self._stick_to_bottom and old_scroll > 0
        added_lines = self._model.prepend_history_blocks(items)
        added_px = added_lines * _LINE_HEIGHT
        if was_at_bottom:
            self._stick_to_bottom = True
            self._maybe_scroll_end()
        else:
            self._stick_to_bottom = False
            self.verticalScrollBar().setValue(old_scroll + added_px)
        return added_lines

    def replace_history_blocks(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, int]],
    ) -> None:
        self._delegate.invalidate()
        self._model.replace_history_blocks(items)
        self._stick_to_bottom = False
        self._top_event_armed = False
        self._bottom_event_armed = False
        self.verticalScrollBar().setValue(0)
        self._notify_viewport_turn()

    def clear_messages(self) -> None:
        self._delegate.invalidate()
        self._model.clear_messages()
        self._stick_to_bottom = True
        self._top_event_armed = True
        self._bottom_event_armed = True

    def jump_to_latest(self) -> None:
        self._stick_to_bottom = True
        self._top_event_armed = True
        self._bottom_event_armed = False
        self._maybe_scroll_end()
        self._notify_viewport_turn()

    def jump_to_turn(self, turn: int) -> None:
        start = self._model._history_start or 0
        end = self._model._history_end or len(self.records)
        target_index = None
        for index in range(start, end):
            if self.records[index].turn == turn:
                target_index = index
                break
        if target_index is None:
            return
        self._stick_to_bottom = False
        self._top_event_armed = False
        self._bottom_event_armed = False
        self.scrollTo(self._model.index(target_index), QAbstractItemView.ScrollHint.PositionAtTop)
        self._stick_to_bottom = False
        bar = self.verticalScrollBar()
        if bar.value() < bar.maximum():
            self._bottom_event_armed = True
        self._notify_viewport_turn()

    def viewport_turn(self) -> int | None:
        index = self.indexAt(QPoint(12, 4))
        if not index.isValid():
            return None
        record = self.records[index.row()]
        return record.turn

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched is self.viewport():
            if isinstance(event, QMouseEvent):
                handled = self._handle_mouse_event(event)
                if handled:
                    return True
            if event.type() == QEvent.Type.Leave:
                self.viewport().unsetCursor()
        return super().eventFilter(watched, event)  # type: ignore[arg-type]

    def keyPressEvent(self, event: object) -> None:
        if (
            isinstance(event, QKeyEvent)
            and event.matches(QKeySequence.StandardKey.Copy)
            and self._copy_selection()
        ):
            event.accept()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def _handle_mouse_event(self, event: QMouseEvent) -> bool:
        etype = event.type()
        if etype == QEvent.Type.MouseButtonDblClick:
            return self._handle_mouse_double_click(event)
        if etype == QEvent.Type.MouseMove:
            return self._handle_mouse_move(event)
        if etype == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._selecting:
                self._selecting = False
                return True
            return False
        if etype == QEvent.Type.MouseButtonPress:
            return self._handle_mouse_press(event)
        return False

    def _handle_mouse_press(self, event: QMouseEvent) -> bool:
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        if not index.isValid():
            self._clear_selection()
            return False
        record = self.records[index.row()]
        option_rect = self.visualRect(index)
        copy_clicked = self._delegate.copy_hit(option_rect, pos)
        header = self._delegate.header_hit(option_rect, pos)
        if event.button() == Qt.MouseButton.RightButton:
            if copy_clicked:
                self._copy_record(record)
                return True
            if self.selected_text():
                self._popup_copy_menu(event.globalPosition().toPoint())
            return True
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if copy_clicked:
            self._copy_record(record)
            return True
        if not is_dialogue_record(record.kind) and (not record.expanded or header):
            self._model.toggle_expanded(index.row())
            self._delegate.invalidate()
            self.updateGeometries()
            self._maybe_scroll_end()
            self._clear_selection()
            return True
        hit = self._delegate.body_hit(option_rect, pos, index)
        if hit is None:
            self._clear_selection()
            return False
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._selection = BodySelection(index.row(), hit, hit)
        self._selecting = True
        self.viewport().update()
        return True

    def _handle_mouse_double_click(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        if not index.isValid():
            return False
        option_rect = self.visualRect(index)
        if self._delegate.copy_hit(option_rect, pos):
            return True
        hit = self._delegate.body_hit(option_rect, pos, index)
        document = self._delegate.document_for(option_rect, index)
        if hit is None or document is None:
            return False
        cursor = QTextCursor(document)
        cursor.setPosition(min(hit, max(0, document.characterCount() - 1)))
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        self._selection = BodySelection(
            index.row(),
            cursor.selectionStart(),
            cursor.selectionEnd(),
        )
        self._selecting = False
        self.viewport().update()
        return True

    def _handle_mouse_move(self, event: QMouseEvent) -> bool:
        pos = event.position().toPoint()
        if self._selecting and self._selection is not None:
            index = self._model.index(self._selection.row)
            if index.isValid():
                hit = self._delegate.body_hit(self.visualRect(index), pos, index, clamp=True)
                if hit is not None:
                    self._selection.position = hit
                    self.viewport().update()
            return True
        self._sync_cursor(pos)
        return False

    def _sync_cursor(self, pos: QPoint) -> None:
        index = self.indexAt(pos)
        if not index.isValid():
            self.viewport().unsetCursor()
            return
        option_rect = self.visualRect(index)
        if self._delegate.copy_hit(option_rect, pos):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            return
        record = self.records[index.row()]
        if not is_dialogue_record(record.kind) and self._delegate.header_hit(option_rect, pos):
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            return
        if self._delegate.body_hit(option_rect, pos, index) is not None:
            self.viewport().setCursor(Qt.CursorShape.IBeamCursor)
            return
        self.viewport().unsetCursor()

    def selected_text(self) -> str:
        selection = self._selection
        if selection is None or selection.is_empty:
            return ""
        index = self._model.index(selection.row)
        if not index.isValid():
            return ""
        document = self._delegate.document_for(self.visualRect(index), index)
        if document is None:
            return ""
        start, end = selection.normalized()
        limit = max(0, document.characterCount() - 1)
        start = max(0, min(start, limit))
        end = max(0, min(end, limit))
        cursor = QTextCursor(document)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText().replace("\u2029", "\n")

    def _copy_selection(self) -> bool:
        text = self.selected_text()
        if not text:
            return False
        QApplication.clipboard().setText(text)
        return True

    def _clear_selection(self) -> None:
        self._selecting = False
        if self._selection is not None:
            self._selection = None
            self.viewport().update()

    def _popup_copy_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        action = menu.addAction("Copy")
        action.triggered.connect(self._copy_selection)
        menu.popup(global_pos)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        width = self._content_width()
        if width != self._wrap_width:
            self._wrap_width = width
            self._delegate.invalidate()
            self.scheduleDelayedItemsLayout()
            self._maybe_scroll_end()

    def _copy_record(self, record: TranscriptRecord) -> None:
        QApplication.clipboard().setText(record.text)
        self.record_copied.emit(record_label(record.kind, record.text))

    def _maybe_scroll_end(self) -> None:
        if not self._stick_to_bottom:
            return
        self.scrollToBottom()

    def _on_scroll(self, value: int) -> None:
        bar = self.verticalScrollBar()
        at_top = value <= 0
        at_bottom = value >= bar.maximum()
        if at_bottom and value > 0:
            self._stick_to_bottom = True
        elif not at_bottom:
            self._stick_to_bottom = False
        if at_top:
            if self._top_event_armed and not at_bottom:
                self._top_event_armed = False
                self.reached_top.emit()
        else:
            self._top_event_armed = True
        if at_bottom:
            if self._bottom_event_armed and not at_top:
                self._bottom_event_armed = False
                self.reached_bottom.emit()
        else:
            self._bottom_event_armed = True
        self._notify_viewport_turn()

    def _notify_viewport_turn(self) -> None:
        turn = self.viewport_turn()
        if turn != self._last_viewport_turn:
            self._last_viewport_turn = turn
            self.viewport_turn_changed.emit(turn)

    def visible_text(self) -> str:
        """Plain text of currently visible rows (test helper)."""

        lines: list[str] = []
        viewport = self.viewport().rect()
        for row in range(len(self.records)):
            rect = self.visualRect(self._model.index(row))
            if not rect.intersects(viewport):
                continue
            record = self.records[row]
            layout = layout_record(
                record.kind,
                record.text,
                width=self._content_width(),
                expanded=record.expanded,
            )
            lines.append(layout.header)
            if layout.body:
                lines.append(layout.body)
        return "\n".join(lines)

    def render_line(self, y: int) -> SimpleLine:
        """Compatibility helper: first visible header or a blank line."""

        index = self.indexAt(QPoint(12, max(4, y * _LINE_HEIGHT)))
        if not index.isValid():
            return SimpleLine("")
        record = self.records[index.row()]
        layout = layout_record(
            record.kind,
            record.text,
            width=self._content_width(),
            expanded=record.expanded,
        )
        return SimpleLine(f"{layout.header}{COPY_SUFFIX}")


class SimpleLine:
    """Tiny stand-in for Textual's Strip.text used by a few tests."""

    def __init__(self, text: str) -> None:
        self.text = text


def copy_hit_column(width: int) -> int:
    return copy_hit_start(width)


def compact_header_cells(text: str) -> int:
    return cell_len(text)

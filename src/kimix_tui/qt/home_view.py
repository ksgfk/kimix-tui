"""Home view: session browser, search, batch delete, and details pane."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_tui.llm_config import LLMConfigReference, config_file_available
from kimix_tui.qt.theme import COLORS
from kimix_tui.session_index import (
    SessionSummary,
    format_file_size,
    format_relative_time,
)

SessionConfigLoader = Callable[[str], LLMConfigReference | None]

_MARK_SIZE = 22
_ROW_HEIGHT = 58


class SelectionMark(QCheckBox):
    """Circular checkbox used for batch-selecting sessions."""

    def __init__(self, parent: QWidget | None = None, *, object_name: str = "session-check") -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setText("")
        self.setFixedSize(_MARK_SIZE, _MARK_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._forced = False

    def set_forced(self, forced: bool) -> None:
        self._forced = forced
        self.update()

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        checked = self.isChecked()
        mixed = self.checkState() == Qt.CheckState.PartiallyChecked
        idle = not (checked or mixed or self._forced or self.underMouse() or self.isDown())
        painter.setOpacity(1.0 if not idle else 0.42)
        box = QRectF(2.5, 2.5, 17, 17)
        if checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["accent"]))
            painter.drawEllipse(box)
            pen = QPen(QColor("#042f2e"), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mark = QPainterPath()
            mark.moveTo(7.2, 11.2)
            mark.lineTo(10.0, 14.1)
            mark.lineTo(15.2, 8.0)
            painter.drawPath(mark)
            return
        border = QColor(COLORS["accent"] if mixed or self.underMouse() or self.isDown() else COLORS["muted"])
        painter.setPen(QPen(border, 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(box)
        if mixed:
            dash = QPen(QColor(COLORS["accent"]), 2.0)
            dash.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(dash)
            painter.drawLine(7, 11, 15, 11)


class SessionRow(QWidget):
    """Selectable session card: click previews, the mark batch-selects."""

    check_toggled = Signal(str)
    opened = Signal(str)

    def __init__(self, summary: SessionSummary, *, selected: bool = False) -> None:
        super().__init__()
        self.summary = summary
        self.selected = selected
        self._active = False
        self._hovered = False
        self._selection_mode = False
        self.setObjectName("session-row")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 12, 8)
        layout.setSpacing(10)
        self._check = SelectionMark(self)
        self._check.setChecked(selected)
        self._check.clicked.connect(self._emit_check)
        layout.addWidget(self._check, 0, Qt.AlignmentFlag.AlignVCenter)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        title = QLabel(summary.title)
        title.setObjectName("session-title")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        meta = QLabel(
            f"{format_relative_time(summary.updated_at)} · {format_file_size(summary.size_bytes)}"
        )
        meta.setObjectName("session-meta")
        meta.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text.addWidget(title)
        text.addWidget(meta)
        layout.addLayout(text, 1)
        self._badge = QLabel()
        self._badge.setObjectName("session-badge")
        self._badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if summary.is_archived:
            self._badge.setText("Archived")
            self._badge.setProperty("kind", "archived")
        elif summary.is_last:
            self._badge.setText("Last")
            self._badge.setProperty("kind", "last")
        else:
            self._badge.hide()
        self._badge.style().unpolish(self._badge)
        self._badge.style().polish(self._badge)
        layout.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set_selected(selected)

    @property
    def checked(self) -> bool:
        return self._check.isChecked()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._check.blockSignals(True)
        self._check.setChecked(selected)
        self._check.blockSignals(False)
        self._check.set_forced(self._selection_mode or self._hovered or selected)
        self._check.update()

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.update()

    def set_selection_mode(self, active: bool) -> None:
        self._selection_mode = active
        self._check.set_forced(active or self._hovered or self.selected)

    def _emit_check(self) -> None:
        self.check_toggled.emit(self.summary.id)

    def enterEvent(self, event: object) -> None:
        self._hovered = True
        self._check.set_forced(True)
        self.update()
        super().enterEvent(event)  # type: ignore[arg-type]

    def leaveEvent(self, event: object) -> None:
        self._hovered = False
        self._check.set_forced(self._selection_mode or self.selected)
        self.update()
        super().leaveEvent(event)  # type: ignore[arg-type]

    def mousePressEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if self.childAt(event.position().toPoint()) is self._check:
            return
        list_widget = self._list_widget()
        if list_widget is not None:
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if list_widget.itemWidget(item) is self:
                    list_widget.setCurrentRow(index)
                    break
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        if self.childAt(event.position().toPoint()) is self._check:
            return
        self.opened.emit(self.summary.id)
        event.accept()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._active:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["boost"]))
            painter.drawRoundedRect(rect, 10, 10)
            painter.setBrush(QColor(COLORS["accent"]))
            painter.drawRoundedRect(QRectF(1.5, 14, 3, rect.height() - 28), 1.5, 1.5)
        elif self._hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["panel"]))
            painter.drawRoundedRect(rect, 10, 10)

    def _list_widget(self) -> QListWidget | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QListWidget):
                return parent
            parent = parent.parentWidget()
        return None


class HomeView(QWidget):
    """Browse project sessions or start a new one."""

    new_session = Signal()
    resume_session = Signal(str)
    open_settings = Signal()
    configure_session = Signal(str)
    quit_requested = Signal()
    delete_requested = Signal(list)
    llm_required = Signal(object)

    def __init__(
        self,
        work_dir: Path,
        *,
        default_config: LLMConfigReference,
        session_config_loader: SessionConfigLoader,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("home-view")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._work_dir = work_dir
        self._default_config = default_config
        self._session_config_loader = session_config_loader
        self._summaries: list[SessionSummary] = []
        self._selected_ids: set[str] = set()
        self._narrow = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        toolbar = QFrame()
        toolbar.setObjectName("home-toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(4, 4, 4, 8)
        context = QVBoxLayout()
        context.setSpacing(2)
        title = QLabel("Sessions")
        title.setObjectName("home-title")
        path = QLabel(str(self._work_dir))
        path.setObjectName("home-path")
        self._home_model = QLabel()
        self._home_model.setObjectName("home-model")
        context.addWidget(title)
        context.addWidget(path)
        context.addWidget(self._home_model)
        toolbar_layout.addLayout(context, 1)
        new_btn = QPushButton("New session")
        new_btn.setObjectName("start-new-session")
        self._new_btn = new_btn
        settings = QPushButton("Settings")
        settings.setObjectName("open-settings")
        toolbar_layout.addWidget(new_btn)
        toolbar_layout.addWidget(settings)
        root.addWidget(toolbar)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("home-workspace")
        browser = QWidget()
        browser.setObjectName("session-browser")
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(0, 0, 8, 0)
        browser_layout.setSpacing(8)
        header_bar = QWidget()
        header_bar.setObjectName("history-header")
        header_bar.setFixedHeight(32)
        header = QHBoxLayout(header_bar)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._history_title = QLabel("History")
        self._history_title.setObjectName("history-title")
        self._selection_count = QLabel("0 selected")
        self._selection_count.setObjectName("selection-count")
        self._session_count = QLabel("")
        self._session_count.setObjectName("session-count")
        self._select_shown = QPushButton("Select all")
        self._select_shown.setObjectName("select-shown")
        self._select_shown.setEnabled(False)
        self._select_shown.setFlat(True)
        self._select_shown.setCursor(Qt.CursorShape.PointingHandCursor)
        self._select_shown.setFixedHeight(28)
        self._delete = QPushButton("Delete")
        self._delete.setObjectName("delete-sessions")
        self._delete.setEnabled(False)
        self._delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete.setFixedHeight(28)
        header.addWidget(self._history_title)
        header.addWidget(self._selection_count)
        header.addStretch()
        header.addWidget(self._select_shown)
        header.addWidget(self._delete)
        header.addWidget(self._session_count)
        browser_layout.addWidget(header_bar)
        self._search = QLineEdit()
        self._search.setObjectName("session-search")
        self._search.setPlaceholderText("Search sessions")
        self._search.setClearButtonEnabled(True)
        browser_layout.addWidget(self._search)

        self._status = QLabel("Loading sessions…")
        self._status.setObjectName("home-status")
        browser_layout.addWidget(self._status)
        self._list = QListWidget()
        self._list.setObjectName("session-list")
        self._list.setUniformItemSizes(True)
        self._list.setSpacing(2)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._list.installEventFilter(self)
        self._list.currentRowChanged.connect(self._on_row_changed)
        browser_layout.addWidget(self._list, 1)
        self._splitter.addWidget(browser)

        self._details = QFrame()
        self._details.setObjectName("session-detail")
        details_layout = QVBoxLayout(self._details)
        details_layout.setContentsMargins(18, 16, 18, 16)
        details_layout.setSpacing(8)
        overline = QLabel("Details")
        overline.setObjectName("detail-overline")
        self._detail_title = QLabel("Loading sessions…")
        self._detail_title.setObjectName("detail-title")
        self._detail_title.setWordWrap(True)
        self._detail_state = QLabel("Reading this project")
        self._detail_state.setObjectName("detail-state")
        details_layout.addWidget(overline)
        details_layout.addWidget(self._detail_title)
        details_layout.addWidget(self._detail_state)
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._open = QPushButton("Open session")
        self._open.setObjectName("open-session")
        self._open.setEnabled(False)
        self._configure = QPushButton("Configure")
        self._configure.setObjectName("configure-session")
        self._configure.setEnabled(False)
        actions.addWidget(self._open)
        actions.addWidget(self._configure)
        actions.addStretch()
        details_layout.addLayout(actions)
        self._meta = QWidget()
        self._meta.setObjectName("detail-metadata")
        form = QVBoxLayout(self._meta)
        form.setContentsMargins(0, 8, 0, 0)
        form.setSpacing(8)
        self._detail_values: dict[str, QLabel] = {}
        for key, label in (
            ("detail-updated", "Updated"),
            ("detail-llm", "LLM"),
            ("detail-provider", "Provider"),
            ("detail-config", "Config"),
            ("detail-size", "Size"),
            ("detail-storage", "Storage"),
            ("detail-todos", "Active todos"),
            ("detail-directories", "Extra folders"),
            ("detail-id", "Session ID"),
            ("detail-path", "Folder"),
        ):
            row = QHBoxLayout()
            name = QLabel(label)
            name.setObjectName("detail-key")
            name.setFixedWidth(110)
            value = QLabel("")
            value.setObjectName(key)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._detail_values[key] = value
            row.addWidget(name)
            row.addWidget(value, 1)
            form.addLayout(row)
        details_layout.addWidget(self._meta)
        details_layout.addStretch()
        browser.setMinimumWidth(200)
        self._details.setMinimumWidth(200)
        self._splitter.addWidget(self._details)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([380, 640])
        root.addWidget(self._splitter, 1)

        self._refresh_model_label()
        new_btn.clicked.connect(self.request_new_session)
        settings.clicked.connect(self.open_settings.emit)
        self._open.clicked.connect(self._open_current)
        self._configure.clicked.connect(self._configure_current)
        self._select_shown.clicked.connect(self._toggle_shown)
        self._delete.clicked.connect(self.request_delete)
        self._search.textChanged.connect(self._render_sessions)
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self, self.focus_search)
        search_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._update_selection_controls([])

    @property
    def summary(self) -> SessionSummary | None:
        item = self._list.currentItem()
        row = self._row_of(item) if item is not None else None
        return row.summary if row is not None else None

    @property
    def configuration_available(self) -> bool:
        summary = self.summary
        if summary is None:
            return False
        saved = self._session_config_loader(summary.id)
        return config_file_available(saved or self._default_config)

    def session_rows(self) -> list[SessionRow]:
        rows: list[SessionRow] = []
        for index in range(self._list.count()):
            row = self._row_at(index)
            if row is not None:
                rows.append(row)
        return rows

    def show_sessions(self, summaries: list[SessionSummary]) -> None:
        self._summaries = summaries
        self._selected_ids.intersection_update(summary.id for summary in summaries)
        self._render_sessions()

    def show_load_error(self, message: str) -> None:
        self._list.clear()
        self._list.hide()
        self._status.show()
        self._status.setText(f"Could not load sessions: {message}")
        self._session_count.setText("Unavailable")
        self._show_empty("Sessions unavailable", "Start a new session to continue")
        self._update_selection_controls([])

    def refresh_configuration(self, default_config: LLMConfigReference) -> None:
        self._default_config = default_config
        self._refresh_model_label()
        if self.summary is not None:
            self._show_session(self.summary)

    def request_new_session(self) -> None:
        if not config_file_available(self._default_config):
            self.llm_required.emit(None)
            return
        self.new_session.emit()

    def open_highlighted(self) -> None:
        summary = self.summary
        if summary is not None:
            self._open_summary(summary)

    def focus_search(self) -> None:
        self._search.setFocus()

    def toggle_selected(self) -> None:
        item = self._list.currentItem()
        row = self._row_of(item) if item is not None else None
        if row is not None:
            self._toggle_row(row)

    def request_delete(self) -> None:
        if self._selected_ids:
            self.delete_requested.emit(sorted(self._selected_ids))

    def apply_deleted(self, ids: list[str]) -> None:
        id_set = set(ids)
        self._summaries = [summary for summary in self._summaries if summary.id not in id_set]
        self._selected_ids.difference_update(id_set)
        self._render_sessions()

    def keyPressEvent(self, event: object) -> None:
        if not isinstance(event, QKeyEvent):
            super().keyPressEvent(event)  # type: ignore[arg-type]
            return
        if self._handle_shortcut(event):
            event.accept()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def eventFilter(self, watched: object, event: object) -> bool:
        if (
            watched is self._list
            and isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and self._handle_shortcut(event)
        ):
            return True
        return super().eventFilter(watched, event)  # type: ignore[arg-type]

    def _handle_shortcut(self, event: QKeyEvent) -> bool:
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_N and mods == Qt.KeyboardModifier.NoModifier:
            self.request_new_session()
            return True
        if key == Qt.Key.Key_Q and mods == Qt.KeyboardModifier.NoModifier:
            self.quit_requested.emit()
            return True
        if key == Qt.Key.Key_Space and mods == Qt.KeyboardModifier.NoModifier:
            self.toggle_selected()
            return True
        if key == Qt.Key.Key_Delete:
            self.request_delete()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and mods == Qt.KeyboardModifier.NoModifier:
            self.open_highlighted()
            return True
        return False

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        self._sync_narrow(self.width())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._sync_narrow(self.width())

    def _sync_narrow(self, width: int) -> None:
        win = self.window()
        measured = min(width, win.width()) if win is not None else width
        narrow = measured < 780
        if narrow == self._narrow and self._splitter.orientation() == (
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        ):
            return
        self._narrow = narrow
        self._splitter.setOrientation(
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        )

    def _refresh_model_label(self) -> None:
        available = config_file_available(self._default_config)
        status = "" if available else " · missing"
        self._home_model.setText(f"New sessions · {self._default_config.label}{status}")

    def _filtered(self) -> list[SessionSummary]:
        query = self._search.text().strip().casefold()
        return [
            summary
            for summary in self._summaries
            if not query or query in summary.title.casefold()
        ]

    def _render_sessions(self) -> None:
        filtered = self._filtered()
        current_id = self.summary.id if self.summary is not None else None
        self._list.clear()
        total = len(self._summaries)
        query = self._search.text().strip()
        self._session_count.setText(
            f"{len(filtered)} of {total}" if query else f"{total} total"
        )
        if not filtered:
            self._list.hide()
            self._status.show()
            if total and query:
                self._status.setText("No sessions match this title")
                self._show_empty("No matching sessions", "Try another title")
            else:
                self._status.setText("No saved sessions in this folder")
                self._show_empty("No sessions yet", "Start a new session in this folder")
            self._update_selection_controls(filtered)
            if not self._search.hasFocus():
                self._new_btn.setFocus()
            return
        self._status.hide()
        self._list.show()
        select_row = 0
        for index, summary in enumerate(filtered):
            row = SessionRow(summary, selected=summary.id in self._selected_ids)
            row.check_toggled.connect(self._on_mark_clicked)
            row.opened.connect(self._open_id)
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, _ROW_HEIGHT))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            if summary.id == current_id:
                select_row = index
        self._list.setCurrentRow(select_row)
        self._update_selection_controls(filtered)
        if not self._search.hasFocus():
            self._list.setFocus()

    def _row_at(self, index: int) -> SessionRow | None:
        item = self._list.item(index)
        widget = self._list.itemWidget(item) if item is not None else None
        return widget if isinstance(widget, SessionRow) else None

    def _row_of(self, item: QListWidgetItem | None) -> SessionRow | None:
        if item is None:
            return None
        widget = self._list.itemWidget(item)
        return widget if isinstance(widget, SessionRow) else None

    def _on_row_changed(self, row: int) -> None:
        for index in range(self._list.count()):
            session_row = self._row_at(index)
            if session_row is not None:
                session_row.set_active(index == row)
        session_row = self._row_at(row)
        if session_row is None:
            self._show_empty("No session selected", "")
            return
        self._show_session(session_row.summary)
        self._update_selection_controls()

    def _show_session(self, summary: SessionSummary) -> None:
        self._meta.show()
        self._detail_title.setText(summary.title)
        if summary.is_archived:
            state = "Archived session"
        elif summary.is_last:
            state = "Last active session"
        else:
            state = "Saved session"
        self._detail_state.setText(state)
        relative = format_relative_time(summary.updated_at)
        timestamp = _format_timestamp(summary.updated_at)
        self._detail_values["detail-updated"].setText(f"{relative} · {timestamp}")
        saved_config = self._session_config_loader(summary.id)
        effective = saved_config or self._default_config
        self._detail_values["detail-llm"].setText(effective.label)
        self._detail_values["detail-provider"].setText(effective.provider_type)
        config_source = str(effective.path)
        if saved_config is None:
            config_source += " · project default"
        if not config_file_available(effective):
            config_source += " · missing"
        self._detail_values["detail-config"].setText(config_source)
        self._detail_values["detail-size"].setText(format_file_size(summary.size_bytes))
        file_label = "file" if summary.file_count == 1 else "files"
        self._detail_values["detail-storage"].setText(
            f"{summary.storage_format} · {summary.file_count} {file_label}"
        )
        self._detail_values["detail-todos"].setText(str(summary.todo_count))
        self._detail_values["detail-directories"].setText(str(summary.additional_dir_count))
        self._detail_values["detail-id"].setText(summary.id)
        self._detail_values["detail-path"].setText(str(self._work_dir))
        self._open.setEnabled(True)
        self._configure.setEnabled(True)

    def _show_empty(self, title: str, state: str) -> None:
        self._detail_title.setText(title)
        self._detail_state.setText(state)
        self._meta.hide()
        self._open.setEnabled(False)
        self._configure.setEnabled(False)

    def _update_selection_controls(self, filtered: list[SessionSummary] | None = None) -> None:
        count = len(self._selected_ids)
        selecting = count > 0
        self._selection_count.setText(f"{count} selected")
        self._selection_count.setVisible(selecting)
        self._history_title.setVisible(not selecting)
        self._session_count.setVisible(not selecting)
        self._delete.setEnabled(selecting)
        self._delete.setVisible(selecting)
        if filtered is None:
            filtered = [row.summary for row in self.session_rows()]
        has_rows = bool(filtered)
        self._select_shown.setEnabled(has_rows)
        self._select_shown.setVisible(has_rows)
        all_selected = has_rows and all(item.id in self._selected_ids for item in filtered)
        self._select_shown.setText("Clear" if all_selected else "Select all")
        for row in self.session_rows():
            row.set_selection_mode(selecting)

    def _toggle_shown(self) -> None:
        rows = self.session_rows()
        if not rows:
            return
        ids = {row.summary.id for row in rows}
        if ids.issubset(self._selected_ids):
            self._selected_ids.difference_update(ids)
        else:
            self._selected_ids.update(ids)
        for row in rows:
            row.set_selected(row.summary.id in self._selected_ids)
        self._update_selection_controls()

    def _on_mark_clicked(self, session_id: str) -> None:
        for row in self.session_rows():
            if row.summary.id != session_id:
                continue
            if row.checked:
                self._selected_ids.add(session_id)
            else:
                self._selected_ids.discard(session_id)
            row.selected = row.checked
            self._update_selection_controls()
            return

    def _toggle_row(self, row: SessionRow) -> None:
        session_id = row.summary.id
        if session_id in self._selected_ids:
            self._selected_ids.remove(session_id)
        else:
            self._selected_ids.add(session_id)
        row.set_selected(session_id in self._selected_ids)
        self._update_selection_controls()

    def _open_current(self) -> None:
        summary = self.summary
        if summary is not None:
            self._open_summary(summary)

    def _open_id(self, session_id: str) -> None:
        for row in self.session_rows():
            if row.summary.id == session_id:
                self._open_summary(row.summary)
                return

    def _open_summary(self, summary: SessionSummary) -> None:
        saved = self._session_config_loader(summary.id)
        if not config_file_available(saved or self._default_config):
            self.llm_required.emit(summary.id)
            return
        self.resume_session.emit(summary.id)

    def _configure_current(self) -> None:
        summary = self.summary
        if summary is not None:
            self.configure_session.emit(summary.id)


def _format_timestamp(updated_at: float) -> str:
    if updated_at <= 0:
        return "Unknown time"
    try:
        return datetime.fromtimestamp(updated_at).astimezone().strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return "Unknown time"

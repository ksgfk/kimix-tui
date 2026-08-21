"""Home view: session browser, search, batch delete, and details pane."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent, QKeySequence, QMouseEvent, QShortcut
from PySide6.QtWidgets import (
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
from kimix_tui.session_index import (
    SessionSummary,
    format_file_size,
    format_relative_time,
)

SessionConfigLoader = Callable[[str], LLMConfigReference | None]


class SessionRow(QWidget):
    """Compact selectable row for a saved session."""

    check_toggled = Signal(str)

    def __init__(self, summary: SessionSummary, *, selected: bool = False) -> None:
        super().__init__()
        self.summary = summary
        self.selected = selected
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(0)
        main = QHBoxLayout()
        self._check = QLabel("[x]" if selected else "[ ]")
        self._check.setObjectName("session-check")
        self._check.setProperty("session_id", summary.id)
        title = QLabel(summary.title)
        title.setObjectName("session-title")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._check.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        main.addWidget(self._check)
        main.addWidget(title, 1)
        meta = QLabel(
            f"{format_relative_time(summary.updated_at)} · {format_file_size(summary.size_bytes)}"
        )
        meta.setObjectName("session-meta")
        meta.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addLayout(main)
        layout.addWidget(meta)
        self.set_selected(selected)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._check.setText("[x]" if selected else "[ ]")

    def mousePressEvent(self, event: object) -> None:
        if not isinstance(event, QMouseEvent):
            return
        child = self.childAt(event.position().toPoint())
        check_rect = self._check.geometry()
        if child is self._check or (
            check_rect.contains(event.position().toPoint()) and event.position().x() <= check_rect.right()
        ):
            self.check_toggled.emit(self.summary.id)
            event.accept()
            return
        list_widget = self._list_widget()
        if list_widget is not None:
            for index in range(list_widget.count()):
                item = list_widget.item(index)
                if list_widget.itemWidget(item) is self:
                    list_widget.setCurrentRow(index)
                    break
        super().mousePressEvent(event)

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
        root.setContentsMargins(16, 12, 16, 12)
        toolbar = QFrame()
        toolbar.setObjectName("home-toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        context = QVBoxLayout()
        title = QLabel("SESSIONS")
        title.setObjectName("home-title")
        path = QLabel(str(self._work_dir))
        path.setObjectName("home-path")
        self._home_model = QLabel()
        self._home_model.setObjectName("home-model")
        context.addWidget(title)
        context.addWidget(path)
        context.addWidget(self._home_model)
        toolbar_layout.addLayout(context, 1)
        new_btn = QPushButton("+ New session")
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
        header = QHBoxLayout()
        history_title = QLabel("History")
        history_title.setObjectName("history-title")
        self._session_count = QLabel("")
        self._session_count.setObjectName("session-count")
        header.addWidget(history_title)
        header.addWidget(self._session_count)
        browser_layout.addLayout(header)
        self._search = QLineEdit()
        self._search.setObjectName("session-search")
        self._search.setPlaceholderText("Search by title")
        browser_layout.addWidget(self._search)
        batch = QHBoxLayout()
        self._selection_count = QLabel("0 selected")
        self._selection_count.setObjectName("selection-count")
        self._select_shown = QPushButton("Select shown")
        self._select_shown.setObjectName("select-shown")
        self._select_shown.setEnabled(False)
        self._delete = QPushButton("Delete")
        self._delete.setObjectName("delete-sessions")
        self._delete.setEnabled(False)
        batch.addWidget(self._selection_count, 1)
        batch.addWidget(self._select_shown)
        batch.addWidget(self._delete)
        browser_layout.addLayout(batch)
        self._status = QLabel("Loading sessions…")
        self._status.setObjectName("home-status")
        browser_layout.addWidget(self._status)
        self._list = QListWidget()
        self._list.setObjectName("session-list")
        self._list.installEventFilter(self)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemActivated.connect(self._open_current)
        browser_layout.addWidget(self._list, 1)
        self._splitter.addWidget(browser)

        self._details = QWidget()
        self._details.setObjectName("session-detail")
        details_layout = QVBoxLayout(self._details)
        overline = QLabel("SESSION DETAILS")
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
        self._open = QPushButton("Open session")
        self._open.setObjectName("open-session")
        self._open.setEnabled(False)
        self._configure = QPushButton("Configure")
        self._configure.setObjectName("configure-session")
        self._configure.setEnabled(False)
        self._toggle = QPushButton("Select")
        self._toggle.setObjectName("toggle-session")
        self._toggle.setEnabled(False)
        actions.addWidget(self._open)
        actions.addWidget(self._configure)
        actions.addWidget(self._toggle)
        details_layout.addLayout(actions)
        self._meta = QWidget()
        self._meta.setObjectName("detail-metadata")
        form = QVBoxLayout(self._meta)
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
        self._toggle.clicked.connect(self.toggle_selected)
        self._select_shown.clicked.connect(self._toggle_shown)
        self._delete.clicked.connect(self.request_delete)
        self._search.textChanged.connect(self._render_sessions)
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self, self.focus_search)
        search_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

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
            row.check_toggled.connect(self._toggle_id)
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, 48))
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
        self._toggle.setEnabled(True)

    def _show_empty(self, title: str, state: str) -> None:
        self._detail_title.setText(title)
        self._detail_state.setText(state)
        self._meta.hide()
        self._open.setEnabled(False)
        self._configure.setEnabled(False)
        self._toggle.setEnabled(False)

    def _update_selection_controls(self, filtered: list[SessionSummary] | None = None) -> None:
        count = len(self._selected_ids)
        self._selection_count.setText(f"{count} selected")
        self._delete.setEnabled(count > 0)
        if filtered is None:
            filtered = [row.summary for row in self.session_rows()]
        self._select_shown.setEnabled(bool(filtered))
        all_selected = bool(filtered) and all(item.id in self._selected_ids for item in filtered)
        self._select_shown.setText("Clear shown" if all_selected else "Select shown")
        selected = self.summary is not None and self.summary.id in self._selected_ids
        self._toggle.setText("Deselect" if selected else "Select")

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

    def _toggle_id(self, session_id: str) -> None:
        for row in self.session_rows():
            if row.summary.id == session_id:
                self._toggle_row(row)
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

    def _open_summary(self, summary: SessionSummary) -> None:
        if not self.configuration_available:
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

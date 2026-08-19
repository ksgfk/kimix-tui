"""Home screen for browsing and opening project sessions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from kimix_tui.llm_config import LLMConfigReference, config_file_available
from kimix_tui.screens.settings import OpenLLMSettings
from kimix_tui.session_index import (
    SessionDeleter,
    SessionLoader,
    SessionSummary,
    delete_sessions,
    format_file_size,
    format_relative_time,
    list_session_summaries,
)

SessionConfigLoader = Callable[[str], LLMConfigReference | None]


@dataclass(frozen=True, slots=True)
class SessionChoice:
    """Result returned when the user leaves the home screen."""

    action: Literal["new", "resume", "quit"]
    session_id: str | None = None


class SessionListItem(ListItem):
    """Compact selectable row for a saved session."""

    def __init__(self, summary: SessionSummary, *, selected: bool = False) -> None:
        super().__init__()
        self.summary = summary
        self.selected = selected

    def compose(self) -> ComposeResult:
        with Horizontal(classes="session-row-main"):
            yield Static(
                "[x]" if self.selected else "[ ]",
                classes="session-check",
                markup=False,
            )
            yield Label(self.summary.title, markup=False, classes="session-title")
        yield Static(
            f"{format_relative_time(self.summary.updated_at)} · "
            f"{format_file_size(self.summary.size_bytes)}",
            markup=False,
            classes="session-meta",
        )

    def on_mount(self) -> None:
        self.set_class(self.selected, "-selected")

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.set_class(selected, "-selected")
        self.query_one(".session-check", Static).update("[x]" if selected else "[ ]")


class SessionListView(ListView):
    """Use mouse selection for preview and Enter for opening a session."""

    def action_select_cursor(self) -> None:
        item = self.highlighted_child
        screen = self.screen
        if isinstance(item, SessionListItem) and isinstance(screen, HomeScreen):
            screen.open_session(item.summary)


class DeleteSessionsScreen(ModalScreen[bool]):
    """Confirm permanent deletion of one or more sessions."""

    CSS = """
    DeleteSessionsScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }

    #delete-dialog {
        width: 58;
        max-width: 92%;
        height: auto;
        padding: 1 2;
        border: round $error;
        background: $surface;
    }

    #delete-title {
        height: 1;
        text-style: bold;
    }

    #delete-copy {
        height: auto;
        margin: 1 0;
        color: $text-muted;
    }

    #delete-actions {
        height: 3;
        align: right middle;
    }

    #cancel-delete, #confirm-delete {
        width: 14;
        min-width: 14;
        height: 3;
        margin-left: 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count

    def compose(self) -> ComposeResult:
        noun = "session" if self._count == 1 else "sessions"
        with Vertical(id="delete-dialog"):
            yield Label(f"Delete {self._count} {noun}?", id="delete-title")
            yield Static(
                "Conversation history and session files will be permanently removed.",
                id="delete-copy",
            )
            with Horizontal(id="delete-actions"):
                yield Button("Cancel", id="cancel-delete")
                yield Button("Delete", id="confirm-delete", variant="error")

    @on(Button.Pressed, "#confirm-delete")
    def confirm_delete(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-delete")
    def cancel_delete(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(False)


class SessionDetails(VerticalScroll):
    """Details for the session currently highlighted on the home screen."""

    def __init__(
        self,
        work_dir: Path,
        *,
        default_config: LLMConfigReference,
        session_config_loader: SessionConfigLoader,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._work_dir = work_dir
        self._default_config = default_config
        self._session_config_loader = session_config_loader
        self._summary: SessionSummary | None = None
        self._configuration_available = False

    def compose(self) -> ComposeResult:
        yield Static("SESSION DETAILS", id="detail-overline")
        yield Static("Loading sessions…", id="detail-title", markup=False)
        yield Static("Reading this project", id="detail-state", markup=False)
        with Horizontal(id="session-actions"):
            yield Button("Open session", id="open-session", variant="primary", disabled=True)
            yield Button("Configure", id="configure-session", disabled=True)
            yield Button("Select", id="toggle-session", disabled=True)
        with Vertical(id="detail-metadata"):
            with Horizontal(classes="detail-row"):
                yield Static("Updated", classes="detail-key")
                yield Static("", id="detail-updated", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("LLM", classes="detail-key")
                yield Static("", id="detail-llm", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Provider", classes="detail-key")
                yield Static("", id="detail-provider", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Config", classes="detail-key")
                yield Static("", id="detail-config", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Size", classes="detail-key")
                yield Static("", id="detail-size", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Storage", classes="detail-key")
                yield Static("", id="detail-storage", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Active todos", classes="detail-key")
                yield Static("", id="detail-todos", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Extra folders", classes="detail-key")
                yield Static("", id="detail-directories", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Session ID", classes="detail-key")
                yield Static("", id="detail-id", classes="detail-value", markup=False)
            with Horizontal(classes="detail-row"):
                yield Static("Folder", classes="detail-key")
                yield Static("", id="detail-path", classes="detail-value", markup=False)

    @property
    def summary(self) -> SessionSummary | None:
        return self._summary

    @property
    def configuration_available(self) -> bool:
        return self._configuration_available

    def show_session(self, summary: SessionSummary) -> None:
        self._summary = summary
        self.query_one("#detail-title", Static).update(summary.title)
        if summary.is_archived:
            state = "Archived session"
        elif summary.is_last:
            state = "Last active session"
        else:
            state = "Saved session"
        self.query_one("#detail-state", Static).update(state)
        relative = format_relative_time(summary.updated_at)
        timestamp = self._format_timestamp(summary.updated_at)
        self.query_one("#detail-updated", Static).update(f"{relative} · {timestamp}")
        saved_config = self._session_config_loader(summary.id)
        effective_config = saved_config or self._default_config
        self._configuration_available = config_file_available(effective_config)
        self.query_one("#detail-llm", Static).update(effective_config.label)
        self.query_one("#detail-provider", Static).update(effective_config.provider_type)
        config_source = str(effective_config.path)
        if saved_config is None:
            config_source += " · project default"
        if not self._configuration_available:
            config_source += " · missing"
        self.query_one("#detail-config", Static).update(config_source)
        self.query_one("#detail-size", Static).update(format_file_size(summary.size_bytes))
        file_label = "file" if summary.file_count == 1 else "files"
        self.query_one("#detail-storage", Static).update(
            f"{summary.storage_format} · {summary.file_count} {file_label}"
        )
        self.query_one("#detail-todos", Static).update(str(summary.todo_count))
        self.query_one("#detail-directories", Static).update(str(summary.additional_dir_count))
        self.query_one("#detail-id", Static).update(summary.id)
        self.query_one("#detail-path", Static).update(str(self._work_dir))
        self.query_one("#detail-metadata", Vertical).display = True
        self.query_one("#open-session", Button).disabled = not self._configuration_available
        self.query_one("#configure-session", Button).disabled = False
        self.query_one("#toggle-session", Button).disabled = False

    def show_empty(self, title: str, state: str) -> None:
        self._summary = None
        self.query_one("#detail-title", Static).update(title)
        self.query_one("#detail-state", Static).update(state)
        self.query_one("#detail-metadata", Vertical).display = False
        self.query_one("#open-session", Button).disabled = True
        self.query_one("#configure-session", Button).disabled = True
        self.query_one("#toggle-session", Button).disabled = True
        self._configuration_available = False

    def refresh_configuration(self, default_config: LLMConfigReference) -> None:
        self._default_config = default_config
        if self._summary is not None:
            self.show_session(self._summary)

    @staticmethod
    def _format_timestamp(updated_at: float) -> str:
        if updated_at <= 0:
            return "Unknown time"
        try:
            return datetime.fromtimestamp(updated_at).astimezone().strftime("%Y-%m-%d %H:%M")
        except OSError, OverflowError, ValueError:
            return "Unknown time"


class HomeScreen(Screen[SessionChoice]):
    """Browse project sessions or start a new one."""

    CSS = """
    HomeScreen {
        background: $surface;
    }

    #home-toolbar {
        height: 4;
        padding: 0 2;
        align: left middle;
        border-bottom: tall $panel;
        background: $surface;
    }

    #home-context {
        width: 1fr;
        height: 3;
    }

    #home-title {
        height: 1;
        text-style: bold;
        color: $accent;
    }

    #home-path {
        height: 1;
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #home-model {
        height: 1;
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #home-actions {
        width: auto;
        height: 3;
    }

    #start-new-session {
        width: 18;
        min-width: 18;
        height: 3;
        margin: 0;
    }

    #open-settings {
        width: 14;
        min-width: 14;
        height: 3;
        margin: 0 0 0 1;
    }

    #home-workspace {
        height: 1fr;
        padding: 1 2;
        layout: horizontal;
    }

    #session-browser {
        width: 42;
        min-width: 32;
        height: 100%;
        padding-right: 1;
    }

    #session-list-header {
        width: 100%;
        height: 1;
        align: left middle;
    }

    #history-title {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    #session-count {
        width: auto;
        height: 1;
        color: $text-muted;
        text-align: right;
    }

    #home-status {
        height: auto;
        min-height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    #session-search {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #batch-actions {
        width: 100%;
        height: 3;
        align: left middle;
    }

    #selection-count {
        width: 1fr;
        height: 1;
        color: $text-muted;
    }

    #select-shown {
        width: 13;
        min-width: 13;
        height: 3;
    }

    #delete-sessions {
        width: 10;
        min-width: 10;
        height: 3;
        margin-left: 1;
    }

    #session-list {
        height: 1fr;
        border: tall $panel;
        background: $surface;
        scrollbar-size: 1 1;
    }

    #session-list > SessionListItem {
        width: 100%;
        height: 2;
        padding: 0 1;
        color: $text-muted;
    }

    #session-list > SessionListItem:hover {
        background: $boost;
        color: $text;
    }

    #session-list > SessionListItem.-highlight {
        padding-left: 0;
        border-left: thick $accent;
        background: $boost;
        color: $text;
    }

    #session-list > SessionListItem.-selected {
        background: $surface-lighten-1;
        color: $text;
    }

    .session-row-main {
        width: 100%;
        height: 1;
    }

    .session-check {
        width: 4;
        height: 1;
        color: $accent;
    }

    .session-title {
        width: 1fr;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    .session-meta {
        width: 100%;
        height: 1;
        padding-left: 4;
        color: $text-muted;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #session-detail {
        width: 1fr;
        height: 100%;
        padding: 2 3;
        border-left: tall $surface-lighten-1;
        background: $panel;
        scrollbar-size: 1 1;
    }

    #detail-overline {
        height: 1;
        color: $accent;
        text-style: bold;
    }

    #detail-title {
        height: auto;
        max-height: 3;
        margin-top: 1;
        color: $text;
        text-style: bold;
        text-wrap: wrap;
    }

    #detail-state {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }

    #detail-metadata {
        height: auto;
        margin-top: 1;
    }

    .detail-row {
        width: 100%;
        height: 1;
    }

    .detail-key {
        width: 16;
        height: 1;
        color: $text-muted;
    }

    .detail-value {
        width: 1fr;
        height: 1;
        color: $text;
    }

    #detail-updated, #detail-config, #detail-storage, #detail-id, #detail-path {
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #session-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #open-session, #configure-session, #toggle-session {
        width: 14;
        min-width: 12;
        height: 3;
    }

    #configure-session, #toggle-session {
        margin-left: 1;
    }

    HomeScreen.-narrow #home-workspace {
        layout: vertical;
    }

    HomeScreen.-narrow #session-browser {
        width: 100%;
        min-width: 0;
        height: 3fr;
        min-height: 8;
        padding-right: 0;
        padding-bottom: 1;
    }

    HomeScreen.-narrow #session-detail {
        width: 100%;
        height: 2fr;
        min-height: 8;
        padding: 1 2;
        border-left: none;
        border-top: tall $surface-lighten-1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("n", "new_session", "New", show=False),
        Binding("ctrl+f", "focus_search", "Search", show=False, priority=True),
        Binding("space", "toggle_selected", "Select", show=False),
        Binding("delete", "delete_selected", "Delete", show=False),
        Binding("f4", "settings", "Settings", show=False),
        Binding("q,escape", "quit_home", "Quit", show=False),
    ]

    def __init__(
        self,
        work_dir: Path,
        *,
        default_config: LLMConfigReference,
        session_config_loader: SessionConfigLoader,
        loader: SessionLoader | None = None,
        deleter: SessionDeleter | None = None,
    ) -> None:
        super().__init__()
        self._work_dir = work_dir
        self._default_config = default_config
        self._session_config_loader = session_config_loader
        self._loader = loader
        self._deleter = deleter
        self._summaries: list[SessionSummary] = []
        self._selected_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        default_available = config_file_available(self._default_config)
        default_status = "" if default_available else " · missing"
        with Horizontal(id="home-toolbar"):
            with Vertical(id="home-context"):
                yield Label("SESSIONS", id="home-title")
                yield Static(str(self._work_dir), id="home-path", markup=False)
                yield Static(
                    f"New sessions · {self._default_config.label}{default_status}",
                    id="home-model",
                    markup=False,
                )
            with Horizontal(id="home-actions"):
                yield Button(
                    "+ New session",
                    id="start-new-session",
                    variant="primary",
                    disabled=not default_available,
                )
                yield Button("Settings", id="open-settings")
        with Horizontal(id="home-workspace"):
            with Vertical(id="session-browser"):
                with Horizontal(id="session-list-header"):
                    yield Label("History", id="history-title")
                    yield Static("", id="session-count")
                yield Input(placeholder="Search by title", id="session-search")
                with Horizontal(id="batch-actions"):
                    yield Static("0 selected", id="selection-count")
                    yield Button("Select shown", id="select-shown", disabled=True)
                    yield Button("Delete", id="delete-sessions", disabled=True)
                yield Static("Loading sessions…", id="home-status")
                yield SessionListView(id="session-list")
            yield SessionDetails(
                self._work_dir,
                default_config=self._default_config,
                session_config_loader=self._session_config_loader,
                id="session-detail",
            )

    def on_mount(self) -> None:
        self.app.title = "Kimix"
        self.app.sub_title = str(self._work_dir)
        self.load_sessions()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 78, "-narrow")

    @work(exclusive=True, group="home")
    async def load_sessions(self) -> None:
        status = self.query_one("#home-status", Static)
        details = self.query_one(SessionDetails)
        status.display = True
        status.update("Loading sessions…")
        self.query_one("#session-count", Static).update("")
        try:
            loader = self._loader or list_session_summaries
            summaries = sorted(
                await loader(self._work_dir),
                key=lambda summary: summary.updated_at,
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001 - home must stay usable
            list_view = self.query_one("#session-list", ListView)
            await list_view.clear()
            list_view.display = False
            status.update(f"Could not load sessions: {exc}")
            self.query_one("#session-count", Static).update("Unavailable")
            details.show_empty("Sessions unavailable", "Start a new session to continue")
            self.query_one("#start-new-session", Button).focus()
            return

        self._summaries = summaries
        self._selected_ids.intersection_update(summary.id for summary in summaries)
        await self._render_sessions()

    async def _render_sessions(self, *, preferred_id: str | None = None) -> None:
        query = self.query_one("#session-search", Input).value.strip().casefold()
        filtered = [
            summary
            for summary in self._summaries
            if not query or query in summary.title.casefold()
        ]
        list_view = self.query_one("#session-list", ListView)
        status = self.query_one("#home-status", Static)
        details = self.query_one(SessionDetails)
        await list_view.clear()
        total = len(self._summaries)
        self.query_one("#session-count", Static).update(
            f"{len(filtered)} of {total}" if query else f"{total} total"
        )
        self._update_selection_controls(filtered)
        if not filtered:
            list_view.display = False
            status.display = True
            if total and query:
                status.update("No sessions match this title")
                details.show_empty("No matching sessions", "Try another title")
            else:
                status.update("No saved sessions in this folder")
                details.show_empty("No sessions yet", "Start a new session in this folder")
            return

        status.display = False
        list_view.display = True
        await list_view.extend(
            [
                SessionListItem(summary, selected=summary.id in self._selected_ids)
                for summary in filtered
            ]
        )
        ids = [summary.id for summary in filtered]
        list_view.index = ids.index(preferred_id) if preferred_id in ids else 0
        if not self.query_one("#session-search", Input).has_focus:
            list_view.focus()

    def _update_selection_controls(
        self,
        filtered: list[SessionSummary] | None = None,
    ) -> None:
        count = len(self._selected_ids)
        self.query_one("#selection-count", Static).update(f"{count} selected")
        self.query_one("#delete-sessions", Button).disabled = count == 0
        if filtered is None:
            filtered = [item.summary for item in self.query(SessionListItem)]
        select_button = self.query_one("#select-shown", Button)
        select_button.disabled = not filtered
        all_selected = bool(filtered) and all(item.id in self._selected_ids for item in filtered)
        select_button.label = "Clear shown" if all_selected else "Select shown"
        details = self.query_one(SessionDetails)
        selected = details.summary is not None and details.summary.id in self._selected_ids
        self.query_one("#toggle-session", Button).label = "Deselect" if selected else "Select"

    @on(Input.Changed, "#session-search")
    async def filter_sessions(self) -> None:
        await self._render_sessions()

    @on(ListView.Highlighted, "#session-list")
    def highlight_session(self, event: ListView.Highlighted) -> None:
        details = self.query_one(SessionDetails)
        if isinstance(event.item, SessionListItem):
            details.show_session(event.item.summary)
            self._update_selection_controls()
        else:
            details.show_empty("No session selected", "")

    @on(Button.Pressed, "#open-session")
    def press_open_session(self) -> None:
        summary = self.query_one(SessionDetails).summary
        if summary is not None:
            self.open_session(summary)

    @on(Button.Pressed, "#configure-session")
    def press_configure_session(self) -> None:
        summary = self.query_one(SessionDetails).summary
        if summary is not None:
            self.post_message(OpenLLMSettings(summary.id))

    @on(Button.Pressed, "#toggle-session")
    def press_toggle_session(self) -> None:
        self.action_toggle_selected()

    @on(Button.Pressed, "#select-shown")
    def press_select_shown(self) -> None:
        items = list(self.query(SessionListItem))
        if not items:
            return
        ids = {item.summary.id for item in items}
        if ids.issubset(self._selected_ids):
            self._selected_ids.difference_update(ids)
        else:
            self._selected_ids.update(ids)
        for item in items:
            item.set_selected(item.summary.id in self._selected_ids)
        self._update_selection_controls()

    @on(Button.Pressed, "#delete-sessions")
    def press_delete_sessions(self) -> None:
        self.action_delete_selected()

    @on(Button.Pressed, "#open-settings")
    def press_settings(self) -> None:
        self.action_settings()

    @on(Button.Pressed, "#start-new-session")
    def press_new_session(self) -> None:
        self.action_new_session()

    def open_session(self, summary: SessionSummary) -> None:
        if not self.query_one(SessionDetails).configuration_available:
            self.post_message(OpenLLMSettings(summary.id))
            return
        self.dismiss(SessionChoice(action="resume", session_id=summary.id))

    def refresh_configuration(self, default_config: LLMConfigReference) -> None:
        self._default_config = default_config
        available = config_file_available(default_config)
        status = "" if available else " · missing"
        self.query_one("#home-model", Static).update(
            f"New sessions · {default_config.label}{status}"
        )
        self.query_one("#start-new-session", Button).disabled = not available
        self.query_one(SessionDetails).refresh_configuration(default_config)

    def action_focus_search(self) -> None:
        self.query_one("#session-search", Input).focus()

    def action_toggle_selected(self) -> None:
        item = self.query_one("#session-list", ListView).highlighted_child
        if not isinstance(item, SessionListItem):
            return
        session_id = item.summary.id
        if session_id in self._selected_ids:
            self._selected_ids.remove(session_id)
        else:
            self._selected_ids.add(session_id)
        item.set_selected(session_id in self._selected_ids)
        self._update_selection_controls()

    def action_delete_selected(self) -> None:
        if not self._selected_ids:
            return
        self.app.push_screen(DeleteSessionsScreen(len(self._selected_ids)), self._delete_confirmed)

    def _delete_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            self.delete_selected_sessions()

    @work(exclusive=True, group="delete-sessions")
    async def delete_selected_sessions(self) -> None:
        ids = [summary.id for summary in self._summaries if summary.id in self._selected_ids]
        if not ids:
            return
        self.query_one("#delete-sessions", Button).disabled = True
        try:
            await (self._deleter or delete_sessions)(self._work_dir, ids)
        except Exception as exc:  # noqa: BLE001 - keep the browser usable
            self.notify(f"Failed to delete sessions: {exc}", severity="error")
            self._update_selection_controls()
            return
        self._summaries = [summary for summary in self._summaries if summary.id not in ids]
        self._selected_ids.difference_update(ids)
        await self._render_sessions()
        noun = "session" if len(ids) == 1 else "sessions"
        self.notify(f"Deleted {len(ids)} {noun}")

    def action_new_session(self) -> None:
        if not config_file_available(self._default_config):
            self.post_message(OpenLLMSettings())
            return
        self.dismiss(SessionChoice(action="new"))

    def action_settings(self) -> None:
        self.post_message(OpenLLMSettings())

    def action_quit_home(self) -> None:
        self.dismiss(SessionChoice(action="quit"))

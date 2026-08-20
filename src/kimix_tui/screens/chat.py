"""Full-screen chat experience and SDK session lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import ClassVar

from kimi_agent_sdk import ApprovalRequest, RunCancelled, ToolError, is_request
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Static

from kimix_tui.backend import (
    SdkSession,
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
)
from kimix_tui.history import (
    HistoryLoader,
    SessionHistory,
    Timeline,
    create_timeline,
)
from kimix_tui.rendering import RenderEvent, RenderState, format_status, render_wire_message
from kimix_tui.screens.requests import ApprovalScreen, QuestionScreen
from kimix_tui.screens.settings import OpenLLMSettings
from kimix_tui.widgets import PromptInput, Transcript

SessionFactory = Callable[[SessionOptions], Awaitable[SdkSession]]
SessionOpenedCallback = Callable[[str], None]


class ChatScreen(Screen[None]):
    """Run one SDK session inside a full-screen chat interface."""

    CSS = """
    ChatScreen {
        layout: vertical;
        background: $background;
    }

    #chat-toolbar, #history-toolbar {
        height: 1;
        padding: 0 1;
        align: left middle;
        background: $surface;
    }

    #chat-title {
        width: auto;
        height: 1;
        padding: 0 1 0 0;
        text-style: bold;
        color: $accent;
        content-align: left middle;
    }

    #status {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        content-align: left middle;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #chat-actions, #history-actions {
        width: auto;
        height: 1;
        align: right middle;
    }

    #chat-toolbar Button, #history-toolbar Button {
        height: 1;
        min-height: 1;
        width: auto;
        min-width: 0;
        margin: 0 0 0 1;
        padding: 0 1;
        border: none;
        background: $surface;
        color: $text-muted;
    }

    #chat-toolbar Button:hover, #history-toolbar Button:hover {
        background: $boost;
        color: $text;
    }

    #chat-toolbar Button:focus, #history-toolbar Button:focus {
        background: $boost;
        color: $accent;
        text-style: bold;
    }

    #leave-session {
        color: $text;
    }

    #history-info {
        width: 1fr;
        height: 1;
        color: $text-muted;
        content-align: left middle;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #history-turn {
        width: 8;
        min-width: 6;
        height: 1;
        margin: 0 0 0 1;
        padding: 0 1;
        content-align: left middle;
        border: none;
        background: $boost;
    }

    #history-turn:focus {
        background: $panel;
        color: $accent;
    }

    ChatScreen.-narrow #history-info {
        display: none;
    }

    ChatScreen.-narrow #history-actions {
        width: 1fr;
    }

    ChatScreen.-narrow #history-actions Button {
        width: 1fr;
        min-width: 0;
        margin-left: 1;
        padding: 0;
    }

    ChatScreen.-narrow #history-turn {
        width: 6;
        min-width: 4;
        padding: 0;
    }

    #transcript {
        height: 1fr;
        padding: 1 3 0 3;
        background: $background;
        scrollbar-size: 1 1;
    }

    #prompt {
        dock: bottom;
        width: 100%;
        height: 3;
        min-height: 3;
        max-height: 8;
        margin: 1 2;
        padding: 0 1;
        border: round $accent;
        background: $panel;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }

    #chat-footer {
        dock: bottom;
        height: 1;
        width: 100%;
        align: left middle;
        background: $footer-background;
    }

    #chat-footer Footer {
        dock: none;
        width: auto;
        height: 1;
        min-width: 0;
    }

    #context {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $footer-description-foreground;
        background: $footer-background;
        content-align: right middle;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+g", "cancel_prompt", "Cancel", priority=True),
        Binding("ctrl+up", "load_older", "Earlier", show=False),
        Binding("ctrl+end", "jump_latest", "Latest", show=False),
        Binding("f2", "focus_prompt", "Prompt", show=False),
        Binding("f3", "focus_history_turn", "Turn", show=False),
        Binding("f4", "settings", "Settings", show=False),
        Binding("escape", "leave_session", "Home", show=False),
    ]

    def __init__(
        self,
        options: SessionOptions,
        *,
        session_factory: SessionFactory = create_sdk_session,
        history_loader: HistoryLoader | None = None,
        on_session_opened: SessionOpenedCallback | None = None,
    ) -> None:
        super().__init__()
        self._options = options
        self._session_factory = session_factory
        self._history_loader = history_loader
        self._on_session_opened = on_session_opened
        self._session: SdkSession | None = None
        self._busy = False
        self._chat_epoch = 0
        self._leaving = False
        self._pending_config_label: str | None = None
        self._render_state = RenderState()
        self._last_wire_status: str | None = None
        self._timeline: Timeline | None = None
        self._history_total_turns = 0
        self._history_loading = False
        self._history_legacy_omitted = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-toolbar"):
            yield Label("CHAT", id="chat-title")
            yield Static("connecting…", id="status", markup=False)
            with Horizontal(id="chat-actions"):
                yield Button("Settings", id="open-settings", compact=True)
                yield Button("Home", id="leave-session", compact=True)
        with Horizontal(id="history-toolbar"):
            yield Static("History · connecting…", id="history-info", markup=False)
            with Horizontal(id="history-actions"):
                yield Button(
                    "←",
                    id="load-older",
                    compact=True,
                    tooltip="Previous turn",
                )
                yield Input(
                    placeholder="Turn #",
                    id="history-turn",
                    type="integer",
                    restrict=r"[0-9]*",
                    disabled=True,
                    tooltip="Seek to turn",
                )
                yield Button(
                    "→",
                    id="load-newer",
                    compact=True,
                    disabled=True,
                    tooltip="Next turn",
                )
                yield Button(
                    "↓",
                    id="jump-latest",
                    compact=True,
                    disabled=True,
                    tooltip="Jump to latest",
                )
        yield Transcript(id="transcript")
        yield PromptInput(
            placeholder="Ask AI, or type /help",
            id="prompt",
            disabled=True,
            tooltip="Enter to send · Ctrl+Enter for newline",
        )
        with Horizontal(id="chat-footer"):
            yield Footer(show_command_palette=False, compact=True)
            yield Static("", id="context", markup=False)

    def on_mount(self) -> None:
        self.app.title = "Kimix"
        self.app.sub_title = "Chat"
        self.open_session()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 64, "-narrow")
        if self.is_mounted and self._history_total_turns:
            self._update_history_toolbar()

    @property
    def session(self) -> SdkSession | None:
        return self._session

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def transcript(self) -> Transcript:
        return self.query_one("#transcript", Transcript)

    def _mounted_transcript(self) -> Transcript | None:
        """Return the transcript if this screen is still attached to the DOM."""

        if not self.is_attached:
            return None
        with suppress(NoMatches):
            return self.query_one("#transcript", Transcript)
        return None

    @work(exclusive=True, group="session")
    async def open_session(self) -> None:
        epoch = self._chat_epoch
        try:
            session = await self._session_factory(self._options)
        except Exception as exc:  # noqa: BLE001 - keep UI alive on SDK startup errors
            if epoch != self._chat_epoch:
                return
            await self._append_transcript("error", f"Failed to open session: {exc}")
            self._set_status("session unavailable")
            return
        if epoch != self._chat_epoch:
            await self._close_sdk_session(session)
            return
        self._session = session
        if self._on_session_opened is not None:
            try:
                self._on_session_opened(session.id)
            except Exception as exc:  # noqa: BLE001 - chat remains usable if metadata fails
                await self._append_transcript(
                    "error", f"Failed to save session configuration metadata: {exc}"
                )
        self._set_status(f"session {session.id}")
        self._set_context(format_status(session.status))
        if epoch != self._chat_epoch:
            return
        await self._append_transcript("system", f"Session: {session.id}")
        if epoch != self._chat_epoch:
            return
        try:
            await self._replay_history()
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._set_input_enabled(True)

    async def _replay_history(self) -> None:
        session = self._session
        epoch = self._chat_epoch
        if session is None:
            return
        timeline: Timeline | None = None
        if self._history_loader is None:
            self._history_loading = True
            self._update_history_toolbar()
        try:
            if self._history_loader is None:
                timeline = await create_timeline(
                    self._options.work_dir,
                    session.id,
                )
                if timeline is None:
                    self._history_loading = False
                    self._update_history_toolbar()
                    return
                history = None
            else:
                history = await self._history_loader(self._options.work_dir, session.id)
        except Exception as exc:  # noqa: BLE001 - chat still works without replay
            self._history_loading = False
            if epoch != self._chat_epoch or self._session is not session:
                return
            self._update_history_toolbar()
            await self.transcript.append_block("error", f"Failed to load history: {exc}")
            return
        if epoch != self._chat_epoch or self._session is not session:
            return
        if self._history_loader is None:
            self._timeline = timeline
            await self._mount_timeline()
            return

        self._reset_history_state()
        if history.omitted_turns:
            shown_turns = sum(1 for block in history.blocks if block.kind == "user")
            await self.transcript.append_block(
                "system",
                f"Showing last {shown_turns} turns ({history.omitted_turns} earlier omitted)",
            )
        if history.blocks:
            history_start = len(self.transcript.records)
            await self.transcript.append_blocks(
                [(block.kind, block.text) for block in history.blocks]
            )
            self.transcript.mark_history_window(history_start, len(self.transcript.records))
        self._history_legacy_omitted = history.omitted_turns
        self._update_history_toolbar()

    async def _mount_timeline(self) -> None:
        """Mount materialized timeline rows after the session prefix."""

        timeline = self._timeline
        if timeline is None:
            self._history_loading = False
            self._update_history_toolbar()
            return
        items = timeline.display_items()
        if items:
            await self.transcript.replace_history_blocks(
                [(kind, text, turn) for kind, text, turn in items]
            )
        else:
            self.transcript.mark_history_window()
        self._history_total_turns = timeline.total_turns
        self._history_legacy_omitted = 0
        self._history_loading = False
        self.transcript.jump_to_latest()
        self._update_history_toolbar()

    def _reset_history_state(self) -> None:
        self._timeline = None
        self._history_total_turns = 0
        self._history_loading = False
        self._history_legacy_omitted = 0

    def _display_turn(self) -> int:
        """1-based turn shown in the toolbar."""

        total = self._history_total_turns
        transcript = self._mounted_transcript()
        if transcript is None:
            return total
        if transcript.pinned_to_latest:
            return total
        viewport = transcript.viewport_turn()
        if viewport is not None:
            return viewport + 1
        return total

    def _update_history_toolbar(self) -> None:
        if not self.is_attached:
            return
        try:
            toolbar = self.query_one("#history-toolbar", Horizontal)
            info = self.query_one("#history-info", Static)
            older = self.query_one("#load-older", Button)
            turn_input = self.query_one("#history-turn", Input)
            newer = self.query_one("#load-newer", Button)
            latest = self.query_one("#jump-latest", Button)
        except NoMatches:
            return
        timeline = self._timeline
        toolbar.display = True

        if timeline is None:
            if self._history_loading:
                info.update("History · loading…")
            elif self._history_legacy_omitted:
                info.update(
                    f"History · {self._history_legacy_omitted} earlier turns unavailable"
                )
            else:
                info.update("History · no turns yet")
            older.disabled = True
            turn_input.disabled = True
            newer.disabled = True
            latest.disabled = True
            return

        total = self._history_total_turns or timeline.total_turns
        current = self._display_turn() if total else 0
        if total <= 0:
            info.update("History · no turns yet")
            older.disabled = True
            turn_input.disabled = True
            newer.disabled = True
            latest.disabled = True
            return

        info.update(f"History · Turn {current} of {total}")
        older.label = "←"
        newer.label = "→"
        latest.label = "↓"
        older.disabled = self._history_loading or current <= 1
        turn_input.placeholder = f"Turn 1-{total}"
        turn_input.disabled = self._history_loading
        newer.disabled = self._history_loading or current >= total
        transcript = self._mounted_transcript()
        latest.disabled = self._history_loading or (
            current >= total and (transcript is None or transcript.pinned_to_latest)
        )
        if self._history_loading:
            info.update("History · loading…")

    @on(Transcript.ReachedTop)
    def _transcript_reached_top(self, event: Transcript.ReachedTop) -> None:
        event.stop()
        if self._timeline is None or self._history_loading:
            return
        if self._timeline.first_materialized_turn() <= 0:
            self._update_history_toolbar()
            return
        self.prefetch_older_history()

    @on(Transcript.ReachedBottom)
    def _transcript_reached_bottom(self, event: Transcript.ReachedBottom) -> None:
        event.stop()
        if self._timeline is None or self._history_loading:
            return
        last = self._timeline.last_materialized_turn()
        if last + 1 >= self._history_total_turns:
            self._update_history_toolbar()
            return
        self.prefetch_newer_history()

    @on(Transcript.ViewportTurn)
    def _transcript_viewport_turn(self, event: Transcript.ViewportTurn) -> None:
        event.stop()
        self._update_history_toolbar()

    @on(Button.Pressed, "#load-older")
    def press_load_older(self) -> None:
        self.load_older_history()

    @on(Button.Pressed, "#load-newer")
    def press_load_newer(self) -> None:
        self.load_newer_history()

    @on(Button.Pressed, "#jump-latest")
    def press_jump_latest(self) -> None:
        self.jump_to_latest()

    @on(Button.Pressed, "#open-settings")
    def press_settings(self) -> None:
        self.action_settings()

    @on(Button.Pressed, "#leave-session")
    def press_leave_session(self) -> None:
        self.action_leave_session()

    @on(Input.Submitted, "#history-turn")
    def submit_history_turn(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if not value:
            return
        try:
            turn = int(value)
        except ValueError:
            self.notify("Enter a numeric turn", severity="warning")
            return
        if turn < 1 or turn > self._history_total_turns:
            self.notify(
                f"Turn must be between 1 and {self._history_total_turns}",
                severity="warning",
            )
            return
        self.jump_to_history_turn(turn)

    def action_load_older(self) -> None:
        self.load_older_history()

    def action_jump_latest(self) -> None:
        self.jump_to_latest()

    def action_focus_history_turn(self) -> None:
        turn_input = self.query_one("#history-turn", Input)
        if not turn_input.disabled:
            turn_input.focus()

    async def _sync_timeline_rows(self) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        items = timeline.display_items()
        await self.transcript.replace_history_blocks(
            [(kind, text, turn) for kind, text, turn in items]
        )
        self._history_total_turns = timeline.total_turns

    async def _seek_timeline_turn(
        self,
        target: int,
        *,
        pin_latest: bool = False,
        epoch: int | None = None,
        session: object | None = None,
    ) -> None:
        """Slide the hydrated window to ``target`` and replace transcript history rows."""

        timeline = self._timeline
        if timeline is None:
            return
        await timeline.slide_to(target)
        if epoch is not None and (epoch != self._chat_epoch or self._session is not session):
            return
        await self._sync_timeline_rows()
        if pin_latest:
            self.transcript.jump_to_latest()
        else:
            self.transcript.jump_to_turn(target)

    @work(exclusive=True, group="history")
    async def jump_to_history_turn(self, turn: int) -> None:
        """Seek the timeline to a one-based turn and scroll it to the top."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        total = self._history_total_turns
        if timeline is None or session is None or self._history_loading or not 1 <= turn <= total:
            return

        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(turn - 1, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not jump to turn {turn}: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def load_older_history(self) -> None:
        """Seek the previous TurnBegin, sliding the hydrated window."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, self._display_turn() - 1)
        first = timeline.first_materialized_turn()
        if current <= 0:
            if first <= 0:
                return
            target = first - 1
        else:
            target = current - 1

        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(target, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load older history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def load_newer_history(self) -> None:
        """Seek the next TurnBegin, sliding the hydrated window."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        total = self._history_total_turns
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, self._display_turn() - 1)
        if current + 1 >= total:
            return

        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(current + 1, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load newer history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def prefetch_older_history(self) -> None:
        """Slide toward older unmaterialized turns when the window hits the top."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        if timeline is None or session is None or self._history_loading:
            return
        first = timeline.first_materialized_turn()
        if first <= 0:
            return

        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(first - 1, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load older history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def prefetch_newer_history(self) -> None:
        """Slide toward newer unmaterialized turns when the window hits the bottom."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        total = self._history_total_turns
        if timeline is None or session is None or self._history_loading:
            return
        last = timeline.last_materialized_turn()
        if last + 1 >= total:
            return

        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(last + 1, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load newer history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def jump_to_latest(self) -> None:
        """Slide to the tail window and pin to the live stream immediately."""

        timeline = self._timeline
        session = self._session
        epoch = self._chat_epoch
        if timeline is None or session is None or self._history_loading:
            self.transcript.jump_to_latest()
            return
        total = self._history_total_turns
        if (
            total > 0
            and self.transcript.pinned_to_latest
            and (self.transcript.viewport_turn() is None or self.transcript.viewport_turn() >= total - 1)
        ):
            self.transcript.jump_to_latest()
            self._update_history_toolbar()
            return
        self._history_loading = True
        self._update_history_toolbar()
        try:
            await self._seek_timeline_turn(max(0, total - 1), pin_latest=True, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001 - history seek must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not jump to latest history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    def _set_status(self, text: str) -> None:
        if not self.is_attached:
            return
        if self._pending_config_label:
            text += f" · next: {self._pending_config_label}"
        with suppress(NoMatches):
            self.query_one("#status", Static).update(text)

    def _set_context(self, text: str) -> None:
        if not self.is_attached:
            return
        with suppress(NoMatches):
            self.query_one("#context", Static).update(text)

    def set_pending_config(self, label: str) -> None:
        self._pending_config_label = label
        session = self._session
        if session is not None:
            detail = self._last_wire_status or format_status(session.status)
            self._set_status(f"session {session.id}")
            self._set_context(detail)

    def _set_input_enabled(self, enabled: bool) -> None:
        if not self.is_attached:
            return
        with suppress(NoMatches):
            prompt = self.query_one("#prompt", PromptInput)
            prompt.disabled = not enabled
            if enabled:
                prompt.focus()

    def _restore_idle_ui(self, session: SdkSession) -> None:
        if self._session is not session:
            return
        self._set_input_enabled(True)
        detail = self._last_wire_status or format_status(session.status)
        self._set_status(f"session {session.id}")
        self._set_context(detail)

    async def _append_transcript(self, kind: str, text: str) -> None:
        transcript = self._mounted_transcript()
        if transcript is None:
            return
        await transcript.append_block(kind, text)

    @on(PromptInput.Submitted, "#prompt")
    async def submit_prompt(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if not text or self._session is None or self._busy:
            return
        event.prompt.clear()
        if text.startswith("/"):
            self.run_command(text)
        else:
            await self.transcript.append_block("user", text)
            self.run_prompt(text)

    @work(exclusive=True, group="prompt")
    async def run_prompt(self, text: str) -> None:
        session = self._session
        if session is None:
            return
        self._busy = True
        self._set_input_enabled(False)
        self._set_status(f"session {session.id} · running")
        try:
            async for message in session.prompt(text, merge_wire_messages=False):
                if self._session is not session:
                    return
                if isinstance(message, ApprovalRequest):
                    await self._render_message(message)
                    await self._handle_approval(message)
                    continue
                if is_request(message):
                    await self._render_message(message)
                    await self._handle_other_request(message)
                    continue
                await self._render_message(message)
        except RunCancelled:
            if self._session is session:
                await self._append_transcript("system", "Generation cancelled")
        except Exception as exc:  # noqa: BLE001 - surface SDK failures in the transcript
            if self._session is session:
                await self._append_transcript("error", f"{type(exc).__name__}: {exc}")
        finally:
            self._busy = False
            transcript = self._mounted_transcript()
            if transcript is not None:
                transcript.finish_stream()
            self._restore_idle_ui(session)

    async def _handle_approval(self, request: ApprovalRequest) -> None:
        transcript = self._mounted_transcript()
        if transcript is not None:
            transcript.finish_stream()
        decision = await self.app.push_screen_wait(
            ApprovalScreen(f"Approve {request.action}?", request.description)
        )
        request.resolve(decision)  # type: ignore[arg-type]
        await self._append_transcript(
            "system",
            f"Approval decision: {decision}\nRequest ID: {request.id}",
        )

    async def _render_message(self, message: object) -> None:
        rendered = render_wire_message(message, state=self._render_state)
        if rendered is None:
            return
        await self._append_rendered(rendered)

    async def _append_rendered(self, rendered: RenderEvent) -> None:
        if rendered.kind == "status":
            self._last_wire_status = rendered.text
            self._set_context(rendered.text)
            return
        transcript = self._mounted_transcript()
        if transcript is None:
            return
        if rendered.starts_stream:
            transcript.finish_stream()
        if rendered.streaming:
            await transcript.append_stream(
                rendered.kind,
                rendered.text,
                replace=rendered.replaces_stream,
            )
        else:
            await transcript.append_block(rendered.kind, rendered.text)

    async def _handle_other_request(self, request: object) -> None:
        """Handle request variants without importing private SDK modules."""

        request_name = type(request).__name__
        if request_name == "QuestionRequest":
            answers: dict[str, str] = {}
            for question in getattr(request, "questions", []):
                answer = await self.app.push_screen_wait(QuestionScreen(question))
                if answer is None:
                    set_exception = getattr(request, "set_exception", None)
                    if callable(set_exception):
                        set_exception(RuntimeError("Question cancelled by user"))
                    await self.transcript.append_block(
                        "error",
                        f"Question cancelled\nRequest ID: {getattr(request, 'id', '')}",
                    )
                    return
                answers[str(getattr(question, "question", "Question"))] = answer
            self._resolve_request(request, answers)
            rendered_answers = "\n".join(
                f"{question}: {answer}" for question, answer in answers.items()
            )
            await self.transcript.append_block("system", f"Question response\n{rendered_answers}")
            return

        if request_name == "HookRequest":
            decision = await self.app.push_screen_wait(
                ApprovalScreen(
                    f"Allow hook {getattr(request, 'event', '')}?",
                    str(getattr(request, "target", "")),
                )
            )
            action = "allow" if decision != "reject" else "block"
            self._resolve_request(request, action, "")
            await self.transcript.append_block(
                "system" if action == "allow" else "error",
                f"Hook decision: {action}\nRequest ID: {getattr(request, 'id', '')}",
            )
            return

        if request_name == "ToolCallRequest":
            error = ToolError(
                message="External client-side tools are not supported by this TUI prototype",
                brief="Unsupported external tool",
            )
            self._resolve_request(request, error)
            await self.transcript.append_block("error", error.message)
            return

        await self.transcript.append_block("error", f"Unsupported SDK request: {request_name}")
        if self._session is not None:
            self._session.cancel()

    @staticmethod
    def _resolve_request(request: object, *args: object) -> None:
        resolver = getattr(request, "resolve", None)
        if not callable(resolver):
            raise TypeError(f"SDK request {type(request).__name__} has no resolver")
        resolver(*args)

    @work(exclusive=True, group="command")
    async def run_command(self, command: str) -> None:
        session = self._session
        if session is None:
            return
        name, _, argument = command.partition(" ")
        if name == "/quit":
            self.action_leave_session()
            return
        if name == "/help":
            await self.transcript.append_block(
                "system",
                "/help  /status  /clear  /compact [instruction]  /quit (back to home)",
            )
            return
        if name == "/status":
            await self.transcript.append_block(
                "system", self._last_wire_status or format_status(session.status)
            )
            return

        self._busy = True
        self._set_input_enabled(False)
        try:
            if name == "/clear":
                await session.clear()
                await self.transcript.clear_messages()
                self._reset_history_state()
                self._update_history_toolbar()
                await self.transcript.append_block("system", "Session context cleared")
            elif name == "/compact":
                await self.transcript.append_block("system", "Compacting context…")
                await session.compact(custom_instruction=argument.strip())
                await self.transcript.append_block("system", "Context compacted")
            else:
                await self.transcript.append_block("error", f"Unknown command: {name}")
        except Exception as exc:  # noqa: BLE001 - commands report errors without exiting
            await self._append_transcript("error", f"{type(exc).__name__}: {exc}")
        finally:
            self._busy = False
            self._restore_idle_ui(session)

    def action_cancel_prompt(self) -> None:
        if self._session is not None and self._busy:
            self._session.cancel()
            self._set_status(f"session {self._session.id} · cancelling…")
        else:
            self.action_focus_prompt()

    def action_focus_prompt(self) -> None:
        if not self._busy:
            self.query_one("#prompt", PromptInput).focus()

    def action_settings(self) -> None:
        session_id = self._session.id if self._session is not None else self._options.session_id
        if session_id is not None:
            self.post_message(OpenLLMSettings(session_id))

    def action_leave_session(self) -> None:
        """Close this chat screen and return to the home screen."""

        if self._leaving:
            return
        self._leaving = True
        if self._session is not None:
            self._session.cancel()
        self.workers.cancel_group(self, "prompt")
        self.workers.cancel_group(self, "history")
        self.leave_session()

    @work(exclusive=True, group="leave")
    async def leave_session(self) -> None:
        await self._release_session()
        self.dismiss()

    async def _release_session(self) -> None:
        self._chat_epoch += 1
        session = self._session
        self._session = None
        self._busy = False
        self._reset_history_state()
        await self._close_sdk_session(session)

    @staticmethod
    async def _close_sdk_session(session: SdkSession | None) -> None:
        if session is None:
            return
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            with suppress(Exception):
                cancel()
        with suppress(Exception):
            await close_sdk_session(session)

    async def on_unmount(self) -> None:
        await self._release_session()

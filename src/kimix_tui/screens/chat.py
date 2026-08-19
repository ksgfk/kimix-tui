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
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Static

from kimix_tui.backend import (
    SdkSession,
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
)
from kimix_tui.history import (
    HISTORY_PAGE_TURNS,
    MAX_HISTORY_BLOCKS,
    MAX_HISTORY_WINDOW_TURNS,
    HistoryLoader,
    SessionHistory,
    WireHistoryPager,
    create_history_pager,
)
from kimix_tui.rendering import RenderEvent, RenderState, format_status, render_wire_message
from kimix_tui.screens.requests import ApprovalScreen, QuestionScreen
from kimix_tui.screens.settings import OpenLLMSettings
from kimix_tui.widgets import Transcript

SessionFactory = Callable[[SessionOptions], Awaitable[SdkSession]]
SessionOpenedCallback = Callable[[str], None]


class ChatScreen(Screen[None]):
    """Run one SDK session inside a full-screen chat interface."""

    CSS = """
    ChatScreen {
        layout: vertical;
        background: $background;
    }

    #status {
        height: 1;
        padding: 0 3;
        color: $text-muted;
        background: $panel;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #history-toolbar {
        display: none;
        height: 3;
        padding: 0 3;
        background: $surface;
        border-bottom: solid $panel-lighten-1;
        align: left middle;
    }

    #history-info {
        width: 1fr;
        height: 3;
        color: $text-muted;
        content-align: left middle;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    #history-actions {
        width: auto;
        height: 3;
        align: right middle;
    }

    #history-actions Button {
        height: 3;
        min-width: 13;
        margin-left: 1;
    }

    #history-turn {
        width: 10;
        min-width: 8;
        height: 3;
        margin-left: 1;
        padding: 0 1;
        content-align: left middle;
        border: tall $panel-lighten-1;
        background: $panel;
    }

    #history-turn:focus {
        border: tall $accent;
    }

    #jump-latest {
        min-width: 10;
    }

    ChatScreen.-narrow #history-toolbar {
        padding: 0 1;
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
        width: 8;
        min-width: 6;
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
        height: 3;
        margin: 1 2;
        padding: 0 1;
        border: round $accent;
        background: $panel;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+g", "cancel_prompt", "Cancel", priority=True),
        Binding("ctrl+up", "load_older", "Earlier"),
        Binding("ctrl+end", "jump_latest", "Latest"),
        Binding("f2", "focus_prompt", "Prompt"),
        Binding("f3", "focus_history_turn", "Turn"),
        Binding("f4", "settings", "Settings"),
        Binding("escape", "leave_session", "Home"),
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
        self._history_pager: WireHistoryPager | None = None
        self._history_start_turn = 0
        self._history_end_turn = 0
        self._history_total_turns = 0
        self._history_loading = False
        self._history_legacy_omitted = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("connecting…", id="status")
        with Horizontal(id="history-toolbar"):
            yield Static("", id="history-info", markup=False)
            with Horizontal(id="history-actions"):
                yield Button(
                    "← Earlier",
                    id="load-older",
                    compact=True,
                    tooltip="Load earlier turns",
                )
                yield Input(
                    placeholder="Turn #",
                    id="history-turn",
                    type="integer",
                    restrict=r"[0-9]*",
                    disabled=True,
                    tooltip="Jump to turn",
                )
                yield Button(
                    "Later →",
                    id="load-newer",
                    compact=True,
                    disabled=True,
                    tooltip="Load later turns",
                )
                yield Button(
                    "↓ Latest",
                    id="jump-latest",
                    compact=True,
                    disabled=True,
                    tooltip="Return to the latest turn",
                )
        yield Transcript(id="transcript")
        yield Input(placeholder="Ask AI, or type /help", id="prompt", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
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

    @work(exclusive=True, group="session")
    async def open_session(self) -> None:
        epoch = self._chat_epoch
        try:
            session = await self._session_factory(self._options)
        except Exception as exc:  # noqa: BLE001 - keep UI alive on SDK startup errors
            if epoch != self._chat_epoch:
                return
            await self.transcript.append_block("error", f"Failed to open session: {exc}")
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
                await self.transcript.append_block(
                    "error", f"Failed to save session configuration metadata: {exc}"
                )
        self._set_status(f"session {session.id} · {format_status(session.status)}")
        if epoch != self._chat_epoch:
            return
        await self.transcript.append_block("system", f"Session: {session.id}")
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
        pager: WireHistoryPager | None = None
        if self._history_loader is None:
            self._history_loading = True
            self._update_history_toolbar()
        try:
            if self._history_loader is None:
                pager = await create_history_pager(
                    self._options.work_dir,
                    session.id,
                    page_turns=HISTORY_PAGE_TURNS,
                    max_blocks=MAX_HISTORY_BLOCKS,
                )
                if pager is None:
                    self._history_loading = False
                    self._update_history_toolbar()
                    return
                history = await pager.latest()
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
            self._history_pager = pager
            await self._mount_history_page(history)
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

    async def _mount_history_page(self, history: SessionHistory) -> None:
        """Mount the initial bounded page and record its transcript slice."""

        history_start = len(self.transcript.records)
        if history.blocks:
            await self.transcript.append_blocks(
                [(block.kind, block.text) for block in history.blocks]
            )
        self.transcript.mark_history_window(history_start, len(self.transcript.records))
        self._history_start_turn = history.start_turn
        self._history_end_turn = history.end_turn
        self._history_total_turns = history.total_turns
        self._history_legacy_omitted = 0
        self._history_loading = False
        self._update_history_toolbar()

    def _reset_history_state(self) -> None:
        self._history_pager = None
        self._history_start_turn = 0
        self._history_end_turn = 0
        self._history_total_turns = 0
        self._history_loading = False
        self._history_legacy_omitted = 0

    def _update_history_toolbar(self) -> None:
        toolbar = self.query_one("#history-toolbar", Horizontal)
        info = self.query_one("#history-info", Static)
        older = self.query_one("#load-older", Button)
        turn_input = self.query_one("#history-turn", Input)
        newer = self.query_one("#load-newer", Button)
        latest = self.query_one("#jump-latest", Button)
        pager = self._history_pager

        if pager is None:
            if self._history_loading:
                toolbar.display = True
                info.update("History · loading…")
                older.disabled = True
                turn_input.disabled = True
                newer.disabled = True
                latest.disabled = True
                return
            if self._history_legacy_omitted:
                toolbar.display = True
                info.update(
                    f"History · {self._history_legacy_omitted} earlier turns unavailable"
                )
            else:
                toolbar.display = False
            older.disabled = True
            turn_input.disabled = True
            newer.disabled = True
            latest.disabled = True
            return

        total = self._history_total_turns
        start = self._history_start_turn
        end = self._history_end_turn
        if total <= 0:
            toolbar.display = False
            older.disabled = True
            turn_input.disabled = True
            newer.disabled = True
            latest.disabled = True
            return

        narrow = self.has_class("-narrow")
        window_full = end - start >= MAX_HISTORY_WINDOW_TURNS and start > 0
        details: list[str] = []
        if start:
            details.append(f"{start} older")
        if total - end:
            details.append(f"{total - end} newer")
        if start == 0 and end == total:
            range_text = f"all {total} turns"
        else:
            range_text = f"turns {start + 1}-{end} of {total}"
        info.update("History · " + range_text + (f" · {' · '.join(details)}" if details else ""))
        toolbar_visible = total > HISTORY_PAGE_TURNS or self._history_loading
        toolbar.display = toolbar_visible
        if narrow:
            older.label = "←"
            newer.label = "→"
            latest.label = "↓"
        else:
            older.label = "← Earlier page" if window_full else "← Earlier"
            newer.label = "Later →"
            latest.label = "↓ Latest"
        older.disabled = self._history_loading or start == 0
        turn_input.placeholder = f"Turn 1-{total}"
        turn_input.disabled = self._history_loading or not toolbar_visible
        newer.disabled = self._history_loading or end >= total
        latest.disabled = self._history_loading or (end >= total and start == max(0, total - MAX_HISTORY_WINDOW_TURNS))
        if self._history_loading:
            info.update("History · loading…")

    @on(Transcript.ReachedTop)
    def _transcript_reached_top(self, event: Transcript.ReachedTop) -> None:
        event.stop()
        if self._history_pager is None or self._history_loading:
            return
        if self._history_start_turn == 0:
            return
        self.load_older_history()

    @on(Transcript.ReachedBottom)
    def _transcript_reached_bottom(self, event: Transcript.ReachedBottom) -> None:
        event.stop()
        if self._history_pager is None or self._history_loading:
            return
        if self._history_end_turn >= self._history_total_turns:
            return
        self.load_newer_history()

    @on(Button.Pressed, "#load-older")
    def press_load_older(self) -> None:
        self.load_older_history()

    @on(Button.Pressed, "#load-newer")
    def press_load_newer(self) -> None:
        self.load_newer_history()

    @on(Button.Pressed, "#jump-latest")
    def press_jump_latest(self) -> None:
        self.jump_to_latest()

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

    @work(exclusive=True, group="history")
    async def jump_to_history_turn(self, turn: int) -> None:
        """Load a bounded window starting at a one-based turn number."""

        pager = self._history_pager
        session = self._session
        epoch = self._chat_epoch
        total = self._history_total_turns
        if pager is None or session is None or self._history_loading or not 1 <= turn <= total:
            return

        self._history_loading = True
        self._update_history_toolbar()
        try:
            start_turn = turn - 1
            end_turn = min(total, start_turn + MAX_HISTORY_WINDOW_TURNS)
            page = await pager.ending_at(
                end_turn,
                page_turns=end_turn - start_turn,
            )
            if epoch != self._chat_epoch or self._session is not session:
                return
            await self.transcript.replace_history_blocks(
                [(block.kind, block.text) for block in page.blocks]
            )
            self._history_start_turn = page.start_turn
            self._history_end_turn = page.end_turn
            self._history_total_turns = page.total_turns
            self.transcript.jump_to_history_start()
        except Exception as exc:  # noqa: BLE001 - history paging must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not jump to turn {turn}: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def load_older_history(self) -> None:
        """Load one older page, or replace the bounded window at its edge."""

        pager = self._history_pager
        session = self._session
        epoch = self._chat_epoch
        if (
            pager is None
            or session is None
            or self._history_loading
            or self._history_start_turn <= 0
        ):
            return

        self._history_loading = True
        self._update_history_toolbar()
        current_start = self._history_start_turn
        current_end = self._history_end_turn
        loaded_turns = current_end - current_start
        try:
            if loaded_turns >= MAX_HISTORY_WINDOW_TURNS:
                page = await pager.before(
                    current_start,
                    page_turns=min(MAX_HISTORY_WINDOW_TURNS, current_start),
                )
                if epoch != self._chat_epoch or self._session is not session:
                    return
                await self.transcript.replace_history_blocks(
                    [(block.kind, block.text) for block in page.blocks]
                )
                self._history_start_turn = page.start_turn
                self._history_end_turn = page.end_turn
                self._history_total_turns = page.total_turns
                self.transcript.jump_to_history_end()
                return

            page_size = min(HISTORY_PAGE_TURNS, MAX_HISTORY_WINDOW_TURNS - loaded_turns)
            page = await pager.before(current_start, page_turns=page_size)
            if epoch != self._chat_epoch or self._session is not session:
                return
            added_lines = await self.transcript.prepend_history_blocks(
                [(block.kind, block.text) for block in page.blocks]
            )
            if page.blocks and added_lines == 0:
                # The transcript character budget is full. Replace the bounded
                # window instead of allowing an unbounded prepend to grow it.
                page = await pager.before(
                    current_start,
                    page_turns=MAX_HISTORY_WINDOW_TURNS,
                )
                if epoch != self._chat_epoch or self._session is not session:
                    return
                await self.transcript.replace_history_blocks(
                    [(block.kind, block.text) for block in page.blocks]
                )
                self._history_start_turn = page.start_turn
                self._history_end_turn = page.end_turn
                self.transcript.jump_to_history_end()
            else:
                self._history_start_turn = page.start_turn
                self._history_end_turn = current_end
            self._history_total_turns = page.total_turns
        except Exception as exc:  # noqa: BLE001 - history paging must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load older history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def load_newer_history(self) -> None:
        """Replace the bounded window with the next newer history page."""

        pager = self._history_pager
        session = self._session
        epoch = self._chat_epoch
        if (
            pager is None
            or session is None
            or self._history_loading
            or self._history_end_turn >= self._history_total_turns
        ):
            return
        self._history_loading = True
        self._update_history_toolbar()
        try:
            start_turn = self._history_end_turn
            end_turn = min(
                self._history_total_turns,
                start_turn + MAX_HISTORY_WINDOW_TURNS,
            )
            page = await pager.ending_at(
                end_turn,
                page_turns=end_turn - start_turn,
            )
            if epoch != self._chat_epoch or self._session is not session:
                return
            await self.transcript.replace_history_blocks(
                [(block.kind, block.text) for block in page.blocks]
            )
            self._history_start_turn = page.start_turn
            self._history_end_turn = page.end_turn
            self._history_total_turns = page.total_turns
            self.transcript.jump_to_history_start()
        except Exception as exc:  # noqa: BLE001 - history paging must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not load newer history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    @work(exclusive=True, group="history")
    async def jump_to_latest(self) -> None:
        """Show the newest bounded history window and return to its tail."""

        pager = self._history_pager
        session = self._session
        epoch = self._chat_epoch
        if pager is None or session is None or self._history_loading:
            return
        if (
            self._history_end_turn >= self._history_total_turns
            and self._history_start_turn
            == max(0, self._history_total_turns - MAX_HISTORY_WINDOW_TURNS)
        ):
            self.transcript.jump_to_latest()
            return
        self._history_loading = True
        self._update_history_toolbar()
        try:
            page = await pager.latest(page_turns=MAX_HISTORY_WINDOW_TURNS)
            if epoch != self._chat_epoch or self._session is not session:
                return
            await self.transcript.replace_history_blocks(
                [(block.kind, block.text) for block in page.blocks]
            )
            self._history_start_turn = page.start_turn
            self._history_end_turn = page.end_turn
            self._history_total_turns = page.total_turns
            self.transcript.jump_to_latest()
        except Exception as exc:  # noqa: BLE001 - history paging must not kill chat
            if epoch == self._chat_epoch and self._session is session:
                self.notify(f"Could not jump to latest history: {exc}", severity="error")
        finally:
            if epoch == self._chat_epoch and self._session is session:
                self._history_loading = False
                self._update_history_toolbar()

    def _set_status(self, text: str) -> None:
        if self._pending_config_label:
            text += f" · next: {self._pending_config_label}"
        self.query_one("#status", Static).update(text)

    def set_pending_config(self, label: str) -> None:
        self._pending_config_label = label
        session = self._session
        if session is not None:
            detail = self._last_wire_status or format_status(session.status)
            self._set_status(f"session {session.id} · {detail}")

    def _set_input_enabled(self, enabled: bool) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = not enabled
        if enabled:
            prompt.focus()

    @on(Input.Submitted, "#prompt")
    async def submit_prompt(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._session is None or self._busy:
            return
        event.input.value = ""
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
                await self.transcript.append_block("system", "Generation cancelled")
        except Exception as exc:  # noqa: BLE001 - surface SDK failures in the transcript
            if self._session is session:
                await self.transcript.append_block("error", f"{type(exc).__name__}: {exc}")
        finally:
            self.transcript.finish_stream()
            self._busy = False
            if self._session is session:
                self._set_input_enabled(True)
                detail = self._last_wire_status or format_status(session.status)
                self._set_status(f"session {session.id} · {detail}")

    async def _handle_approval(self, request: ApprovalRequest) -> None:
        self.transcript.finish_stream()
        decision = await self.app.push_screen_wait(
            ApprovalScreen(f"Approve {request.action}?", request.description)
        )
        request.resolve(decision)  # type: ignore[arg-type]
        await self.transcript.append_block(
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
            session = self._session
            prefix = f"session {session.id} · " if session is not None else ""
            self._last_wire_status = rendered.text
            self._set_status(prefix + rendered.text)
            return
        if rendered.starts_stream:
            self.transcript.finish_stream()
        if rendered.streaming:
            await self.transcript.append_stream(
                rendered.kind,
                rendered.text,
                replace=rendered.replaces_stream,
            )
        else:
            await self.transcript.append_block(rendered.kind, rendered.text)

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
            await self.transcript.append_block("error", f"{type(exc).__name__}: {exc}")
        finally:
            self._busy = False
            if self._session is session:
                self._set_input_enabled(True)
                detail = self._last_wire_status or format_status(session.status)
                self._set_status(f"session {session.id} · {detail}")

    def action_cancel_prompt(self) -> None:
        if self._session is not None and self._busy:
            self._session.cancel()
            self._set_status(f"session {self._session.id} · cancelling…")
        else:
            self.action_focus_prompt()

    def action_focus_prompt(self) -> None:
        if not self._busy:
            self.query_one("#prompt", Input).focus()

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

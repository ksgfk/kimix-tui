"""Full-screen chat experience and SDK session lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import ClassVar

from kimi_agent_sdk import ApprovalRequest, RunCancelled, ToolError, is_request
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from kimix_tui.backend import SdkSession, SessionOptions, create_sdk_session
from kimix_tui.history import HistoryLoader, load_session_history
from kimix_tui.rendering import format_status, render_wire_message
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
        background: $surface;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }

    #transcript {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }

    .message {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 1;
        margin-bottom: 1;
    }

    .thinking { color: $text-muted; }
    .user { border-left: thick $secondary; }
    .tool { border-left: thick $accent; background: $boost; }
    .tool_result { border-left: thick $primary; }
    .error { border-left: thick $error; }
    .approval { border-left: thick $warning; }

    #prompt {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+g", "cancel_prompt", "Cancel", priority=True),
        Binding("f2", "focus_prompt", "Prompt"),
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("connecting…", id="status")
        yield Transcript(id="transcript")
        yield Input(placeholder="Ask Kimi, or type /help", id="prompt", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.open_session()

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
        self._set_input_enabled(True)
        self._set_status(f"session {session.id} · {format_status(session.status)}")
        if epoch != self._chat_epoch:
            return
        await self.transcript.append_block("system", f"Session: {session.id}")
        if epoch != self._chat_epoch:
            return
        await self._replay_history()

    async def _replay_history(self) -> None:
        session = self._session
        epoch = self._chat_epoch
        if session is None:
            return
        try:
            loader = self._history_loader or load_session_history
            history = await loader(self._options.work_dir, session.id)
        except Exception as exc:  # noqa: BLE001 - chat still works without replay
            if epoch != self._chat_epoch or self._session is not session:
                return
            await self.transcript.append_block("error", f"Failed to load history: {exc}")
            return
        if epoch != self._chat_epoch or self._session is not session:
            return
        if history.omitted_turns:
            shown_turns = sum(1 for block in history.blocks if block.kind == "user")
            items: list[tuple[str, str]] = [
                (
                    "system",
                    f"Showing last {shown_turns} turns ({history.omitted_turns} earlier omitted)",
                )
            ]
        else:
            items = []
        items.extend((block.kind, block.text) for block in history.blocks)
        if items:
            await self.transcript.append_blocks(items)

    def _set_status(self, text: str) -> None:
        if self._pending_config_label:
            text += f" · next: {self._pending_config_label}"
        self.query_one("#status", Static).update(text)

    def set_pending_config(self, label: str) -> None:
        self._pending_config_label = label
        session = self._session
        if session is not None:
            self._set_status(f"session {session.id} · {format_status(session.status)}")

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
                    await self._handle_approval(message)
                    continue
                if is_request(message):
                    await self._handle_other_request(message)
                    continue

                rendered = render_wire_message(message)
                if rendered is None:
                    continue
                if rendered.kind == "status":
                    self._set_status(f"session {session.id} · {rendered.text}")
                elif rendered.streaming:
                    await self.transcript.append_stream(rendered.kind, rendered.text)
                else:
                    await self.transcript.append_block(rendered.kind, rendered.text)
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
                self._set_status(f"session {session.id} · {format_status(session.status)}")

    async def _handle_approval(self, request: ApprovalRequest) -> None:
        self.transcript.finish_stream()
        await self.transcript.append_block(
            "approval",
            f"{request.sender}: {request.action}\n{request.description}",
        )
        decision = await self.app.push_screen_wait(
            ApprovalScreen(f"Approve {request.action}?", request.description)
        )
        request.resolve(decision)  # type: ignore[arg-type]

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
                    return
                answers[str(getattr(question, "question", "Question"))] = answer
            self._resolve_request(request, answers)
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
            return

        if request_name == "ToolCallRequest":
            self._resolve_request(
                request,
                ToolError(
                    message="External client-side tools are not supported by this TUI prototype",
                    brief="Unsupported external tool",
                ),
            )
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
            await self.transcript.append_block("system", format_status(session.status))
            return

        self._busy = True
        self._set_input_enabled(False)
        try:
            if name == "/clear":
                await session.clear()
                await self.transcript.clear_messages()
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
                self._set_status(f"session {session.id} · {format_status(session.status)}")

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
            await session.close()

    async def on_unmount(self) -> None:
        await self._release_session()

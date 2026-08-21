"""Kimix worker thread: asyncio loop, coalesced transcript deltas, request futures."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from kimi_agent_sdk import ApprovalRequest, RunCancelled, ToolError, is_request
from PySide6.QtCore import QObject, Signal

from kimix_tui.backend import (
    SdkSession,
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
)
from kimix_tui.history import HistoryLoader, Timeline, create_timeline
from kimix_tui.rendering import RenderEvent, RenderState, format_status, render_wire_message
from kimix_tui.session_index import (
    SessionDeleter,
    SessionLoader,
    delete_sessions,
    list_session_summaries,
)

SessionFactory = Callable[[SessionOptions], Awaitable[SdkSession]]
SessionOpenedCallback = Callable[[str], None]

_COALESCE_SECONDS = 0.016


@dataclass(frozen=True, slots=True)
class TranscriptDelta:
    kind: str
    text: str
    streaming: bool = False
    starts_stream: bool = False
    replaces_stream: bool = False
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class HistoryPage:
    items: tuple[tuple[str, str, int], ...]
    total_turns: int
    target_turn: int | None
    pin_latest: bool
    epoch: int


@dataclass(frozen=True, slots=True)
class ApprovalAsk:
    title: str
    description: str
    token: int
    epoch: int


@dataclass(frozen=True, slots=True)
class QuestionAsk:
    prompt: str
    body: str
    token: int
    epoch: int


class StreamCoalescer:
    """Merge streaming fragments and flush about every 16ms."""

    def __init__(self, emit: Callable[[TranscriptDelta], None], loop: asyncio.AbstractEventLoop) -> None:
        self._emit = emit
        self._loop = loop
        self._kind: str | None = None
        self._text = ""
        self._starts = False
        self._replace = False
        self._epoch = 0
        self._handle: asyncio.TimerHandle | None = None

    @property
    def pending(self) -> bool:
        return self._kind is not None and bool(self._text)

    def feed(self, event: RenderEvent, epoch: int) -> None:
        if event.kind == "status":
            self.flush()
            self._emit(
                TranscriptDelta("status", event.text, epoch=epoch),
            )
            return
        if event.streaming:
            if event.starts_stream or self._kind != event.kind:
                self.flush()
                self._kind = event.kind
                self._text = event.text
                self._starts = True
                self._replace = event.replaces_stream
                self._epoch = epoch
            elif event.replaces_stream:
                self._text = event.text
                self._replace = True
                self._epoch = epoch
            else:
                self._text += event.text
                self._epoch = epoch
            self._schedule()
            return
        self.flush()
        self._emit(
            TranscriptDelta(
                event.kind,
                event.text,
                streaming=False,
                starts_stream=event.starts_stream,
                replaces_stream=event.replaces_stream,
                epoch=epoch,
            )
        )

    def flush(self) -> None:
        self._cancel_timer()
        if self._kind is None or not self._text:
            self._kind = None
            self._text = ""
            return
        self._emit(
            TranscriptDelta(
                self._kind,
                self._text,
                streaming=True,
                starts_stream=self._starts,
                replaces_stream=self._replace,
                epoch=self._epoch,
            )
        )
        self._starts = False
        self._replace = False
        self._text = ""

    def reset(self) -> None:
        self._cancel_timer()
        self._kind = None
        self._text = ""
        self._starts = False
        self._replace = False

    def _schedule(self) -> None:
        if self._handle is not None:
            return
        self._handle = self._loop.call_later(_COALESCE_SECONDS, self.flush)

    def _cancel_timer(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


class KimixBridge(QObject):
    """Owns a private asyncio loop. SDK objects never leave the worker thread."""

    session_opened = Signal(str, str, int)
    session_failed = Signal(str, int)
    session_closed = Signal(int)
    transcript_delta = Signal(object)
    history_page = Signal(object)
    history_loading = Signal(bool, int)
    generation_started = Signal(int)
    generation_finished = Signal(int)
    approval_asked = Signal(object)
    question_asked = Signal(object)
    notify = Signal(str, str, str)
    sessions_listed = Signal(object)
    sessions_list_failed = Signal(str)
    sessions_deleted = Signal(object)
    input_enabled = Signal(bool, int)

    def __init__(
        self,
        *,
        session_factory: SessionFactory = create_sdk_session,
        history_loader: HistoryLoader | None = None,
        session_loader: SessionLoader | None = None,
        session_deleter: SessionDeleter | None = None,
        on_session_opened: SessionOpenedCallback | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_factory = session_factory
        self._history_loader = history_loader
        self._session_loader = session_loader
        self._session_deleter = session_deleter
        self._on_session_opened = on_session_opened
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._tasks = 0
        self._busy = False
        self._epoch = 0
        self._session: SdkSession | None = None
        self._session_id: str | None = None
        self._options: SessionOptions | None = None
        self._timeline: Timeline | None = None
        self._history_total = 0
        self._history_loading = False
        self._render_state = RenderState()
        self._last_wire_status: str | None = None
        self._coalescer: StreamCoalescer | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._tokens = count(1)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="kimix-bridge", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        self.submit(self._shutdown())
        if self._thread is not None:
            self._thread.join(timeout=8)
        self._thread = None
        self._loop = None

    def is_idle(self) -> bool:
        with self._lock:
            pending = self._coalescer.pending if self._coalescer is not None else False
            return self._tasks == 0 and not self._busy and not pending

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    def submit(self, coro: Awaitable[object]) -> None:
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._tracked(coro), loop)

    def open_session(
        self,
        options: SessionOptions,
        *,
        on_session_opened: SessionOpenedCallback | None = None,
    ) -> None:
        if on_session_opened is not None:
            self._on_session_opened = on_session_opened
        self.submit(self._open_session(options))

    def close_session(self) -> None:
        self.submit(self._close_session())

    def run_prompt(self, text: str) -> None:
        self.submit(self._run_prompt(text))

    def run_command(self, command: str) -> None:
        self.submit(self._run_command(command))

    def cancel_prompt(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._cancel_prompt)

    def load_sessions(self, work_dir: Path) -> None:
        self.submit(self._load_sessions(work_dir))

    def delete_sessions(self, work_dir: Path, session_ids: list[str]) -> None:
        self.submit(self._delete_sessions(work_dir, session_ids))

    def jump_to_turn(self, turn: int) -> None:
        self.submit(self._jump_to_turn(turn))

    def load_older(self, current_turn: int) -> None:
        self.submit(self._load_older(current_turn))

    def load_newer(self, current_turn: int) -> None:
        self.submit(self._load_newer(current_turn))

    def prefetch_older(self) -> None:
        self.submit(self._prefetch_older())

    def prefetch_newer(self) -> None:
        self.submit(self._prefetch_newer())

    def jump_to_latest(self) -> None:
        self.submit(self._jump_to_latest())

    def resolve_request(self, token: int, epoch: int, value: object) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._resolve_request, token, epoch, value)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._coalescer = StreamCoalescer(self._emit_delta, loop)
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        with suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    async def _tracked(self, coro: Awaitable[object]) -> None:
        with self._lock:
            self._tasks += 1
        try:
            await coro
        finally:
            with self._lock:
                self._tasks -= 1

    async def _shutdown(self) -> None:
        await self._release_session()
        loop = self._loop
        if loop is not None:
            loop.stop()

    def _emit_delta(self, delta: TranscriptDelta) -> None:
        self.transcript_delta.emit(delta)

    def _cancel_prompt(self) -> None:
        session = self._session
        if session is not None and self._busy:
            with suppress(Exception):
                session.cancel()

    def _resolve_request(self, token: int, epoch: int, value: object) -> None:
        future = self._pending.pop(token, None)
        if future is None or future.done():
            return
        with self._lock:
            current = self._epoch
        if epoch != current:
            future.cancel()
            return
        future.set_result(value)

    async def _open_session(self, options: SessionOptions) -> None:
        epoch = self._bump_epoch()
        self._options = options
        self._render_state = RenderState()
        self._last_wire_status = None
        self._timeline = None
        self._history_total = 0
        try:
            session = await self._session_factory(options)
        except Exception as exc:  # noqa: BLE001
            if epoch != self.epoch:
                return
            self.session_failed.emit(f"Failed to open session: {exc}", epoch)
            return
        if epoch != self.epoch:
            await self._close_sdk(session)
            return
        with self._lock:
            self._session = session
            self._session_id = session.id
        if self._on_session_opened is not None:
            try:
                self._on_session_opened(session.id)
            except Exception as exc:  # noqa: BLE001
                self._emit_delta(
                    TranscriptDelta("error", f"Failed to save session configuration metadata: {exc}", epoch=epoch)
                )
        status = format_status(session.status)
        self.session_opened.emit(session.id, status, epoch)
        self._emit_delta(TranscriptDelta("system", f"Session: {session.id}", epoch=epoch))
        try:
            await self._replay_history(session, epoch)
        finally:
            if epoch == self.epoch and self._session is session:
                self.input_enabled.emit(True, epoch)

    async def _replay_history(self, session: SdkSession, epoch: int) -> None:
        if self._history_loader is None:
            self._history_loading = True
            self.history_loading.emit(True, epoch)
        try:
            if self._history_loader is None:
                assert self._options is not None
                timeline = await create_timeline(self._options.work_dir, session.id)
                if timeline is None:
                    self._history_loading = False
                    self.history_loading.emit(False, epoch)
                    return
                self._timeline = timeline
                await self._publish_history(pin_latest=True, epoch=epoch)
                return
            assert self._options is not None
            history = await self._history_loader(self._options.work_dir, session.id)
        except Exception as exc:  # noqa: BLE001
            self._history_loading = False
            self.history_loading.emit(False, epoch)
            if epoch != self.epoch or self._session is not session:
                return
            self._emit_delta(TranscriptDelta("error", f"Failed to load history: {exc}", epoch=epoch))
            return
        if epoch != self.epoch or self._session is not session:
            return
        if self._history_loader is not None:
            if history.omitted_turns:
                shown_turns = sum(1 for block in history.blocks if block.kind == "user")
                self._emit_delta(
                    TranscriptDelta(
                        "system",
                        f"Showing last {shown_turns} turns ({history.omitted_turns} earlier omitted)",
                        epoch=epoch,
                    )
                )
            items = tuple((block.kind, block.text, 0) for block in history.blocks)
            self.history_page.emit(
                HistoryPage(items=items, total_turns=0, target_turn=None, pin_latest=True, epoch=epoch)
            )
            self._history_loading = False
            self.history_loading.emit(False, epoch)

    async def _publish_history(
        self,
        *,
        target: int | None = None,
        pin_latest: bool = False,
        epoch: int,
    ) -> None:
        timeline = self._timeline
        if timeline is None:
            self._history_loading = False
            self.history_loading.emit(False, epoch)
            return
        items = tuple((kind, text, turn) for kind, text, turn in timeline.display_items())
        self._history_total = timeline.total_turns
        self._history_loading = False
        self.history_page.emit(
            HistoryPage(
                items=items,
                total_turns=timeline.total_turns,
                target_turn=target,
                pin_latest=pin_latest,
                epoch=epoch,
            )
        )
        self.history_loading.emit(False, epoch)

    async def _seek(self, target: int, *, pin_latest: bool, epoch: int, session: SdkSession) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        await timeline.slide_to(target)
        if epoch != self.epoch or self._session is not session:
            return
        await self._publish_history(target=target, pin_latest=pin_latest, epoch=epoch)

    async def _jump_to_turn(self, turn: int) -> None:
        session = self._session
        epoch = self.epoch
        timeline = self._timeline
        total = self._history_total
        if timeline is None or session is None or self._history_loading or not 1 <= turn <= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(turn - 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not jump to turn {turn}: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _load_older(self, current_turn: int) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, current_turn - 1)
        first = timeline.first_materialized_turn()
        if current <= 0:
            if first <= 0:
                return
            target = first - 1
        else:
            target = current - 1
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(target, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not load older history: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _load_newer(self, current_turn: int) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        total = self._history_total
        if timeline is None or session is None or self._history_loading:
            return
        current = max(0, current_turn - 1)
        if current + 1 >= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(current + 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not load newer history: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _prefetch_older(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            return
        first = timeline.first_materialized_turn()
        if first <= 0:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(first - 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not load older history: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _prefetch_newer(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        total = self._history_total
        if timeline is None or session is None or self._history_loading:
            return
        last = timeline.last_materialized_turn()
        if last + 1 >= total:
            return
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(last + 1, pin_latest=False, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not load newer history: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _jump_to_latest(self) -> None:
        timeline = self._timeline
        session = self._session
        epoch = self.epoch
        if timeline is None or session is None or self._history_loading:
            self.history_page.emit(
                HistoryPage(items=(), total_turns=self._history_total, target_turn=None, pin_latest=True, epoch=epoch)
            )
            return
        total = self._history_total
        self._history_loading = True
        self.history_loading.emit(True, epoch)
        try:
            await self._seek(max(0, total - 1), pin_latest=True, epoch=epoch, session=session)
        except Exception as exc:  # noqa: BLE001
            if epoch == self.epoch and self._session is session:
                self.notify.emit(f"Could not jump to latest history: {exc}", "error", "")
        finally:
            if epoch == self.epoch and self._session is session:
                self._history_loading = False
                self.history_loading.emit(False, epoch)

    async def _run_prompt(self, text: str) -> None:
        session = self._session
        epoch = self.epoch
        if session is None:
            return
        with self._lock:
            self._busy = True
        self.generation_started.emit(epoch)
        self.input_enabled.emit(False, epoch)
        coalescer = self._coalescer
        try:
            async for message in session.prompt(text, merge_wire_messages=False):
                if self._session is not session or epoch != self.epoch:
                    return
                if isinstance(message, ApprovalRequest):
                    await self._render_message(message, epoch)
                    await self._handle_approval(message, epoch)
                    continue
                if is_request(message):
                    await self._render_message(message, epoch)
                    await self._handle_other_request(message, epoch)
                    continue
                await self._render_message(message, epoch)
        except RunCancelled:
            if self._session is session and epoch == self.epoch:
                if coalescer:
                    coalescer.flush()
                self._emit_delta(TranscriptDelta("system", "Generation cancelled", epoch=epoch))
        except Exception as exc:  # noqa: BLE001
            if self._session is session and epoch == self.epoch:
                if coalescer:
                    coalescer.flush()
                self._emit_delta(
                    TranscriptDelta("error", f"{type(exc).__name__}: {exc}", epoch=epoch)
                )
        finally:
            if coalescer:
                coalescer.flush()
            with self._lock:
                self._busy = False
            self.generation_finished.emit(epoch)
            if self._session is session and epoch == self.epoch:
                self.input_enabled.emit(True, epoch)

    async def _render_message(self, message: object, epoch: int) -> None:
        rendered = render_wire_message(message, state=self._render_state)
        if rendered is None or self._coalescer is None:
            return
        self._coalescer.feed(rendered, epoch)
        if rendered.kind == "status":
            self._last_wire_status = rendered.text

    async def _handle_approval(self, request: ApprovalRequest, epoch: int) -> None:
        if self._coalescer:
            self._coalescer.flush()
        decision = await self._ask(
            ApprovalAsk(
                title=f"Approve {request.action}?",
                description=request.description,
                token=next(self._tokens),
                epoch=epoch,
            )
        )
        if decision is None or epoch != self.epoch:
            return
        request.resolve(decision)  # type: ignore[arg-type]
        self._emit_delta(
            TranscriptDelta(
                "system",
                f"Approval decision: {decision}\nRequest ID: {request.id}",
                epoch=epoch,
            )
        )

    async def _handle_other_request(self, request: object, epoch: int) -> None:
        request_name = type(request).__name__
        if request_name == "QuestionRequest":
            answers: dict[str, str] = {}
            for question in getattr(request, "questions", []):
                prompt = str(getattr(question, "question", "Question"))
                options = getattr(question, "options", [])
                option_lines = [
                    f"- {getattr(option, 'label', option)}"
                    + (
                        f": {getattr(option, 'description', '')}"
                        if getattr(option, "description", "")
                        else ""
                    )
                    for option in options
                ]
                answer = await self._ask(
                    QuestionAsk(
                        prompt=prompt,
                        body="\n".join(option_lines),
                        token=next(self._tokens),
                        epoch=epoch,
                    )
                )
                if answer is None:
                    set_exception = getattr(request, "set_exception", None)
                    if callable(set_exception):
                        set_exception(RuntimeError("Question cancelled by user"))
                    self._emit_delta(
                        TranscriptDelta(
                            "error",
                            f"Question cancelled\nRequest ID: {getattr(request, 'id', '')}",
                            epoch=epoch,
                        )
                    )
                    return
                answers[prompt] = str(answer)
            self._resolve_sdk_request(request, answers)
            rendered_answers = "\n".join(
                f"{question}: {answer}" for question, answer in answers.items()
            )
            self._emit_delta(
                TranscriptDelta("system", f"Question response\n{rendered_answers}", epoch=epoch)
            )
            return

        if request_name == "HookRequest":
            decision = await self._ask(
                ApprovalAsk(
                    title=f"Allow hook {getattr(request, 'event', '')}?",
                    description=str(getattr(request, "target", "")),
                    token=next(self._tokens),
                    epoch=epoch,
                )
            )
            if decision is None or epoch != self.epoch:
                return
            action = "allow" if decision != "reject" else "block"
            self._resolve_sdk_request(request, action, "")
            self._emit_delta(
                TranscriptDelta(
                    "system" if action == "allow" else "error",
                    f"Hook decision: {action}\nRequest ID: {getattr(request, 'id', '')}",
                    epoch=epoch,
                )
            )
            return

        if request_name == "ToolCallRequest":
            error = ToolError(
                message="External client-side tools are not supported by this TUI prototype",
                brief="Unsupported external tool",
            )
            self._resolve_sdk_request(request, error)
            self._emit_delta(TranscriptDelta("error", error.message, epoch=epoch))
            return

        self._emit_delta(
            TranscriptDelta("error", f"Unsupported SDK request: {request_name}", epoch=epoch)
        )
        if self._session is not None:
            self._session.cancel()

    async def _ask(self, ask: ApprovalAsk | QuestionAsk) -> object | None:
        loop = self._loop
        assert loop is not None
        future: asyncio.Future[object] = loop.create_future()
        self._pending[ask.token] = future
        if isinstance(ask, ApprovalAsk):
            self.approval_asked.emit(ask)
        else:
            self.question_asked.emit(ask)
        try:
            return await future
        except asyncio.CancelledError:
            return None

    @staticmethod
    def _resolve_sdk_request(request: object, *args: object) -> None:
        resolver = getattr(request, "resolve", None)
        if not callable(resolver):
            raise TypeError(f"SDK request {type(request).__name__} has no resolver")
        resolver(*args)

    async def _run_command(self, command: str) -> None:
        session = self._session
        epoch = self.epoch
        if session is None:
            return
        name, _, argument = command.partition(" ")
        if name == "/quit":
            await self._close_session()
            return
        if name == "/help":
            self._emit_delta(
                TranscriptDelta(
                    "system",
                    "/help  /status  /clear  /compact [instruction]  /quit (back to home)",
                    epoch=epoch,
                )
            )
            return
        if name == "/status":
            self._emit_delta(
                TranscriptDelta(
                    "system",
                    self._last_wire_status or format_status(session.status),
                    epoch=epoch,
                )
            )
            return
        with self._lock:
            self._busy = True
        self.input_enabled.emit(False, epoch)
        try:
            if name == "/clear":
                await session.clear()
                self._timeline = None
                self._history_total = 0
                self._emit_delta(TranscriptDelta("system", "__clear__", epoch=epoch))
                self._emit_delta(TranscriptDelta("system", "Session context cleared", epoch=epoch))
            elif name == "/compact":
                self._emit_delta(TranscriptDelta("system", "Compacting context…", epoch=epoch))
                await session.compact(custom_instruction=argument.strip())
                self._emit_delta(TranscriptDelta("system", "Context compacted", epoch=epoch))
            else:
                self._emit_delta(TranscriptDelta("error", f"Unknown command: {name}", epoch=epoch))
        except Exception as exc:  # noqa: BLE001
            self._emit_delta(
                TranscriptDelta("error", f"{type(exc).__name__}: {exc}", epoch=epoch)
            )
        finally:
            with self._lock:
                self._busy = False
            if self._session is session and epoch == self.epoch:
                self.input_enabled.emit(True, epoch)

    async def _close_session(self) -> None:
        epoch = await self._release_session()
        self.session_closed.emit(epoch)

    async def _release_session(self) -> int:
        epoch = self._bump_epoch()
        session = self._session
        with self._lock:
            self._session = None
            self._session_id = None
            self._busy = False
        self._timeline = None
        self._history_total = 0
        if self._coalescer:
            self._coalescer.reset()
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()
        await self._close_sdk(session)
        return epoch

    def _bump_epoch(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    @staticmethod
    async def _close_sdk(session: SdkSession | None) -> None:
        if session is None:
            return
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            with suppress(Exception):
                cancel()
        with suppress(Exception):
            await close_sdk_session(session)
        close = getattr(session, "close", None)
        if callable(close) and not getattr(session, "closed", False):
            with suppress(Exception):
                await close()

    async def _load_sessions(self, work_dir: Path) -> None:
        try:
            loader = self._session_loader or list_session_summaries
            summaries = sorted(
                await loader(work_dir),
                key=lambda summary: summary.updated_at,
                reverse=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.sessions_list_failed.emit(str(exc))
            return
        self.sessions_listed.emit(summaries)

    async def _delete_sessions(self, work_dir: Path, session_ids: list[str]) -> None:
        try:
            await (self._session_deleter or delete_sessions)(work_dir, session_ids)
        except Exception as exc:  # noqa: BLE001
            self.notify.emit(f"Failed to delete sessions: {exc}", "error", "")
            return
        self.sessions_deleted.emit(session_ids)

    def current_status_detail(self, session_status: object | None = None) -> str:
        if self._last_wire_status:
            return self._last_wire_status
        if session_status is not None:
            return format_status(session_status)
        return "ready"

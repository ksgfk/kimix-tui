"""Replay recent session history from the kimi-cli wire log."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kaos.path import KaosPath

from kimix_tui.rendering import (
    RenderState,
    bounded_concat,
    render_wire_message,
    truncate_display,
    user_input_text,
)

MAX_HISTORY_TURNS = 4
MAX_HISTORY_BLOCKS = 32
_SKIP_RENDER_KINDS = {"status"}

HistoryLoader = Callable[[Path, str], Awaitable["SessionHistory"]]


@dataclass(frozen=True, slots=True)
class HistoryBlock:
    """One transcript row reconstructed from persisted wire messages."""

    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """Tail of a session's conversation, ready to mount in the TUI."""

    blocks: list[HistoryBlock]
    omitted_turns: int = 0


@dataclass
class HistoryAccumulator:
    """Fold wire messages into a bounded last-N-turns window.

    Only the retained turns stay in memory; earlier turns are discarded as
    soon as they fall out of the window.
    """

    max_turns: int = MAX_HISTORY_TURNS
    max_blocks: int = MAX_HISTORY_BLOCKS
    omitted_turns: int = 0
    _turns: deque[list[HistoryBlock]] = field(default_factory=deque)
    _current: list[HistoryBlock] = field(default_factory=list)
    _render_state: RenderState = field(default_factory=RenderState)

    def feed(self, message: object) -> None:
        name = type(message).__name__
        if name == "TurnBegin":
            self._flush_turn()
            text = user_input_text(getattr(message, "user_input", ""))
            if text:
                self._add("user", text, merge=False)
            return
        if name == "SteerInput":
            text = user_input_text(getattr(message, "user_input", ""))
            if text:
                self._add("user", text, merge=False)
            return
        rendered = render_wire_message(message, state=self._render_state)
        if rendered is None or rendered.kind in _SKIP_RENDER_KINDS:
            return
        self._add(
            rendered.kind,
            rendered.text,
            merge=rendered.streaming and not rendered.starts_stream,
            replace=rendered.replaces_stream,
        )

    def finish(self) -> SessionHistory:
        self._flush_turn()
        blocks = [block for turn in self._turns for block in turn]
        if self.max_blocks > 0 and len(blocks) > self.max_blocks:
            blocks = blocks[-self.max_blocks :]
        return SessionHistory(blocks=blocks, omitted_turns=self.omitted_turns)

    def _add(self, kind: str, text: str, *, merge: bool, replace: bool = False) -> None:
        if replace and self._current and self._current[-1].kind == kind:
            self._current[-1] = HistoryBlock(kind, truncate_display(text))
            return
        if merge and self._current and self._current[-1].kind == kind:
            previous = self._current[-1]
            self._current[-1] = HistoryBlock(kind, bounded_concat(previous.text, text))
            return
        self._current.append(HistoryBlock(kind, bounded_concat("", text)))

    def _flush_turn(self) -> None:
        if not self._current:
            return
        self._turns.append(self._current)
        self._current = []
        while self.max_turns > 0 and len(self._turns) > self.max_turns:
            self._turns.popleft()
            self.omitted_turns += 1


def blocks_from_wire_messages(messages: Sequence[object]) -> list[HistoryBlock]:
    """Merge streaming fragments and keep conversation-visible events."""

    accumulator = HistoryAccumulator(max_turns=0, max_blocks=0)
    for message in messages:
        accumulator.feed(message)
    return accumulator.finish().blocks


def take_last_turns(
    blocks: Sequence[HistoryBlock],
    *,
    max_turns: int = MAX_HISTORY_TURNS,
) -> SessionHistory:
    """Keep the most recent user turns; report how many earlier turns were dropped."""

    if max_turns <= 0 or not blocks:
        return SessionHistory(blocks=list(blocks), omitted_turns=0)
    user_indexes = [index for index, block in enumerate(blocks) if block.kind == "user"]
    if len(user_indexes) <= max_turns:
        return SessionHistory(blocks=list(blocks), omitted_turns=0)
    start = user_indexes[-max_turns]
    return SessionHistory(
        blocks=list(blocks[start:]),
        omitted_turns=len(user_indexes) - max_turns,
    )


async def _feed_wire_log(
    work_dir: Path,
    session_id: str,
    accumulator: HistoryAccumulator,
) -> None:
    from kimi_cli.session import Session as CliSession

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return
    async for record in cli_session.wire_file.iter_records():
        try:
            accumulator.feed(record.to_wire_message())
        except Exception:  # noqa: BLE001, S112 - skip unreadable historical records
            continue


async def load_session_history(
    work_dir: Path,
    session_id: str,
    *,
    max_turns: int = 0,
    max_blocks: int = 0,
    messages: Sequence[object] | None = None,
) -> SessionHistory:
    """Load the conversation of ``session_id`` from kimi-cli ``wire.jsonl``.

    The wire log is scanned sequentially and folded into truncated display
    blocks. ``max_turns`` / ``max_blocks`` of 0 keep the full compact history
    so the TUI can scroll back; wire objects are not retained.
    """

    accumulator = HistoryAccumulator(max_turns=max_turns, max_blocks=max_blocks)
    if messages is None:
        await _feed_wire_log(work_dir, session_id, accumulator)
    else:
        for message in messages:
            accumulator.feed(message)
    return accumulator.finish()

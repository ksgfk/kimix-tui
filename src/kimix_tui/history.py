"""Replay recent session history from the kimi-cli wire log."""

from __future__ import annotations

import asyncio
import re
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
from kimix_tui.transcript_paint import is_dialogue_record

MAX_HISTORY_TURNS = 4
MAX_HISTORY_BLOCKS = 32
HISTORY_PAGE_TURNS = MAX_HISTORY_TURNS
MAX_HISTORY_WINDOW_TURNS = 64
_SKIP_RENDER_KINDS = {"status"}
_TURN_BEGIN_RECORD = re.compile(
    rb'"message"\s*:\s*\{\s*"type"\s*:\s*"TurnBegin"\s*,\s*"payload"\s*:'
)

HistoryLoader = Callable[[Path, str], Awaitable["SessionHistory"]]


@dataclass(frozen=True, slots=True)
class HistoryBlock:
    """One transcript row reconstructed from persisted wire messages."""

    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """A conversation page ready to mount in the TUI.

    The extra position fields are optional so callers that provide the legacy
    two-argument history loader keep working.  A paged loader fills them in so
    the UI can describe the visible window without retaining the whole log.
    """

    blocks: list[HistoryBlock]
    omitted_turns: int = 0
    total_turns: int = 0
    start_turn: int = 0
    end_turn: int = 0
    has_older: bool = False


@dataclass(frozen=True, slots=True)
class WireHistoryIndex:
    """Byte offsets for the turn boundaries in one immutable wire-log snapshot."""

    path: Path
    turn_offsets: tuple[int, ...]
    file_size: int

    @property
    def total_turns(self) -> int:
        return len(self.turn_offsets)


@dataclass(slots=True)
class WireHistoryPager:
    """Read bounded conversation pages from an indexed ``wire.jsonl`` file."""

    index: WireHistoryIndex
    page_turns: int = HISTORY_PAGE_TURNS
    max_blocks: int = MAX_HISTORY_BLOCKS

    async def latest(self, *, page_turns: int | None = None) -> SessionHistory:
        """Return the newest page without scanning records outside that page."""

        return await load_wire_history_page(
            self.index,
            end_turn=self.index.total_turns,
            page_turns=page_turns or self.page_turns,
            max_blocks=self.max_blocks,
        )

    async def before(
        self,
        end_turn: int,
        *,
        page_turns: int | None = None,
    ) -> SessionHistory:
        """Return the page immediately preceding ``end_turn``."""

        return await load_wire_history_page(
            self.index,
            end_turn=end_turn,
            page_turns=page_turns or self.page_turns,
            max_blocks=self.max_blocks,
        )

    async def ending_at(
        self,
        end_turn: int,
        *,
        page_turns: int | None = None,
    ) -> SessionHistory:
        """Return a page ending at ``end_turn`` for explicit page navigation."""

        return await self.before(
            end_turn,
            page_turns=page_turns,
        )


@dataclass
class HistoryAccumulator:
    """Fold wire messages into a bounded last-N-turns window.

    Dialogue text is retained in full; auxiliary records use the display bound
    so a verbose tool event cannot dominate a history page.
    """

    max_turns: int = MAX_HISTORY_TURNS
    max_blocks: int = MAX_HISTORY_BLOCKS
    omitted_turns: int = 0
    _turns: deque[list[HistoryBlock]] = field(default_factory=deque)
    _current: list[HistoryBlock] = field(default_factory=list)
    _render_state: RenderState = field(default_factory=RenderState)
    _block_count: int = 0

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
        return SessionHistory(blocks=blocks, omitted_turns=self.omitted_turns)

    def _add(self, kind: str, text: str, *, merge: bool, replace: bool = False) -> None:
        if replace and self._current and self._current[-1].kind == kind:
            self._current[-1] = HistoryBlock(
                kind,
                text if is_dialogue_record(kind) else truncate_display(text),
            )
            return
        if merge and self._current and self._current[-1].kind == kind:
            previous = self._current[-1]
            self._current[-1] = HistoryBlock(
                kind,
                previous.text + text
                if is_dialogue_record(kind)
                else bounded_concat(previous.text, text),
            )
            return
        self._current.append(
            HistoryBlock(
                kind,
                text if is_dialogue_record(kind) else bounded_concat("", text),
            )
        )
        self._block_count += 1
        self._trim_blocks()

    def _trim_blocks(self) -> None:
        """Keep the accumulator bounded even before the current turn closes."""

        if self.max_blocks <= 0:
            return
        while self._block_count > self.max_blocks:
            if self._turns:
                self._block_count -= len(self._turns.popleft())
                continue
            if self._current:
                self._current.pop(0)
                self._block_count -= 1
                continue
            break

    def _flush_turn(self) -> None:
        if not self._current:
            return
        self._turns.append(self._current)
        self._current = []
        while self.max_turns > 0 and len(self._turns) > self.max_turns:
            self._block_count -= len(self._turns.popleft())
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


def _scan_wire_history_index(path: Path) -> WireHistoryIndex:
    """Build turn offsets in one sequential pass without retaining log lines."""

    offsets: list[int] = []
    file_size = 0
    try:
        with path.open("rb") as wire_file:
            for line in wire_file:
                if _TURN_BEGIN_RECORD.search(line):
                    offsets.append(file_size)
                file_size += len(line)
    except OSError:
        return WireHistoryIndex(path=path, turn_offsets=(), file_size=0)
    return WireHistoryIndex(
        path=path,
        turn_offsets=tuple(offsets),
        file_size=file_size,
    )


async def create_history_pager(
    work_dir: Path,
    session_id: str,
    *,
    page_turns: int = HISTORY_PAGE_TURNS,
    max_blocks: int = MAX_HISTORY_BLOCKS,
) -> WireHistoryPager | None:
    """Open and index a session's wire log for incremental history browsing."""

    from kimi_cli.session import Session as CliSession

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return None
    index = await asyncio.to_thread(_scan_wire_history_index, cli_session.wire_file.path)
    return WireHistoryPager(
        index=index,
        page_turns=max(1, page_turns),
        max_blocks=max(0, max_blocks),
    )


def _read_wire_history_range(
    path: Path,
    start_offset: int,
    end_offset: int,
    *,
    max_blocks: int,
) -> list[HistoryBlock]:
    """Parse one bounded byte range and discard raw records as they are folded."""

    from kimi_cli.wire.file import WireFileMetadata, WireMessageRecord
    from pydantic_core import from_json

    accumulator = HistoryAccumulator(max_turns=0, max_blocks=max_blocks)
    try:
        with path.open("rb") as wire_file:
            wire_file.seek(max(0, start_offset))
            position = max(0, start_offset)
            first_line = start_offset == 0
            while position < end_offset:
                raw_line = wire_file.readline()
                if not raw_line:
                    break
                position += len(raw_line)
                try:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    # Historical records contain mostly unique short identifiers.
                    # Disable Pydantic's process-wide string cache for this replay
                    # path, while retaining normal model validation.
                    data = from_json(raw_line, cache_strings=False)
                    if (
                        first_line
                        and isinstance(data, dict)
                        and data.get("type") == "metadata"
                    ):
                        parsed = WireFileMetadata.model_validate(data)
                    else:
                        parsed = WireMessageRecord.model_validate(data)
                    first_line = False
                    to_wire_message = getattr(parsed, "to_wire_message", None)
                    if callable(to_wire_message):
                        accumulator.feed(to_wire_message())
                except Exception:  # noqa: BLE001, S112 - skip malformed historical records
                    continue
    except OSError:
        return []
    return accumulator.finish().blocks


async def load_wire_history_page(
    index: WireHistoryIndex,
    *,
    end_turn: int | None = None,
    page_turns: int = HISTORY_PAGE_TURNS,
    max_blocks: int = MAX_HISTORY_BLOCKS,
) -> SessionHistory:
    """Load one turn page from ``index`` using a worker-thread file read."""

    total_turns = index.total_turns
    if total_turns == 0:
        blocks = await asyncio.to_thread(
            _read_wire_history_range,
            index.path,
            0,
            index.file_size,
            max_blocks=max(0, max_blocks),
        )
        return SessionHistory(blocks=blocks)

    end = total_turns if end_turn is None else min(total_turns, max(0, end_turn))
    if end == 0:
        return SessionHistory(
            blocks=[],
            total_turns=total_turns,
            start_turn=0,
            end_turn=0,
            has_older=False,
        )
    turns = max(1, page_turns)
    start = max(0, end - turns)
    start_offset = 0 if start == 0 else index.turn_offsets[start]
    end_offset = index.file_size if end == total_turns else index.turn_offsets[end]
    blocks = await asyncio.to_thread(
        _read_wire_history_range,
        index.path,
        start_offset,
        end_offset,
        max_blocks=max(0, max_blocks),
    )
    return SessionHistory(
        blocks=blocks,
        omitted_turns=start,
        total_turns=total_turns,
        start_turn=start,
        end_turn=end,
        has_older=start > 0,
    )


def _tail_wire_record_lines(path: Path, max_turns: int) -> tuple[list[bytes], int]:
    """Scan a wire log cheaply and retain raw lines for only its last turns."""

    turns: deque[list[bytes]] = deque()
    current: list[bytes] = []
    omitted_turns = 0
    with path.open("rb") as wire_file:
        for line in wire_file:
            if _TURN_BEGIN_RECORD.search(line):
                if current:
                    turns.append(current)
                    if len(turns) > max_turns:
                        turns.popleft()
                        omitted_turns += 1
                current = [line]
            elif current:
                current.append(line)
    if current:
        turns.append(current)
        if len(turns) > max_turns:
            turns.popleft()
            omitted_turns += 1
    return [line for turn in turns for line in turn], omitted_turns


async def _feed_wire_log(
    work_dir: Path,
    session_id: str,
    accumulator: HistoryAccumulator,
) -> None:
    from kimi_cli.session import Session as CliSession
    from kimi_cli.wire.file import parse_wire_file_line

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return
    if accumulator.max_turns > 0:
        try:
            lines, omitted_turns = await asyncio.to_thread(
                _tail_wire_record_lines,
                cli_session.wire_file.path,
                accumulator.max_turns,
            )
        except OSError:
            return
        accumulator.omitted_turns += omitted_turns
        for line in lines:
            try:
                record = parse_wire_file_line(line)
                to_wire_message = getattr(record, "to_wire_message", None)
                if callable(to_wire_message):
                    accumulator.feed(to_wire_message())
            except Exception:  # noqa: BLE001, S112 - skip unreadable historical records
                continue
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

    A bounded turn request scans raw JSONL in a worker thread and parses only
    the retained tail. ``max_turns`` / ``max_blocks`` of 0 keep the full compact
    history so the TUI can scroll back; wire objects are not retained.
    """

    accumulator = HistoryAccumulator(max_turns=max_turns, max_blocks=max_blocks)
    if messages is None:
        await _feed_wire_log(work_dir, session_id, accumulator)
    else:
        for message in messages:
            accumulator.feed(message)
    return accumulator.finish()

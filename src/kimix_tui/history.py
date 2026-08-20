"""Session timeline: wire.jsonl → TurnIndex → sliding hydrated window."""

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
    render_wire_message,
    user_input_text,
)
from kimix_tui.transcript_paint import default_expanded, is_compact_record, is_dialogue_record

# Legacy paging caps. Chat no longer windows history with these; they remain
# for the memory-diagnosis script and injected HistoryLoader tests.
MAX_HISTORY_TURNS = 4
MAX_HISTORY_BLOCKS = 32
HISTORY_PAGE_TURNS = MAX_HISTORY_TURNS
MAX_HISTORY_WINDOW_TURNS = 64

HYDRATED_BODY_BUDGET = 6 * 1024 * 1024
EAGER_FILE_SIZE = 8 * 1024 * 1024
INITIAL_HYDRATE_TURNS = 3
WINDOW_RADIUS = INITIAL_HYDRATE_TURNS
UNMATERIALIZED_TURN_LINES = 6
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


TurnIndex = WireHistoryIndex


@dataclass(slots=True)
class RecordStub:
    """One timeline row: cheap metadata plus an optional hydrated body."""

    kind: str
    turn: int
    summary: str
    estimated_lines: int
    body: str | None = None

    @property
    def hydrated(self) -> bool:
        return self.body is not None

    @property
    def text(self) -> str:
        return self.body or ""


def estimate_stub_lines(kind: str, text: str, *, expanded: bool | None = None, width: int = 80) -> int:
    """Conservative line count matching Transcript estimates."""

    if expanded is None:
        expanded = default_expanded(kind)
    if is_compact_record(kind, expanded=expanded):
        return 1
    wrap_width = max(8, width - 2)
    lines = text.split("\n") if text else [""]
    body_lines = 0
    for line in lines:
        cells = max(1, len(line))
        body_lines += max(1, (cells + wrap_width - 1) // wrap_width)
    return 1 + body_lines + 1


def stub_from_block(block: HistoryBlock, turn: int, *, hydrate: bool = True) -> RecordStub:
    """Fold one history block into a timeline stub."""

    body = block.text if hydrate else None
    return RecordStub(
        kind=block.kind,
        turn=turn,
        summary="",
        estimated_lines=estimate_stub_lines(block.kind, block.text),
        body=body,
    )


@dataclass(slots=True)
class Timeline:
    """Sliding window over an indexed wire log or in-memory turns."""

    index: WireHistoryIndex | None = None
    hydrated_budget: int = HYDRATED_BODY_BUDGET
    wrap_width: int = 80
    _turns: list[list[RecordStub] | None] = field(default_factory=list)
    _source_blocks: list[list[HistoryBlock]] | None = None

    @classmethod
    def from_turn_blocks(cls, turns: Sequence[Sequence[HistoryBlock]]) -> Timeline:
        """Build a fully materialized in-memory timeline (tests / fakes)."""

        timeline = cls()
        source = [list(turn) for turn in turns]
        timeline._source_blocks = source
        timeline._turns = [
            [stub_from_block(block, turn_index) for block in turn]
            for turn_index, turn in enumerate(source)
        ]
        return timeline

    @property
    def total_turns(self) -> int:
        if self.index is not None:
            return self.index.total_turns
        return len(self._turns)

    @property
    def materialized_turn_count(self) -> int:
        return sum(1 for turn in self._turns if turn is not None)

    def has_older_unmaterialized(self, first_materialized: int) -> bool:
        return first_materialized > 0

    def turn_range(self, turn: int) -> tuple[int, int]:
        """Byte range for a 0-based turn, or (0, 0) for in-memory timelines."""

        index = self.index
        if index is None or not index.turn_offsets:
            return 0, 0
        turn = max(0, min(turn, index.total_turns - 1))
        start = index.turn_offsets[turn]
        end = index.file_size if turn + 1 >= index.total_turns else index.turn_offsets[turn + 1]
        return start, end

    def stubs_for_turn(self, turn: int) -> list[RecordStub] | None:
        if turn < 0 or turn >= len(self._turns):
            return None
        return self._turns[turn]

    def iter_materialized_stubs(self) -> list[RecordStub]:
        stubs: list[RecordStub] = []
        for turn in self._turns:
            if turn is not None:
                stubs.extend(turn)
        return stubs

    def display_items(self) -> list[tuple[str, str, int]]:
        """Kind, full text, and 0-based turn for the current sliding window."""

        return [
            (stub.kind, stub.body, stub.turn)
            for stub in self.iter_materialized_stubs()
            if stub.body is not None
        ]

    def first_materialized_turn(self) -> int:
        for index, turn in enumerate(self._turns):
            if turn is not None:
                return index
        return 0

    def last_materialized_turn(self) -> int:
        last = -1
        for index, turn in enumerate(self._turns):
            if turn is not None:
                last = index
        return last

    def virtual_lines(self) -> int:
        """Scrollbar height: unmaterialized estimates + materialized stub heights."""

        total = 0
        expected = self.total_turns
        for turn_index in range(expected):
            turn = self._turns[turn_index] if turn_index < len(self._turns) else None
            if turn is None:
                total += UNMATERIALIZED_TURN_LINES
            else:
                total += sum(max(1, stub.estimated_lines) for stub in turn)
        return total

    def first_line_of_turn(self, turn: int) -> int:
        turn = max(0, min(turn, max(0, self.total_turns - 1)))
        line = 0
        for turn_index in range(turn):
            rows = self._turns[turn_index] if turn_index < len(self._turns) else None
            if rows is None:
                line += UNMATERIALIZED_TURN_LINES
            else:
                line += sum(max(1, stub.estimated_lines) for stub in rows)
        return line

    def turn_at_line(self, line: int) -> int:
        if self.total_turns <= 0:
            return 0
        remaining = max(0, line)
        for turn_index in range(self.total_turns):
            rows = self._turns[turn_index] if turn_index < len(self._turns) else None
            height = (
                UNMATERIALIZED_TURN_LINES
                if rows is None
                else sum(max(1, stub.estimated_lines) for stub in rows)
            )
            if remaining < height:
                return turn_index
            remaining -= height
        return self.total_turns - 1

    def hydrated_chars(self) -> int:
        total = 0
        for turn in self._turns:
            if turn is None:
                continue
            for stub in turn:
                if stub.body is not None:
                    total += len(stub.body)
        return total

    async def open(self) -> None:
        """Materialize enough history for an IM-style latest view."""

        total = self.total_turns
        if total <= 0:
            if self.index is not None and self.index.file_size > 0:
                await self._materialize_byte_range(0, self.index.file_size, start_turn=0)
            else:
                self._turns = []
            return
        self._ensure_turn_slots(total)
        eager = self.index is None or self.index.file_size <= EAGER_FILE_SIZE
        if eager:
            await self.materialize_turns(0, total, hydrate=True)
            return
        start = max(0, total - INITIAL_HYDRATE_TURNS)
        await self.materialize_turns(start, total, hydrate=True)

    async def materialize_turns(
        self,
        start: int,
        end: int,
        *,
        hydrate: bool = True,
    ) -> int:
        """Parse ``[start, end)`` into stubs. Return newly materialized turn count."""

        total = self.total_turns
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        self._ensure_turn_slots(total)
        missing = [turn for turn in range(start, end) if self._turns[turn] is None]
        if not missing:
            if hydrate:
                self.hydrate_turns(start, end)
            return 0
        added = 0
        if self._source_blocks is not None:
            for turn in missing:
                blocks = self._source_blocks[turn]
                self._turns[turn] = [
                    stub_from_block(block, turn, hydrate=hydrate) for block in blocks
                ]
                added += 1
            return added
        if self.index is None:
            return 0
        run_start = missing[0]
        run_end = missing[-1] + 1
        start_offset, _ = self.turn_range(run_start)
        _, end_offset = self.turn_range(run_end - 1)
        await self._materialize_byte_range(
            start_offset,
            end_offset,
            start_turn=run_start,
            hydrate=hydrate,
        )
        return sum(1 for turn in missing if self._turns[turn] is not None)

    def window_bounds(self, keep_turn: int, radius: int = WINDOW_RADIUS) -> tuple[int, int]:
        """Inclusive-start, exclusive-end turn range around ``keep_turn``."""

        total = self.total_turns
        if total <= 0:
            return 0, 0
        keep = max(0, min(keep_turn, total - 1))
        start = max(0, keep - radius)
        end = min(total, keep + radius + 1)
        return start, end

    def drop_outside_window(self, *, keep_turn: int, radius: int = WINDOW_RADIUS) -> None:
        """Drop turns outside the window. Does not leave summary stubs."""

        if self.total_turns <= 0:
            return
        keep_start, keep_end = self.window_bounds(keep_turn, radius)
        for turn_index, rows in enumerate(self._turns):
            if rows is not None and not keep_start <= turn_index < keep_end:
                self._turns[turn_index] = None

    async def slide_to(self, turn: int, *, radius: int = WINDOW_RADIUS) -> None:
        """Keep only ``turn ± radius``, fully hydrated. Never parse the gap."""

        total = self.total_turns
        if total <= 0:
            return
        turn = max(0, min(turn, total - 1))
        self.drop_outside_window(keep_turn=turn, radius=radius)
        start, end = self.window_bounds(turn, radius)
        await self.materialize_turns(start, end, hydrate=True)

    async def ensure_turn(self, turn: int, *, radius: int = WINDOW_RADIUS) -> None:
        """Slide the hydrated window so ``turn`` is inside it."""

        await self.slide_to(turn, radius=radius)

    def hydrate_turns(self, start: int, end: int) -> None:
        """Restore bodies for already materialized turns from in-memory sources."""

        if self._source_blocks is None:
            return
        for turn_index in range(max(0, start), min(end, len(self._turns))):
            rows = self._turns[turn_index]
            source = self._source_blocks[turn_index]
            if rows is None:
                continue
            for stub, block in zip(rows, source, strict=False):
                if stub.body is None:
                    stub.body = block.text

    async def rehydrate_turns(self, start: int, end: int) -> None:
        """Restore bodies for ``[start, end)`` from memory or the wire log."""

        start = max(0, start)
        end = min(end, self.total_turns)
        if self._source_blocks is not None:
            self.hydrate_turns(start, end)
            return
        if self.index is None:
            return
        for turn_index in range(start, end):
            rows = self._turns[turn_index]
            if rows is None or all(stub.body is not None for stub in rows):
                continue
            start_offset, end_offset = self.turn_range(turn_index)
            blocks = await asyncio.to_thread(
                _read_wire_history_range,
                self.index.path,
                start_offset,
                end_offset,
                max_blocks=0,
            )
            for stub, block in zip(rows, blocks, strict=False):
                stub.body = block.text

    def unload_distant(self, *, keep_turn: int, radius: int = WINDOW_RADIUS) -> None:
        """Drop turns outside ``keep_turn ± radius`` (alias of drop_outside_window)."""

        self.drop_outside_window(keep_turn=keep_turn, radius=radius)


    def _ensure_turn_slots(self, total: int) -> None:
        if len(self._turns) < total:
            self._turns.extend([None] * (total - len(self._turns)))

    async def _materialize_byte_range(
        self,
        start_offset: int,
        end_offset: int,
        *,
        start_turn: int,
        hydrate: bool = True,
    ) -> None:
        if self.index is None:
            return
        turns = await asyncio.to_thread(
            _read_wire_turns,
            self.index.path,
            start_offset,
            end_offset,
        )
        self._ensure_turn_slots(max(self.total_turns, start_turn + len(turns)))
        for offset, blocks in enumerate(turns):
            turn_index = start_turn + offset
            if turn_index >= len(self._turns):
                break
            self._turns[turn_index] = [
                stub_from_block(block, turn_index, hydrate=hydrate) for block in blocks
            ]


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
    """Fold wire messages into transcript blocks.

    Dialogue text is retained in full; auxiliary records use the display bound
    so a verbose tool event cannot dominate a row. Turn/block caps are optional
    and unused by the continuous timeline.
    """

    max_turns: int = 0
    max_blocks: int = 0
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

    def finish_turns(self) -> list[list[HistoryBlock]]:
        self._flush_turn()
        return [list(turn) for turn in self._turns]

    def _add(self, kind: str, text: str, *, merge: bool, replace: bool = False) -> None:
        if replace and self._current and self._current[-1].kind == kind:
            self._current[-1] = HistoryBlock(kind, text)
            return
        if merge and self._current and self._current[-1].kind == kind:
            previous = self._current[-1]
            self._current[-1] = HistoryBlock(kind, previous.text + text)
            return
        self._current.append(HistoryBlock(kind, text))
        self._block_count += 1
        self._trim_blocks()

    def _trim_blocks(self) -> None:
        """Drop oldest auxiliary records; never discard user/assistant dialogue."""

        if self.max_blocks <= 0:
            return
        while self._block_count > self.max_blocks:
            if not self._drop_oldest_auxiliary_block():
                return

    def _drop_oldest_auxiliary_block(self) -> bool:
        """Remove the oldest non-dialogue block. Return False if only dialogue remains."""

        for turn_index, turn in enumerate(self._turns):
            for block_index, block in enumerate(turn):
                if is_dialogue_record(block.kind):
                    continue
                del turn[block_index]
                self._block_count -= 1
                if not turn:
                    del self._turns[turn_index]
                return True
        for block_index, block in enumerate(self._current):
            if is_dialogue_record(block.kind):
                continue
            del self._current[block_index]
            self._block_count -= 1
            return True
        return False

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


async def create_timeline(work_dir: Path, session_id: str) -> Timeline | None:
    """Open and index a session's wire log as a continuous timeline."""

    from kimi_cli.session import Session as CliSession

    kaos_dir = KaosPath.unsafe_from_local_path(Path(work_dir).resolve()).canonical()
    cli_session = await CliSession.find(kaos_dir, session_id)
    if cli_session is None:
        return None
    index = await asyncio.to_thread(_scan_wire_history_index, cli_session.wire_file.path)
    timeline = Timeline(index=index)
    await timeline.open()
    return timeline


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

    turns = _read_wire_turns(path, start_offset, end_offset, max_blocks=max_blocks)
    return [block for turn in turns for block in turn]


def _read_wire_turns(
    path: Path,
    start_offset: int,
    end_offset: int,
    *,
    max_blocks: int = 0,
) -> list[list[HistoryBlock]]:
    """Parse one bounded byte range into per-turn block lists."""

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
    return accumulator.finish_turns()


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

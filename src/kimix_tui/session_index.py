"""List historical Kimix sessions from the kimi-cli work-dir store."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import time
from typing import Any

from kaos.path import KaosPath

from kimix_tui.kimi_workdir import resolve_kimi_work_dir

_LAST_ID_UNSET: object = object()
_PREVIEW_MAX = 80
_UNTITLED = "Untitled"

SessionLister = Callable[[KaosPath], Awaitable[Sequence[Any]]]
SessionLoader = Callable[[Path], Awaitable[list["SessionSummary"]]]
SessionDeleter = Callable[[Path, Sequence[str]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Display fields for one resumable session on the TUI home screen."""

    id: str
    title: str
    preview: str
    updated_at: float
    is_last: bool = False
    size_bytes: int = 0
    file_count: int = 0
    storage_format: str = "Unknown"
    is_archived: bool = False
    todo_count: int = 0
    additional_dir_count: int = 0


def _preview_from_title(title: str) -> str:
    text = " ".join(title.split()) if title else _UNTITLED
    if not text:
        text = _UNTITLED
    if len(text) <= _PREVIEW_MAX:
        return text
    return text[: _PREVIEW_MAX - 1].rstrip() + "…"


def format_relative_time(updated_at: float, *, now: float | None = None) -> str:
    """Return a short relative timestamp for home-screen session details."""

    if updated_at <= 0:
        return "unknown"
    current = now if now is not None else time()
    delta = max(0, int(current - updated_at))
    if delta < 45:
        return "just now"
    if delta < 90:
        return "1 minute ago"
    if delta < 3600:
        return f"{delta // 60} minutes ago"
    if delta < 5400:
        return "1 hour ago"
    if delta < 86400:
        return f"{delta // 3600} hours ago"
    if delta < 172800:
        return "yesterday"
    local_tz = datetime.now(UTC).astimezone().tzinfo
    moment = datetime.fromtimestamp(updated_at, tz=local_tz)
    current_dt = datetime.fromtimestamp(current, tz=local_tz)
    if current_dt.year == moment.year:
        return moment.strftime("%b %d")
    return moment.strftime("%Y-%m-%d")


def format_file_size(size_bytes: int) -> str:
    """Format a byte count in binary KB or MB units."""

    size = max(0, size_bytes)
    if size == 0:
        return "0 KB"
    if size < 1024 * 1024:
        value = max(0.1, size / 1024)
        return f"{value:.1f}".rstrip("0").rstrip(".") + " KB"
    value = size / (1024 * 1024)
    return f"{value:.1f}".rstrip("0").rstrip(".") + " MB"


def _session_storage_stats(session: Any) -> tuple[int, int, str]:
    try:
        session_dir = Path(session.dir)
    except AttributeError, TypeError, OSError:
        session_dir = None

    size_bytes = 0
    file_count = 0
    if session_dir is not None:
        for root, _dirs, files in session_dir.walk(on_error=lambda _error: None):
            for name in files:
                try:
                    size_bytes += (root / name).stat().st_size
                    file_count += 1
                except OSError:
                    continue

    try:
        suffix = Path(session.context_file).suffix.lower()
    except AttributeError, TypeError:
        suffix = ""
    storage_format = {".db": "SQLite", ".jsonl": "JSONL"}.get(suffix, "Unknown")
    return size_bytes, file_count, storage_format


def _collection_size(value: object) -> int:
    return len(value) if isinstance(value, Sized) else 0


def summaries_from_sessions(
    sessions: Sequence[Any],
    *,
    last_session_id: str | None = None,
) -> list[SessionSummary]:
    """Map kimi-cli session objects (or test doubles) into home-screen rows."""

    summaries: list[SessionSummary] = []
    for session in sessions:
        session_id = str(getattr(session, "id", "") or "")
        if not session_id:
            continue
        title = str(getattr(session, "title", "") or _UNTITLED)
        updated_at = float(getattr(session, "updated_at", 0.0) or 0.0)
        size_bytes, file_count, storage_format = _session_storage_stats(session)
        state = getattr(session, "state", None)
        summaries.append(
            SessionSummary(
                id=session_id,
                title=title,
                preview=_preview_from_title(title),
                updated_at=updated_at,
                is_last=session_id == last_session_id,
                size_bytes=size_bytes,
                file_count=file_count,
                storage_format=storage_format,
                is_archived=bool(getattr(state, "archived", False)),
                todo_count=_collection_size(getattr(state, "todos", ())),
                additional_dir_count=_collection_size(getattr(state, "additional_dirs", ())),
            )
        )
    return sorted(summaries, key=lambda summary: summary.updated_at, reverse=True)


async def _list_cli_sessions(work_dir: KaosPath) -> Sequence[Any]:
    from kimi_cli.session import Session as CliSession

    return await CliSession.list(work_dir)


def _last_session_id(work_dir: KaosPath) -> str | None:
    from kimi_cli.metadata import load_metadata

    meta = load_metadata().get_work_dir_meta(work_dir)
    if meta is None:
        return None
    return meta.last_session_id


async def list_session_summaries(
    work_dir: Path,
    *,
    list_sessions: SessionLister | None = None,
    last_session_id: str | None | object = _LAST_ID_UNSET,
) -> list[SessionSummary]:
    """List non-empty sessions for ``work_dir`` using kimi-cli storage rules."""

    kaos_dir = resolve_kimi_work_dir(work_dir)
    sessions = await (list_sessions or _list_cli_sessions)(kaos_dir)
    resolved_last = (
        _last_session_id(kaos_dir) if last_session_id is _LAST_ID_UNSET else last_session_id
    )
    last_id = resolved_last if isinstance(resolved_last, str) else None
    return summaries_from_sessions(sessions, last_session_id=last_id)


async def delete_sessions(work_dir: Path, session_ids: Sequence[str]) -> None:
    """Permanently delete Kimi sessions and clear a deleted last-session pointer."""

    from kimi_cli.metadata import load_metadata, save_metadata
    from kimi_cli.session import Session as CliSession

    ids = list(dict.fromkeys(session_id for session_id in session_ids if session_id))
    if not ids:
        return
    kaos_dir = resolve_kimi_work_dir(work_dir)
    for session_id in ids:
        session = await CliSession.find(kaos_dir, session_id)
        if session is not None:
            await session.delete()

    metadata = load_metadata()
    work_dir_meta = metadata.get_work_dir_meta(kaos_dir)
    if work_dir_meta is not None and work_dir_meta.last_session_id in ids:
        work_dir_meta.last_session_id = None
        save_metadata(metadata)

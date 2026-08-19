from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kimix_tui.session_index import (
    delete_sessions,
    format_file_size,
    format_relative_time,
    list_session_summaries,
    summaries_from_sessions,
)


def test_summaries_mark_last_session_and_skip_blank_ids() -> None:
    sessions = [
        SimpleNamespace(id="sess-a", title="Fix login redirect", updated_at=100.0),
        SimpleNamespace(id="", title="ignored", updated_at=90.0),
        SimpleNamespace(id="sess-b", title="", updated_at=50.0),
    ]

    summaries = summaries_from_sessions(sessions, last_session_id="sess-a")

    assert [item.id for item in summaries] == ["sess-a", "sess-b"]
    assert summaries[0].is_last is True
    assert summaries[0].preview == "Fix login redirect"
    assert summaries[1].title == "Untitled"
    assert summaries[1].preview == "Untitled"
    assert summaries[1].is_last is False


def test_preview_truncates_long_titles() -> None:
    title = "A" * 100
    summaries = summaries_from_sessions([SimpleNamespace(id="long", title=title, updated_at=1.0)])

    assert summaries[0].title == title
    assert summaries[0].preview.endswith("…")
    assert len(summaries[0].preview) == 80


def test_format_relative_time_buckets() -> None:
    now = 1_700_000_000.0
    assert format_relative_time(now - 10, now=now) == "just now"
    assert format_relative_time(now - 120, now=now) == "2 minutes ago"
    assert format_relative_time(now - 8000, now=now) == "2 hours ago"
    assert format_relative_time(now - 100_000, now=now) == "yesterday"
    assert format_relative_time(0, now=now) == "unknown"


def test_format_file_size_uses_kb_and_mb_tiers() -> None:
    assert format_file_size(0) == "0 KB"
    assert format_file_size(512) == "0.5 KB"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(1024 * 1024) == "1 MB"
    assert format_file_size(int(5.5 * 1024 * 1024)) == "5.5 MB"


def test_summary_collects_session_directory_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "stored-session"
    nested_dir = session_dir / "subagents" / "worker"
    nested_dir.mkdir(parents=True)
    context_file = session_dir / "context.db"
    context_file.write_bytes(b"x" * 1536)
    (nested_dir / "wire.jsonl").write_bytes(b"y" * 1024)
    session = SimpleNamespace(
        id="stored-session",
        title="Stored session",
        updated_at=100.0,
        dir=session_dir,
        context_file=context_file,
        state=SimpleNamespace(
            archived=True,
            todos=[object(), object()],
            additional_dirs=["D:/shared"],
        ),
    )

    summary = summaries_from_sessions([session])[0]

    assert summary.size_bytes == 2560
    assert summary.file_count == 2
    assert summary.storage_format == "SQLite"
    assert summary.is_archived is True
    assert summary.todo_count == 2
    assert summary.additional_dir_count == 1


async def test_list_session_summaries_uses_injected_lister(tmp_path: Path) -> None:
    sessions = [
        SimpleNamespace(id="older", title="Oldest", updated_at=10.0),
        SimpleNamespace(id="newer", title="Newest", updated_at=20.0),
    ]

    async def fake_list(_work_dir: object) -> list[SimpleNamespace]:
        return sessions

    summaries = await list_session_summaries(
        tmp_path,
        list_sessions=fake_list,
        last_session_id="older",
    )

    assert [item.id for item in summaries] == ["newer", "older"]
    assert summaries[1].is_last is True


async def test_list_session_summaries_empty(tmp_path: Path) -> None:
    async def fake_list(_work_dir: object) -> list[SimpleNamespace]:
        return []

    summaries = await list_session_summaries(
        tmp_path,
        list_sessions=fake_list,
        last_session_id=None,
    )

    assert summaries == []


async def test_delete_sessions_uses_kimi_storage_and_clears_last_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))
    from kimi_cli.metadata import load_metadata, save_metadata
    from kimi_cli.session import Session as CliSession

    from kimix_tui.kimi_workdir import resolve_kimi_work_dir

    work_dir = tmp_path / "project"
    resolved = resolve_kimi_work_dir(work_dir)
    metadata = load_metadata()
    work_dir_meta = metadata.new_work_dir_meta(resolved)
    work_dir_meta.last_session_id = "session-1"
    save_metadata(metadata)
    deleted: list[str] = []

    class FakeSession:
        async def delete(self) -> None:
            deleted.append("session-1")

    async def find(_work_dir: object, session_id: str) -> FakeSession | None:
        return FakeSession() if session_id == "session-1" else None

    monkeypatch.setattr(CliSession, "find", staticmethod(find))

    await delete_sessions(work_dir, ["session-1", "missing", "session-1"])

    assert deleted == ["session-1"]
    saved_meta = load_metadata().get_work_dir_meta(resolved)
    assert saved_meta is not None
    assert saved_meta.last_session_id is None

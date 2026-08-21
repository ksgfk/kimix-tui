from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from qtutil import find, launch_app, wait_chat_ready, wait_home, wait_idle, widget_text

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.request_dialogs import DeleteSessionsDialog
from kimix_tui.session_index import SessionSummary


def _config_store(tmp_path: Path) -> LLMConfigStore:
    config_file = tmp_path / "test-provider.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "name": "Test Model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    store = LLMConfigStore(
        tmp_path / "kimix-tui.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-tui.json"
        ),
    )
    store.set_default(tmp_path, inspect_llm_config(config_file))
    return store


class FakeSession:
    def __init__(self, session_id: str = "fake-session") -> None:
        self.id = session_id
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self.prompts: list[str] = []
        self.closed = False

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        self.prompts.append(user_input)
        assert merge_wire_messages is False
        if False:  # pragma: no cover
            yield None

    def cancel(self) -> None:
        return None

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _summaries() -> list[SessionSummary]:
    return [
        SessionSummary(
            id="sess-1",
            title="Fix login",
            preview="Fix login",
            updated_at=1_700_000_000.0,
            is_last=True,
            size_bytes=1_572_864,
            file_count=4,
            storage_format="SQLite",
            todo_count=2,
            additional_dir_count=1,
        ),
        SessionSummary(
            id="sess-2",
            title="Untitled",
            preview="Untitled",
            updated_at=1_699_996_400.0,
            size_bytes=42 * 1024,
            file_count=2,
            storage_format="JSONL",
            is_archived=True,
        ),
    ]


async def _history_loader(_work_dir: Path) -> list[SessionSummary]:
    return list(reversed(_summaries()))


async def _empty_loader(_work_dir: Path) -> list[SessionSummary]:
    return []


def test_missing_session_id_opens_home(qtbot, tmp_path: Path) -> None:
    session = FakeSession()

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert [row.summary.id for row in home.session_rows()] == ["sess-1", "sess-2"]
    assert home.summary == _summaries()[0]
    assert widget_text(home, "detail-size") == "1.5 MB"
    assert widget_text(home, "detail-storage") == "SQLite · 4 files"
    assert widget_text(home, "detail-todos") == "2"
    assert widget_text(home, "detail-directories") == "1"


def test_new_session_shortcut_skips_resume(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession()

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_N)
    chat = wait_chat_ready(qtbot, app)
    assert opened[0].session_id is None
    assert chat.prompt_enabled is True


def test_new_session_button_starts_chat(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession()

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_empty_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "start-new-session"), Qt.MouseButton.LeftButton)
    wait_chat_ready(qtbot, app)
    assert opened[0].session_id is None
    assert isinstance(app.screen, ChatView)


def test_click_previews_session_before_opening(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession("sess-2")

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    qtbot.mouseClick(rows[1], Qt.MouseButton.LeftButton, pos=QPoint(90, 14))

    assert app.screen is home
    assert opened == []
    assert home.summary is not None
    assert home.summary.id == "sess-2"
    assert widget_text(home, "detail-state") == "Archived session"
    assert widget_text(home, "detail-size") == "42 KB"

    qtbot.mouseClick(find(home, "open-session"), Qt.MouseButton.LeftButton)
    wait_chat_ready(qtbot, app)
    assert opened[0].session_id == "sess-2"
    assert isinstance(app.screen, ChatView)


def test_click_session_check_toggles_selection(qtbot, tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    rows = home.session_rows()
    first = rows[0].summary
    qtbot.mouseClick(rows[1], Qt.MouseButton.LeftButton, pos=QPoint(8, 8))

    assert rows[1].selected is True
    assert find(rows[1], "session-check", QLabel).text() == "[x]"
    assert widget_text(home, "selection-count") == "1 selected"
    assert home.summary == first

    qtbot.mouseClick(rows[1], Qt.MouseButton.LeftButton, pos=QPoint(8, 8))
    assert rows[1].selected is False
    assert find(rows[1], "session-check", QLabel).text() == "[ ]"
    assert widget_text(home, "selection-count") == "0 selected"


def test_home_filters_sessions_by_title(qtbot, tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    search = find(home, "session-search", QLineEdit)
    search.setText("LOGIN")
    assert [row.summary.id for row in home.session_rows()] == ["sess-1"]
    assert widget_text(home, "session-count") == "1 of 2"

    search.setText("missing title")
    assert home.session_rows() == []
    assert "No sessions match" in widget_text(home, "home-status")


def test_home_selects_and_deletes_sessions_in_batch(qtbot, tmp_path: Path) -> None:
    deleted: list[str] = []

    async def deleter(_work_dir: Path, session_ids: Sequence[str]) -> None:
        deleted.extend(session_ids)

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        session_deleter=deleter,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    find(home, "select-shown", QPushButton).click()
    assert widget_text(home, "selection-count") == "2 selected"
    assert find(home, "delete-sessions", QPushButton).isEnabled()
    assert all(row.selected for row in home.session_rows())

    find(home, "delete-sessions", QPushButton).click()
    qtbot.waitUntil(lambda: isinstance(app.screen, DeleteSessionsDialog), timeout=10_000)
    find(app.screen, "confirm-delete", QPushButton).click()
    qtbot.waitUntil(lambda: home.session_rows() == [], timeout=10_000)
    wait_idle(qtbot, app)

    assert deleted == ["sess-1", "sess-2"]
    assert app.screen is home
    assert home.session_rows() == []
    assert widget_text(home, "selection-count") == "0 selected"


def test_enter_resumes_highlighted_session(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    session = FakeSession("sess-1")

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    chat = wait_chat_ready(qtbot, app)
    assert opened[0].session_id == "sess-1"
    assert any(record.text == "Session: sess-1" for record in chat.transcript.records)
    assert find(chat, "leave-session", QPushButton).text() == "Home"


def test_escape_from_chat_returns_home_and_releases_session(qtbot, tmp_path: Path) -> None:
    opened: list[SessionOptions] = []
    sessions: list[FakeSession] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        session = FakeSession(options.session_id or "created")
        sessions.append(session)
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    wait_home(qtbot, app)
    qtbot.keyClick(app.screen, Qt.Key.Key_Return)
    chat = wait_chat_ready(qtbot, app)
    assert chat.session_id == sessions[0].id
    assert any(record.text == "Session: sess-1" for record in chat.transcript.records)

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    home = wait_home(qtbot, app)
    assert sessions[0].closed is True
    assert window.chat is None
    assert chat.busy is False
    assert app._options.session_id is None

    qtbot.keyClick(home, Qt.Key.Key_N)
    new_chat = wait_chat_ready(qtbot, app)
    assert opened[1].session_id is None
    assert new_chat.session_id == sessions[1].id
    assert sessions[1].closed is False
    assert all(record.text != "Session: sess-1" for record in new_chat.transcript.records)


def test_quit_command_returns_home(qtbot, tmp_path: Path) -> None:
    session = FakeSession("fake-session")

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText("/quit")
    qtbot.keyClick(prompt, Qt.Key.Key_Return)
    wait_home(qtbot, app)
    assert session.closed is True
    assert app.window is not None
    assert app.window.chat is None


def test_home_uses_master_detail_layout_on_wide_window(qtbot, tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app, size=(1100, 700))
    home = wait_home(qtbot, app)
    browser = find(home, "session-browser")
    details = find(home, "session-detail")
    assert details.x() > browser.x()
    assert abs(details.y() - browser.y()) < 8
    assert home._splitter.orientation() == Qt.Orientation.Horizontal


def test_home_stacks_sections_on_narrow_window(qtbot, tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app, size=(700, 700))
    home = wait_home(qtbot, app)
    window.setMinimumSize(400, 400)
    window.resize(700, 700)
    QApplication.processEvents()
    home._sync_narrow(window.width())
    QApplication.processEvents()
    browser = find(home, "session-browser")
    details = find(home, "session-detail")
    assert home._splitter.orientation() == Qt.Orientation.Vertical
    assert details.y() > browser.y()
    assert abs(details.x() - browser.x()) < 8
    assert find(home, "open-session").geometry().bottom() <= details.geometry().bottom()
    assert find(home, "toggle-session").geometry().right() <= details.geometry().right()
    assert find(home, "delete-sessions").geometry().right() <= browser.geometry().right()
    assert find(home, "session-list").height() > 0

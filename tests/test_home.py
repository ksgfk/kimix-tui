from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Footer, Input, Static

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.screens.chat import ChatScreen
from kimix_tui.screens.home import (
    DeleteSessionsScreen,
    HomeScreen,
    SessionCheck,
    SessionDetails,
    SessionListItem,
)
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
        if False:  # pragma: no cover - keep this an async generator
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


def _shown_binding_labels(app: KimixTuiApp) -> set[str]:
    return {
        active.binding.description
        for active in app.screen.active_bindings.values()
        if active.binding.show and active.binding.description
    }


@pytest.mark.asyncio
async def test_missing_session_id_opens_home(tmp_path: Path) -> None:
    session = FakeSession()

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert list(app.screen.query(Footer)) == []
        assert [item.summary.id for item in app.screen.query(SessionListItem)] == [
            "sess-1",
            "sess-2",
        ]
        assert app.screen.query_one(SessionDetails).summary == _summaries()[0]
        assert str(app.screen.query_one("#detail-size", Static).content) == "1.5 MB"
        assert str(app.screen.query_one("#detail-storage", Static).content) == "SQLite · 4 files"
        assert str(app.screen.query_one("#detail-todos", Static).content) == "2"
        assert str(app.screen.query_one("#detail-directories", Static).content) == "1"
        labels = _shown_binding_labels(app)
        assert "Cancel" not in labels
        assert "Prompt" not in labels
        assert "Quit" not in labels
        assert "Sessions" not in labels


@pytest.mark.asyncio
async def test_new_session_shortcut_skips_resume(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("n")
        await app.workers.wait_for_complete()

        assert opened[0].session_id is None
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        prompt = chat.query_one("#prompt", Input)
        assert prompt.disabled is False

    assert session.closed is True


@pytest.mark.asyncio
async def test_new_session_button_starts_chat(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#start-new-session")
        await app.workers.wait_for_complete()

        assert opened[0].session_id is None
        assert isinstance(app.screen, ChatScreen)


@pytest.mark.asyncio
async def test_click_previews_session_before_opening(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        items = list(home.query(SessionListItem))

        await pilot.click(items[1])
        await pilot.pause()

        details = home.query_one(SessionDetails)
        assert app.screen is home
        assert opened == []
        assert details.summary is not None
        assert details.summary.id == "sess-2"
        assert str(home.query_one("#detail-state", Static).content) == "Archived session"
        assert str(home.query_one("#detail-size", Static).content) == "42 KB"

        await pilot.click("#open-session")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert opened[0].session_id == "sess-2"
        assert isinstance(app.screen, ChatScreen)


@pytest.mark.asyncio
async def test_click_session_check_toggles_selection(tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        items = list(home.query(SessionListItem))
        second_check = items[1].query_one(SessionCheck)

        await pilot.click(second_check)
        await pilot.pause()

        assert items[1].selected is True
        assert str(second_check.content) == "[x]"
        assert str(home.query_one("#selection-count", Static).content) == "1 selected"
        assert home.query_one(SessionDetails).summary == items[0].summary

        await pilot.click(second_check)
        await pilot.pause()

        assert items[1].selected is False
        assert str(second_check.content) == "[ ]"
        assert str(home.query_one("#selection-count", Static).content) == "0 selected"


@pytest.mark.asyncio
async def test_home_filters_sessions_by_title(tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )

    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        search = home.query_one("#session-search", Input)
        search.value = "LOGIN"
        await pilot.pause()

        assert [item.summary.id for item in home.query(SessionListItem)] == ["sess-1"]
        assert str(home.query_one("#session-count", Static).content) == "1 of 2"

        search.value = "missing title"
        await pilot.pause()
        assert list(home.query(SessionListItem)) == []
        assert "No sessions match" in str(home.query_one("#home-status", Static).content)


@pytest.mark.asyncio
async def test_home_selects_and_deletes_sessions_in_batch(tmp_path: Path) -> None:
    deleted: list[str] = []

    async def deleter(_work_dir: Path, session_ids: Sequence[str]) -> None:
        deleted.extend(session_ids)

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        session_deleter=deleter,
        config_store=_config_store(tmp_path),
    )

    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        await pilot.click("#select-shown")
        await pilot.pause()

        assert str(home.query_one("#selection-count", Static).content) == "2 selected"
        assert home.query_one("#delete-sessions", Button).disabled is False
        assert all(item.selected for item in home.query(SessionListItem))

        await pilot.click("#delete-sessions")
        await pilot.pause()
        assert isinstance(app.screen, DeleteSessionsScreen)
        await pilot.click("#confirm-delete")
        await app.workers.wait_for_complete()

        assert deleted == ["sess-1", "sess-2"]
        assert app.screen is home
        assert list(home.query(SessionListItem)) == []
        assert str(home.query_one("#selection-count", Static).content) == "0 selected"


@pytest.mark.asyncio
async def test_enter_resumes_highlighted_session(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert opened[0].session_id == "sess-1"
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        assert any(record.text == "Session: sess-1" for record in chat.transcript.records)
        labels = _shown_binding_labels(app)
        assert "Home" in labels
        assert "Quit" not in labels


@pytest.mark.asyncio
async def test_escape_from_chat_returns_home_and_releases_session(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        chat = app.screen
        assert isinstance(chat, ChatScreen)
        assert chat.session is sessions[0]
        assert any(record.text == "Session: sess-1" for record in chat.transcript.records)

        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(app.screen, HomeScreen)
        labels = _shown_binding_labels(app)
        assert "Cancel" not in labels
        assert "Prompt" not in labels
        assert "Quit" not in labels
        assert sessions[0].closed is True
        assert chat.session is None
        assert chat.busy is False
        assert all(not isinstance(screen, ChatScreen) for screen in app.screen_stack)
        assert app._options.session_id is None

        await pilot.press("n")
        await app.workers.wait_for_complete()

        assert opened[1].session_id is None
        new_chat = app.screen
        assert isinstance(new_chat, ChatScreen)
        assert new_chat.session is sessions[1]
        assert sessions[1].closed is False
        assert all(record.text != "Session: sess-1" for record in new_chat.transcript.records)

    assert all(session.closed for session in sessions)


@pytest.mark.asyncio
async def test_quit_command_returns_home(tmp_path: Path) -> None:
    session = FakeSession("fake-session")

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        prompt = chat.query_one("#prompt", Input)
        prompt.value = "/quit"
        prompt.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(app.screen, HomeScreen)
        assert session.closed is True
        assert chat.session is None


@pytest.mark.asyncio
async def test_home_uses_master_detail_layout_on_wide_terminal(tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        browser = home.query_one("#session-browser")
        details = home.query_one(SessionDetails)

        assert details.region.x > browser.region.x
        assert details.region.y == browser.region.y
        assert all(item.region.height == 2 for item in home.query(SessionListItem))


@pytest.mark.asyncio
async def test_home_stacks_sections_on_narrow_terminal(tmp_path: Path) -> None:
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=_history_loader,
        config_store=_config_store(tmp_path),
    )

    async with app.run_test(size=(70, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        browser = home.query_one("#session-browser")
        details = home.query_one(SessionDetails)

        assert home.has_class("-narrow")
        assert details.region.y > browser.region.y
        assert details.region.x == browser.region.x
        assert home.query_one("#open-session", Button).region.bottom <= details.region.bottom
        assert home.query_one("#toggle-session", Button).region.right <= details.region.right
        assert home.query_one("#delete-sessions", Button).region.right <= browser.region.right
        assert home.query_one("#session-list").region.height > 0

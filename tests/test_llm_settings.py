from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_llm_config import write_json_config, write_llm_config
from textual.widgets import Button, Input, Static

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.llm_config import (
    LLMConfigStore,
    inspect_llm_config,
    unavailable_config_reference,
)
from kimix_tui.screens.chat import ChatScreen
from kimix_tui.screens.home import HomeScreen
from kimix_tui.screens.settings import ConfigPathItem, LLMSettingsScreen, ProjectDefaultItem
from kimix_tui.session_index import SessionSummary


class FakeSession:
    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self.closed = False

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
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


async def session_loader(_work_dir: Path) -> list[SessionSummary]:
    return [
        SessionSummary(
            id="session-1",
            title="Existing session",
            preview="Existing session",
            updated_at=100.0,
        )
    ]


def config_files(tmp_path: Path) -> tuple[Path, Path]:
    first = write_llm_config(
        tmp_path / "first.json",
        model="first-model",
        display_name="First Model",
    )
    second = write_llm_config(
        tmp_path / "second.json",
        model="second-model",
        display_name="Second Model",
        provider="anthropic",
        base_url="https://api.anthropic.test/v1",
    )
    return first, second


def config_store(tmp_path: Path) -> LLMConfigStore:
    return LLMConfigStore(
        tmp_path / "metadata.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-tui.json"
        ),
    )


@pytest.mark.asyncio
async def test_session_without_config_continues_to_inherit_default(tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("session-1")

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )
    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert opened[0].config_file == first.resolve()
        assert store.session_for(tmp_path, "session-1") is None


@pytest.mark.asyncio
async def test_saved_session_config_overrides_startup_config(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_session(tmp_path, "session-1", inspect_llm_config(second))
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("session-1")

    app = KimixTuiApp(
        SessionOptions(tmp_path, config_file=first, model="cli-override"),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )
    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert opened[0].config_file == second.resolve()
        assert opened[0].model is None


@pytest.mark.asyncio
async def test_home_can_configure_existing_session(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.add_config(inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#configure-session")
        await pilot.pause()
        assert isinstance(app.screen, LLMSettingsScreen)
        assert list(app.screen.query("#config-path")) == []
        assert list(app.screen.query("#load-config")) == []
        assert list(app.screen.query("#delete-config")) == []
        item = next(
            item
            for item in app.screen.query(ConfigPathItem)
            if item.reference.path == second.resolve()
        )
        await pilot.click(item)
        await pilot.pause()
        assert str(app.screen.query_one("#config-model", Static).content) == "Second Model"
        assert str(app.screen.query_one("#config-provider", Static).content) == "anthropic"

        await pilot.click("#apply-settings")
        await pilot.pause()

        assert isinstance(app.screen, HomeScreen)
        saved = store.session_for(tmp_path, "session-1")
        assert saved is not None
        assert saved.path == second.resolve()
        assert str(app.screen.query_one("#detail-llm", Static).content) == "Second Model"


@pytest.mark.asyncio
async def test_session_can_return_to_project_default(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.set_session(tmp_path, "session-1", inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        assert str(home.query_one("#detail-llm", Static).content) == "Second Model"

        await pilot.click("#configure-session")
        await pilot.pause()
        settings = app.screen
        assert isinstance(settings, LLMSettingsScreen)
        project_default = settings.query_one(ProjectDefaultItem)
        await pilot.click(project_default)
        await pilot.pause()
        assert str(settings.query_one("#config-model", Static).content) == "First Model"

        await pilot.click("#apply-settings")
        await pilot.pause()
        home = app.screen
        assert isinstance(home, HomeScreen)
        assert store.session_for(tmp_path, "session-1") is None
        assert str(home.query_one("#detail-llm", Static).content) == "First Model"
        assert "project default" in str(home.query_one("#detail-config", Static).content)


@pytest.mark.asyncio
async def test_home_default_config_applies_to_new_session(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("new-session")

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )
    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#open-settings")
        await pilot.pause()
        assert isinstance(app.screen, LLMSettingsScreen)
        app.screen.query_one("#config-path", Input).value = str(second)
        await pilot.click("#load-config")
        await pilot.click("#apply-settings")
        await pilot.pause()

        assert isinstance(app.screen, HomeScreen)
        assert str(app.screen.query_one("#home-model", Static).content).endswith("Second Model")
        saved_default = store.default_for(tmp_path)
        assert saved_default is not None
        assert saved_default.path == second.resolve()

        await pilot.press("n")
        await app.workers.wait_for_complete()

        assert opened[0].config_file == second.resolve()
        saved_session = store.session_for(tmp_path, "new-session")
        assert saved_session is not None
        assert saved_session.path == second.resolve()


@pytest.mark.asyncio
async def test_add_config_does_not_change_project_default(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#open-settings")
        await pilot.pause()
        app.screen.query_one("#config-path", Input).value = str(second)
        await pilot.click("#load-config")
        await pilot.click("#cancel-settings")
        await pilot.pause()

        saved_default = store.default_for(tmp_path)
        assert saved_default is not None
        assert saved_default.path == first.resolve()
        assert {reference.path for reference in store.configs()} == {
            first.resolve(),
            second.resolve(),
        }


@pytest.mark.asyncio
async def test_project_settings_can_remove_non_default_config(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.add_config(inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#open-settings")
        await pilot.pause()
        settings = app.screen
        assert isinstance(settings, LLMSettingsScreen)
        assert settings.query_one("#delete-config", Button).disabled is True
        second_item = next(
            item
            for item in settings.query(ConfigPathItem)
            if item.reference.path == second.resolve()
        )
        await pilot.click(second_item)
        await pilot.pause()
        assert settings.query_one("#delete-config", Button).disabled is False

        await pilot.click("#delete-config")
        await pilot.pause()

        assert {reference.path for reference in store.configs()} == {first.resolve()}
        assert all(
            item.reference.path != second.resolve() for item in settings.query(ConfigPathItem)
        )
        assert settings.query_one("#delete-config", Button).disabled is True


@pytest.mark.asyncio
async def test_missing_session_config_requires_reconfiguration(tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.set_session(
        tmp_path,
        "session-1",
        unavailable_config_reference(tmp_path / "missing.json"),
    )
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        assert home.query_one("#open-session", Button).disabled is True
        assert "missing" in str(home.query_one("#detail-config", Static).content)

        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LLMSettingsScreen)


@pytest.mark.asyncio
async def test_deleted_session_config_keeps_path_until_reconfigured(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    reference = inspect_llm_config(first)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(second))
    store.set_session(tmp_path, "session-1", reference)
    first.unlink()
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("session-1")

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        assert str(home.query_one("#detail-llm", Static).content) == "Configuration unavailable"
        assert str(home.query_one("#detail-provider", Static).content) == "Unavailable"
        assert str(first.resolve()) in str(home.query_one("#detail-config", Static).content)
        assert home.query_one("#open-session", Button).disabled is True

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LLMSettingsScreen)
        assert str(app.screen.query_one("#config-model", Static).content) == "Configuration unavailable"
        assert "does not exist" in str(app.screen.query_one("#settings-error", Static).content)
        assert app.screen.query_one("#apply-settings", Button).disabled is True

        item = next(
            item
            for item in app.screen.query(ConfigPathItem)
            if item.reference.path == second.resolve()
        )
        await pilot.click(item)
        await pilot.click("#apply-settings")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#open-session", Button).disabled is False

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert opened[0].config_file == second.resolve()


@pytest.mark.asyncio
async def test_direct_resume_with_missing_config_opens_settings(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(second))
    store.set_session(tmp_path, "session-1", inspect_llm_config(first))
    first.unlink()
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("session-1")

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="session-1"),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert isinstance(app.screen, LLMSettingsScreen)
        assert opened == []


@pytest.mark.asyncio
async def test_settings_stacks_library_and_details_on_narrow_terminal(tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(70, 24)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#open-settings")
        await pilot.pause()
        settings = app.screen
        assert isinstance(settings, LLMSettingsScreen)
        sources = settings.query_one("#config-sources")
        details = settings.query_one("#config-details")
        dialog = settings.query_one("#settings-dialog")

        assert settings.has_class("-narrow")
        assert details.region.y > sources.region.y
        assert settings.query_one("#settings-actions").region.bottom <= dialog.region.bottom
        assert settings.query_one("#delete-config", Button).region.x >= dialog.region.x
        assert settings.query_one("#apply-settings", Button).region.right <= dialog.region.right


@pytest.mark.asyncio
async def test_home_refreshes_changed_external_config(tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_session(tmp_path, "session-1", inspect_llm_config(first))
    write_llm_config(
        first,
        model="changed-model",
        display_name="Changed Model",
        provider="google_genai",
        base_url="https://google.test/v1",
    )
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)):
        await app.workers.wait_for_complete()
        home = app.screen
        assert isinstance(home, HomeScreen)
        assert str(home.query_one("#detail-llm", Static).content) == "Changed Model"
        assert str(home.query_one("#detail-provider", Static).content) == "google_genai"


@pytest.mark.asyncio
async def test_settings_applies_external_json_config(tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    json_config = write_json_config(
        tmp_path / "external.json",
        model="claude-json",
        display_name="Claude JSON",
    )
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.add_config(inspect_llm_config(json_config))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )

    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.click("#configure-session")
        await pilot.pause()
        assert isinstance(app.screen, LLMSettingsScreen)
        item = next(
            item
            for item in app.screen.query(ConfigPathItem)
            if item.reference.path == json_config.resolve()
        )
        await pilot.click(item)
        await pilot.pause()

        assert str(app.screen.query_one("#config-format", Static).content) == "JSON"
        assert str(app.screen.query_one("#config-model", Static).content) == "Claude JSON"
        await pilot.click("#apply-settings")
        await pilot.pause()

        saved = store.session_for(tmp_path, "session-1")
        assert saved is not None
        assert saved.path == json_config.resolve()
        assert saved.file_format == "JSON"


@pytest.mark.asyncio
async def test_chat_config_change_is_delayed_until_next_resume(tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_session(tmp_path, "session-1", inspect_llm_config(first))
    store.add_config(inspect_llm_config(second))
    opened: list[SessionOptions] = []

    async def factory(options: SessionOptions) -> FakeSession:
        opened.append(options)
        return FakeSession("session-1")

    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        session_loader=session_loader,
        config_store=store,
    )
    async with app.run_test(size=(110, 35)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        assert opened[0].config_file == first.resolve()

        await pilot.press("f4")
        await pilot.pause()
        assert isinstance(app.screen, LLMSettingsScreen)
        item = next(
            item
            for item in app.screen.query(ConfigPathItem)
            if item.reference.path == second.resolve()
        )
        await pilot.click(item)
        await pilot.click("#apply-settings")
        await pilot.pause()

        assert app.screen is chat
        assert len(opened) == 1
        assert "next: Second Model" in str(chat.query_one("#status", Static).content)
        saved = store.session_for(tmp_path, "session-1")
        assert saved is not None
        assert saved.path == second.resolve()

        await pilot.press("escape")
        await app.workers.wait_for_complete()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert opened[1].config_file == second.resolve()

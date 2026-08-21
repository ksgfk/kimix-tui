from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QPushButton
from qtutil import find, launch_app, wait_chat_ready, wait_home, widget_text
from test_llm_config import write_json_config, write_llm_config

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.llm_config import (
    LLMConfigStore,
    inspect_llm_config,
    unavailable_config_reference,
)
from kimix_tui.qt.settings_dialog import LLMSettingsDialog
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
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )


def _select_config(dialog: LLMSettingsDialog, path: Path) -> None:
    target = path.resolve()
    for row, item in enumerate(dialog.config_items()):
        if item.project_default:
            continue
        if item.reference.path.resolve() == target:
            dialog._list.setCurrentItem(item)
            return
    raise AssertionError(f"config {path} not listed")


def _wait_settings(qtbot, app: KimixTuiApp) -> LLMSettingsDialog:
    qtbot.waitUntil(lambda: isinstance(app.screen, LLMSettingsDialog), timeout=10_000)
    dialog = app.screen
    assert isinstance(dialog, LLMSettingsDialog)
    return dialog


def test_new_session_without_llm_shows_configuration_toast(qtbot, tmp_path: Path) -> None:
    store = config_store(tmp_path)
    store.set_default(tmp_path, unavailable_config_reference(tmp_path / "missing.json"))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert find(home, "start-new-session", QPushButton).isEnabled()
    qtbot.mouseClick(find(home, "start-new-session"), Qt.MouseButton.LeftButton)
    _wait_settings(qtbot, app)
    notification = list(app._notifications)[-1]
    assert notification.title == "LLM configuration required"
    assert notification.message == "Select a valid LLM configuration to continue."
    assert notification.severity == "warning"


def test_session_without_config_continues_to_inherit_default(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    wait_chat_ready(qtbot, app)
    assert opened[0].config_file == first.resolve()
    assert store.session_for(tmp_path, "session-1") is None


def test_saved_session_config_overrides_startup_config(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    wait_chat_ready(qtbot, app)
    assert opened[0].config_file == second.resolve()
    assert opened[0].model is None


def test_home_can_configure_existing_session(qtbot, tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.add_config(inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "configure-session"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    assert dialog.findChild(QLineEdit, "config-path") is None
    assert dialog.findChild(QPushButton, "browse-config") is None
    assert dialog.findChild(QPushButton, "load-config") is None
    assert dialog.findChild(QPushButton, "delete-config") is None
    _select_config(dialog, second)
    assert widget_text(dialog, "config-model") == "Second Model"
    assert widget_text(dialog, "config-provider") == "anthropic"
    find(dialog, "apply-settings", QPushButton).click()
    home = wait_home(qtbot, app)
    saved = store.session_for(tmp_path, "session-1")
    assert saved is not None
    assert saved.path == second.resolve()
    assert widget_text(home, "detail-llm") == "Second Model"


def test_session_can_return_to_project_default(qtbot, tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.set_session(tmp_path, "session-1", inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert widget_text(home, "detail-llm") == "Second Model"
    qtbot.mouseClick(find(home, "configure-session"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    dialog._list.setCurrentRow(0)
    assert widget_text(dialog, "config-model") == "First Model"
    find(dialog, "apply-settings", QPushButton).click()
    home = wait_home(qtbot, app)
    assert store.session_for(tmp_path, "session-1") is None
    assert widget_text(home, "detail-llm") == "First Model"
    assert "project default" in widget_text(home, "detail-config")


def test_home_default_config_applies_to_new_session(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    find(dialog, "config-path", QLineEdit).setText(str(second))
    find(dialog, "load-config", QPushButton).click()
    find(dialog, "apply-settings", QPushButton).click()
    home = wait_home(qtbot, app)
    assert widget_text(home, "home-model").endswith("Second Model")
    saved_default = store.default_for(tmp_path)
    assert saved_default is not None
    assert saved_default.path == second.resolve()

    qtbot.keyClick(home, Qt.Key.Key_N)
    wait_chat_ready(qtbot, app)
    assert opened[0].config_file == second.resolve()
    saved_session = store.session_for(tmp_path, "new-session")
    assert saved_session is not None
    assert saved_session.path == second.resolve()


def test_add_config_does_not_change_project_default(qtbot, tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    find(dialog, "config-path", QLineEdit).setText(str(second))
    find(dialog, "load-config", QPushButton).click()
    find(dialog, "cancel-settings", QPushButton).click()
    wait_home(qtbot, app)
    saved_default = store.default_for(tmp_path)
    assert saved_default is not None
    assert saved_default.path == first.resolve()
    assert {reference.path for reference in store.configs()} == {
        first.resolve(),
        second.resolve(),
    }


def test_settings_browse_adds_json_config(qtbot, tmp_path: Path, monkeypatch) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    captured: dict[str, str] = {}

    def fake_pick(_parent: object, start_directory: str) -> Path:
        captured["start"] = start_directory
        return second

    monkeypatch.setattr("kimix_tui.qt.settings_dialog.pick_json_file", fake_pick)
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    find(dialog, "browse-config", QPushButton).click()
    assert captured["start"] == str(first.parent)
    assert find(dialog, "config-path", QLineEdit).text() == str(second.resolve())
    assert widget_text(dialog, "config-model") == "Second Model"
    find(dialog, "cancel-settings", QPushButton).click()
    wait_home(qtbot, app)
    assert {reference.path for reference in store.configs()} == {
        first.resolve(),
        second.resolve(),
    }


def test_settings_browse_cancel_leaves_library_unchanged(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    monkeypatch.setattr(
        "kimix_tui.qt.settings_dialog.pick_json_file",
        lambda _parent, _start_directory: None,
    )
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    find(dialog, "browse-config", QPushButton).click()
    assert find(dialog, "config-path", QLineEdit).text() == str(first.resolve())
    assert widget_text(dialog, "config-model") == "First Model"
    assert {reference.path for reference in store.configs()} == {first.resolve()}


def test_project_settings_can_remove_non_default_config(qtbot, tmp_path: Path) -> None:
    first, second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    store.add_config(inspect_llm_config(second))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    assert find(dialog, "delete-config", QPushButton).isEnabled() is False
    _select_config(dialog, second)
    assert find(dialog, "delete-config", QPushButton).isEnabled()
    find(dialog, "delete-config", QPushButton).click()
    assert {reference.path for reference in store.configs()} == {first.resolve()}
    assert all(
        item.reference.path.resolve() != second.resolve()
        for item in dialog.config_items()
        if not item.project_default
    )
    assert find(dialog, "delete-config", QPushButton).isEnabled() is False


def test_missing_session_config_requires_reconfiguration(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert find(home, "open-session", QPushButton).isEnabled()
    assert "missing" in widget_text(home, "detail-config")
    qtbot.mouseClick(find(home, "open-session"), Qt.MouseButton.LeftButton)
    _wait_settings(qtbot, app)
    notification = list(app._notifications)[-1]
    assert notification.title == "LLM configuration required"
    assert notification.message == "Select a valid LLM configuration to continue."
    assert notification.severity == "warning"


def test_deleted_session_config_keeps_path_until_reconfigured(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert widget_text(home, "detail-llm") == "Configuration unavailable"
    assert widget_text(home, "detail-provider") == "Unavailable"
    assert str(first.resolve()) in widget_text(home, "detail-config")
    assert find(home, "open-session", QPushButton).isEnabled()

    qtbot.keyClick(home, Qt.Key.Key_Return)
    dialog = _wait_settings(qtbot, app)
    assert widget_text(dialog, "config-model") == "Configuration unavailable"
    assert "does not exist" in widget_text(dialog, "settings-error")
    assert find(dialog, "apply-settings", QPushButton).isEnabled() is False

    _select_config(dialog, second)
    find(dialog, "apply-settings", QPushButton).click()
    home = wait_home(qtbot, app)
    assert find(home, "open-session", QPushButton).isEnabled()

    qtbot.keyClick(home, Qt.Key.Key_Return)
    wait_chat_ready(qtbot, app)
    assert opened[0].config_file == second.resolve()


def test_direct_resume_with_missing_config_opens_settings(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    _wait_settings(qtbot, app)
    assert opened == []


def test_settings_stacks_library_and_details_on_narrow_window(qtbot, tmp_path: Path) -> None:
    first, _second = config_files(tmp_path)
    store = config_store(tmp_path)
    store.set_default(tmp_path, inspect_llm_config(first))
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_loader=session_loader,
        config_store=store,
    )
    launch_app(qtbot, app, size=(700, 560))
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "open-settings"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    dialog.resize(700, 560)
    qtbot.waitUntil(lambda: dialog._splitter.orientation() == Qt.Orientation.Vertical, timeout=5_000)
    sources = find(dialog, "config-sources")
    details = find(dialog, "config-details")
    assert details.y() > sources.y()
    assert find(dialog, "settings-actions").geometry().bottom() <= dialog.rect().bottom()
    assert find(dialog, "delete-config").x() >= 0
    assert find(dialog, "apply-settings").geometry().right() <= dialog.width()


def test_home_refreshes_changed_external_config(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    assert widget_text(home, "detail-llm") == "Changed Model"
    assert widget_text(home, "detail-provider") == "google_genai"


def test_settings_applies_external_json_config(qtbot, tmp_path: Path) -> None:
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
    launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.mouseClick(find(home, "configure-session"), Qt.MouseButton.LeftButton)
    dialog = _wait_settings(qtbot, app)
    _select_config(dialog, json_config)
    assert widget_text(dialog, "config-format") == "JSON"
    assert widget_text(dialog, "config-model") == "Claude JSON"
    find(dialog, "apply-settings", QPushButton).click()
    wait_home(qtbot, app)
    saved = store.session_for(tmp_path, "session-1")
    assert saved is not None
    assert saved.path == json_config.resolve()
    assert saved.file_format == "JSON"


def test_chat_config_change_is_delayed_until_next_resume(qtbot, tmp_path: Path) -> None:
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
    window = launch_app(qtbot, app)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    chat = wait_chat_ready(qtbot, app)
    assert opened[0].config_file == first.resolve()

    qtbot.keyClick(window, Qt.Key.Key_F4)
    dialog = _wait_settings(qtbot, app)
    _select_config(dialog, second)
    find(dialog, "apply-settings", QPushButton).click()
    qtbot.waitUntil(lambda: app.screen is chat, timeout=10_000)
    assert len(opened) == 1
    assert "next: Second Model" in widget_text(chat, "status")
    saved = store.session_for(tmp_path, "session-1")
    assert saved is not None
    assert saved.path == second.resolve()

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    home = wait_home(qtbot, app)
    qtbot.keyClick(home, Qt.Key.Key_Return)
    wait_chat_ready(qtbot, app)
    assert opened[1].config_file == second.resolve()

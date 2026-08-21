"""Application-level routing between home and chat windows."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from functools import partial

from PySide6.QtWidgets import QApplication

from kimix_tui.backend import SessionOptions, create_sdk_session
from kimix_tui.history import HistoryLoader
from kimix_tui.llm_config import (
    LLMConfigError,
    LLMConfigReference,
    LLMConfigStore,
    config_file_available,
    default_config_path,
    inspect_llm_config,
    unavailable_config_reference,
)
from kimix_tui.qt.bridge import KimixBridge, SessionFactory
from kimix_tui.qt.main_window import MainWindow
from kimix_tui.qt.settings_dialog import LLMSettingsDialog, LLMSettingsResult
from kimix_tui.qt.theme import apply_theme
from kimix_tui.session_index import SessionDeleter, SessionLoader


@dataclass(frozen=True, slots=True)
class AppNotification:
    message: str
    title: str = ""
    severity: str = "information"


class KimixTuiApp:
    """Route between home and chat, owning LLM config and the Kimix bridge."""

    TITLE = "Kimix"

    def __init__(
        self,
        options: SessionOptions,
        *,
        session_factory: SessionFactory = create_sdk_session,
        session_loader: SessionLoader | None = None,
        history_loader: HistoryLoader | None = None,
        config_store: LLMConfigStore | None = None,
        session_deleter: SessionDeleter | None = None,
    ) -> None:
        self._options = options
        self._session_factory = session_factory
        self._session_loader = session_loader
        self._history_loader = history_loader
        self._session_deleter = session_deleter
        self._config_store = config_store or LLMConfigStore()
        startup_path = options.config_file or default_config_path()
        try:
            startup_config = inspect_llm_config(startup_path, model_override=options.model)
        except LLMConfigError:
            startup_config = unavailable_config_reference(
                startup_path,
                model_override=options.model,
            )
        saved_default = self._config_store.default_for(options.work_dir)
        if saved_default is not None:
            saved_default = self._refresh_reference(saved_default)
        has_startup_override = options.config_file is not None or options.model is not None
        self._default_config = (
            startup_config if has_startup_override else saved_default or startup_config
        )
        self.bridge = KimixBridge(
            session_factory=session_factory,
            history_loader=history_loader,
            session_loader=session_loader,
            session_deleter=session_deleter,
        )
        self.window: MainWindow | None = None
        self._notifications: list[AppNotification] = []
        self._qt_app: QApplication | None = None

    @property
    def options(self) -> SessionOptions:
        return self._options

    @property
    def default_config(self) -> LLMConfigReference:
        return self._default_config

    @property
    def screen(self) -> object:
        if self.window is None:
            return None
        return self.window.current_view

    def note(self, message: str, severity: str = "information", title: str = "") -> None:
        self._notifications.append(AppNotification(message, title, severity))

    def session_config(self, session_id: str) -> LLMConfigReference | None:
        reference = self._config_store.session_for(self._options.work_dir, session_id)
        return self._refresh_reference(reference) if reference is not None else None

    @staticmethod
    def _refresh_reference(reference: LLMConfigReference) -> LLMConfigReference:
        try:
            refreshed = inspect_llm_config(
                reference.path,
                model_override=reference.model_override,
            )
        except LLMConfigError as exc:
            return replace(reference, error=str(exc))
        return refreshed

    def ensure_application(self) -> QApplication:
        existing = QApplication.instance()
        if existing is None:
            self._qt_app = QApplication(sys.argv)
            apply_theme(self._qt_app)
            return self._qt_app
        apply_theme(existing)
        return existing

    def create_window(self) -> MainWindow:
        self.ensure_application()
        self.bridge.start()
        self.window = MainWindow(self)
        self._startup()
        return self.window

    def run(self) -> None:
        qt_app = self.ensure_application()
        window = self.create_window()
        window.show()
        raise SystemExit(qt_app.exec())

    def shutdown(self) -> None:
        self.bridge.stop()

    def _startup(self) -> None:
        if self._options.session_id:
            reference = self.session_config(self._options.session_id) or self._default_config
            if not config_file_available(reference):
                self._show_home()
                self.open_llm_settings(self._options.session_id)
                return
            self._prepare_session_options(self._options.session_id, reference)
            self._show_chat(reference, record_session_config=False)
            return
        self._show_home()

    def _show_home(self, *, reload: bool = True) -> None:
        if self.window is None:
            return
        self.window.show_home(reload=reload)

    def start_new_session(self) -> None:
        if not config_file_available(self._default_config):
            if self.window is not None:
                self.window.show_notification(
                    "Select a valid LLM configuration to continue.",
                    "warning",
                    "LLM configuration required",
                )
            self.open_llm_settings(None)
            return
        self._prepare_session_options(None, self._default_config)
        self._show_chat(self._default_config, record_session_config=True)

    def resume_session(self, session_id: str) -> None:
        reference = self.session_config(session_id) or self._default_config
        self._prepare_session_options(session_id, reference)
        self._show_chat(reference, record_session_config=False)

    def leave_chat(self) -> None:
        self.bridge.close_session()
        self._options = replace(self._options, session_id=None)
        self._show_home(reload=False)
        if self.window is not None:
            self.window.remove_chat()

    def open_chat_settings(self) -> None:
        session_id = self.bridge.session_id or self._options.session_id
        if session_id is not None:
            self.open_llm_settings(session_id)

    def _show_chat(self, reference: LLMConfigReference, *, record_session_config: bool) -> None:
        if self.window is None:
            return
        if not config_file_available(reference):
            session_id = self._options.session_id
            self.window.show_notification(
                reference.error or "LLM configuration is unavailable",
                "error",
                "",
            )
            self._show_home()
            self.open_llm_settings(session_id)
            return
        chat = self.window.show_chat()
        on_opened = (
            partial(self._record_session_config, reference=reference)
            if record_session_config
            else None
        )
        self.bridge.open_session(self._options, on_session_opened=on_opened)
        chat._session_label = "connecting…"

    def _prepare_session_options(
        self,
        session_id: str | None,
        reference: LLMConfigReference,
    ) -> None:
        self._options = replace(
            self._options,
            session_id=session_id,
            config_file=reference.path,
            model=reference.model_override,
        )

    def _record_session_config(self, session_id: str, *, reference: LLMConfigReference) -> None:
        self._config_store.set_session(self._options.work_dir, session_id, reference)

    def open_llm_settings(self, session_id: str | None) -> None:
        if self.window is None:
            return
        session_config = self.session_config(session_id) if session_id is not None else None
        current = session_config or self._default_config
        references = [
            self._refresh_reference(reference)
            for reference in self._config_store.references_for(self._options.work_dir)
        ]
        scope_label = f"Session {session_id}" if session_id is not None else "New sessions"
        dialog = LLMSettingsDialog(
            current=current,
            references=references,
            scope_label=scope_label,
            project_default=self._default_config if session_id is not None else None,
            inherits_project_default=session_id is not None and session_config is None,
            manage_library=session_id is None,
            registrar=self._config_store.add_config,
            remover=self._config_store.remove_config,
            parent=self.window,
        )
        self.window.set_modal(dialog)

        def _applied(result: object) -> None:
            if isinstance(result, LLMSettingsResult):
                self._on_llm_settings(result, session_id)

        dialog.applied.connect(_applied)
        dialog.finished.connect(lambda: self.window is not None and self.window.set_modal(None))
        dialog.open()

    def _on_llm_settings(self, result: LLMSettingsResult, session_id: str | None) -> None:
        reference = result.reference
        try:
            if session_id is None:
                self._config_store.set_default(self._options.work_dir, reference)
                self._default_config = reference
            elif result.use_project_default:
                self._config_store.clear_session(self._options.work_dir, session_id)
            else:
                self._config_store.set_session(self._options.work_dir, session_id, reference)
        except OSError as exc:
            if self.window is not None:
                self.window.show_notification(
                    f"Failed to save LLM configuration metadata: {exc}",
                    "error",
                    "",
                )
            return
        if self.window is None:
            return
        if self.window.home is not None:
            self.window.home.refresh_configuration(self._default_config)
        if self.window.chat is not None:
            label = (
                f"Project default · {reference.label}"
                if result.use_project_default
                else reference.label
            )
            self.window.chat.set_pending_config(label)

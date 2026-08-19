"""Application-level routing between Textual screens."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from textual import on
from textual.app import App

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
from kimix_tui.screens.chat import ChatScreen, SessionFactory
from kimix_tui.screens.home import HomeScreen, SessionChoice
from kimix_tui.screens.settings import LLMSettingsResult, LLMSettingsScreen, OpenLLMSettings
from kimix_tui.session_index import SessionDeleter, SessionLoader


class KimixTuiApp(App[None]):
    """Route between full-screen application states."""

    TITLE = "Kimix TUI"
    SUB_TITLE = "Kimix Worker"

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
        super().__init__()
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

    def on_mount(self) -> None:
        if self._options.session_id:
            reference = self._session_config(self._options.session_id) or self._default_config
            if not config_file_available(reference):
                session_id = self._options.session_id
                self._show_home()
                self.call_after_refresh(partial(self._show_llm_settings, session_id))
                return
            self._prepare_session_options(self._options.session_id, reference)
            self._show_chat(reference, record_session_config=False)
            return
        self._show_home()

    def _show_home(self) -> None:
        if isinstance(self.screen, HomeScreen):
            return
        self.push_screen(
            HomeScreen(
                self._options.work_dir,
                default_config=self._default_config,
                session_config_loader=self._session_config,
                loader=self._session_loader,
                deleter=self._session_deleter,
            ),
            self._on_home_choice,
        )

    def _on_home_choice(self, choice: SessionChoice | None) -> None:
        if choice is None or choice.action == "quit":
            self.exit()
            return
        if choice.action == "resume" and choice.session_id:
            reference = self._session_config(choice.session_id) or self._default_config
            self._prepare_session_options(choice.session_id, reference)
        else:
            reference = self._default_config
            self._prepare_session_options(None, reference)
        self.title = self.TITLE
        self.sub_title = self.SUB_TITLE
        self._show_chat(reference, record_session_config=choice.action == "new")

    def _show_chat(
        self,
        reference: LLMConfigReference,
        *,
        record_session_config: bool,
    ) -> None:
        if isinstance(self.screen, ChatScreen):
            return
        if not config_file_available(reference):
            session_id = self._options.session_id
            self.notify(reference.error or "LLM configuration is unavailable", severity="error")
            self._show_home()
            self.call_after_refresh(partial(self._show_llm_settings, session_id))
            return
        self.push_screen(
            ChatScreen(
                self._options,
                session_factory=self._session_factory,
                history_loader=self._history_loader,
                on_session_opened=(
                    partial(self._record_session_config, reference=reference)
                    if record_session_config
                    else None
                ),
            ),
            self._on_chat_closed,
        )

    def _on_chat_closed(self, _result: None) -> None:
        self._options = replace(self._options, session_id=None)
        self._show_home()

    def _session_config(self, session_id: str) -> LLMConfigReference | None:
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

    def _record_session_config(
        self,
        session_id: str,
        *,
        reference: LLMConfigReference,
    ) -> None:
        self._config_store.set_session(self._options.work_dir, session_id, reference)

    @on(OpenLLMSettings)
    def open_llm_settings(self, event: OpenLLMSettings) -> None:
        event.stop()
        self._show_llm_settings(event.session_id)

    def _show_llm_settings(self, session_id: str | None) -> None:
        source_screen = self.screen
        session_config = self._session_config(session_id) if session_id is not None else None
        current = session_config or self._default_config
        references = [
            self._refresh_reference(reference)
            for reference in self._config_store.references_for(self._options.work_dir)
        ]
        scope_label = f"Session {session_id}" if session_id is not None else "New sessions"
        self.push_screen(
            LLMSettingsScreen(
                current=current,
                references=references,
                scope_label=scope_label,
                project_default=self._default_config if session_id is not None else None,
                inherits_project_default=session_id is not None and session_config is None,
                manage_library=session_id is None,
                registrar=self._config_store.add_config,
                remover=self._config_store.remove_config,
            ),
            partial(
                self._on_llm_settings_closed,
                session_id=session_id,
                source_screen=source_screen,
            ),
        )

    def _on_llm_settings_closed(
        self,
        result: LLMSettingsResult | None,
        *,
        session_id: str | None,
        source_screen: object,
    ) -> None:
        if result is None:
            return
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
            self.notify(f"Failed to save LLM configuration metadata: {exc}", severity="error")
            return

        if isinstance(source_screen, HomeScreen):
            source_screen.refresh_configuration(self._default_config)
        elif isinstance(source_screen, ChatScreen):
            label = (
                f"Project default · {reference.label}"
                if result.use_project_default
                else reference.label
            )
            source_screen.set_pending_config(label)

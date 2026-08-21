"""Stacked main window: home, chat, and modal dialogs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QLabel, QMainWindow, QStackedWidget, QWidget

from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.home_view import HomeView
from kimix_tui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog
from kimix_tui.qt.theme import COLORS

if TYPE_CHECKING:
    from kimix_tui.app import KimixTuiApp


class Toast(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toast")
        self.setStyleSheet(
            f"background: {COLORS['panel']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 8px 12px;"
        )
        self.hide()

    def show_message(self, message: str, title: str = "") -> None:
        self.setText(f"{title}\n{message}" if title else message)
        self.adjustSize()
        self.move(16, self.parent().height() - self.height() - 16)  # type: ignore[union-attr]
        self.show()
        self.raise_()


class MainWindow(QMainWindow):
    """Top-level window hosting Home and Chat views."""

    def __init__(self, app: KimixTuiApp) -> None:
        super().__init__()
        self.controller = app
        self.setWindowTitle("Kimix")
        self.resize(1100, 720)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self.home: HomeView | None = None
        self.chat: ChatView | None = None
        self._modal: QWidget | None = None
        self._toast = Toast(self)
        self._esc_shortcut: QShortcut | None = None
        self._install_shortcuts()

    @property
    def current_view(self) -> QWidget | None:
        if self._modal is not None:
            return self._modal
        return self._stack.currentWidget()

    def show_home(self, *, reload: bool = True) -> HomeView:
        created = self.home is None
        if created:
            self.home = HomeView(
                self.controller.options.work_dir,
                default_config=self.controller.default_config,
                session_config_loader=self.controller.session_config,
            )
            self._connect_home(self.home)
            self._stack.addWidget(self.home)
        self._stack.setCurrentWidget(self.home)
        if reload or created:
            self.controller.bridge.load_sessions(self.controller.options.work_dir)
        self.home.refresh_configuration(self.controller.default_config)
        return self.home

    def show_chat(self) -> ChatView:
        self.remove_chat()
        self.chat = ChatView(self.controller.bridge)
        self._connect_chat(self.chat)
        self._stack.addWidget(self.chat)
        self._stack.setCurrentWidget(self.chat)
        return self.chat

    def remove_chat(self) -> None:
        if self.chat is None:
            return
        self.chat.disconnect_bridge()
        chat = self.chat
        self.chat = None
        self._stack.removeWidget(chat)
        chat.deleteLater()

    def set_modal(self, widget: QWidget | None) -> None:
        self._modal = widget
        if self._esc_shortcut is not None:
            self._esc_shortcut.setEnabled(widget is None)

    def _connect_home(self, home: HomeView) -> None:
        home.new_session.connect(self.controller.start_new_session)
        home.resume_session.connect(self.controller.resume_session)
        home.open_settings.connect(lambda: self.controller.open_llm_settings(None))
        home.configure_session.connect(self.controller.open_llm_settings)
        home.quit_requested.connect(self.close)
        home.delete_requested.connect(self._confirm_delete)
        home.llm_required.connect(self._home_llm_required)
        self.controller.bridge.sessions_listed.connect(home.show_sessions)
        self.controller.bridge.sessions_list_failed.connect(home.show_load_error)
        self.controller.bridge.sessions_deleted.connect(home.apply_deleted)
        self.controller.bridge.notify.connect(self.show_notification)

    def _connect_chat(self, chat: ChatView) -> None:
        chat.leave_requested.connect(self.controller.leave_chat)
        chat.open_settings.connect(self.controller.open_chat_settings)
        chat.approval_asked.connect(self._show_approval)
        chat.question_asked.connect(self._show_question)
        chat.notify.connect(self.show_notification)

    def _home_llm_required(self, session_id: object) -> None:
        self.show_notification(
            "Select a valid LLM configuration to continue.",
            "warning",
            "LLM configuration required",
        )
        self.controller.open_llm_settings(session_id if isinstance(session_id, str) else None)

    def _confirm_delete(self, ids: list[str]) -> None:
        dialog = DeleteSessionsDialog(len(ids), self)
        self.set_modal(dialog)

        def _done(result: int) -> None:
            self.set_modal(None)
            if result != int(QDialog.DialogCode.Accepted):
                return
            self.controller.bridge.delete_sessions(self.controller.options.work_dir, ids)
            noun = "session" if len(ids) == 1 else "sessions"
            self.show_notification(f"Deleted {len(ids)} {noun}", "information", "")

        dialog.finished.connect(_done)
        dialog.open()

    def _show_approval(self, ask: object) -> None:
        from kimix_tui.qt.bridge import ApprovalAsk

        if not isinstance(ask, ApprovalAsk):
            return
        dialog = ApprovalDialog(ask.title, ask.description, self)
        self.set_modal(dialog)

        def _done(decision: str) -> None:
            self.set_modal(None)
            self.controller.bridge.resolve_request(ask.token, ask.epoch, decision)

        dialog.decided.connect(_done)
        dialog.open()

    def _show_question(self, ask: object) -> None:
        from kimix_tui.qt.bridge import QuestionAsk

        if not isinstance(ask, QuestionAsk):
            return
        dialog = QuestionDialog(ask.prompt, ask.body, self)
        self.set_modal(dialog)

        def _done(answer: object) -> None:
            self.set_modal(None)
            self.controller.bridge.resolve_request(ask.token, ask.epoch, answer)

        dialog.answered.connect(_done)
        dialog.open()

    def show_notification(self, message: str, severity: str = "information", title: str = "") -> None:
        self.controller.note(message, severity, title)
        self._toast.show_message(message, title)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_F4), self, self._shortcut_settings)
        self._esc_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._shortcut_escape)
        QShortcut(QKeySequence("Ctrl+G"), self, self._shortcut_cancel)
        QShortcut(QKeySequence("Ctrl+Up"), self, self._shortcut_older)
        QShortcut(QKeySequence("Ctrl+End"), self, self._shortcut_latest)
        QShortcut(QKeySequence(Qt.Key.Key_F2), self, self._shortcut_prompt)
        QShortcut(QKeySequence(Qt.Key.Key_F3), self, self._shortcut_turn)

    def _on_home(self) -> bool:
        return isinstance(self.current_view, HomeView)

    def _on_chat(self) -> bool:
        return isinstance(self.current_view, ChatView)

    def _shortcut_settings(self) -> None:
        if self._on_home():
            self.controller.open_llm_settings(None)
        elif self._on_chat():
            self.controller.open_chat_settings()

    def _shortcut_escape(self) -> None:
        if self._modal is not None:
            return
        if self._on_chat():
            self.controller.leave_chat()
        elif self._on_home():
            self.close()

    def _shortcut_cancel(self) -> None:
        if self.chat is not None and self._on_chat():
            self.chat.cancel_prompt()

    def _shortcut_older(self) -> None:
        if self.chat is not None and self._on_chat():
            self.chat.load_older_history()

    def _shortcut_latest(self) -> None:
        if self.chat is not None and self._on_chat():
            self.chat.jump_to_latest()

    def _shortcut_prompt(self) -> None:
        if self.chat is not None and self._on_chat():
            self.chat.focus_prompt()

    def _shortcut_turn(self) -> None:
        if self.chat is not None and self._on_chat():
            self.chat.focus_history_turn()

    def closeEvent(self, event: object) -> None:
        self.controller.shutdown()
        super().closeEvent(event)  # type: ignore[arg-type]

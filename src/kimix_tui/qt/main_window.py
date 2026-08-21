"""Stacked main window: home, chat, and modal dialogs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QKeySequence, QMouseEvent, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsOpacityEffect,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.home_view import HomeView
from kimix_tui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog

if TYPE_CHECKING:
    from kimix_tui.app import KimixTuiApp


class Toast(QLabel):
    """Centered snackbar that auto-hides after a short delay."""

    INFO_MS = 2_200
    WARN_MS = 4_000

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toast")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade.finished.connect(self._on_fade_finished)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._hiding = False
        self.hide()

    def show_message(
        self,
        message: str,
        title: str = "",
        *,
        severity: str = "information",
        duration_ms: int | None = None,
    ) -> None:
        self._hiding = False
        self._fade.stop()
        self._timer.stop()
        self._effect.setOpacity(1.0)
        self.setText(f"{title}\n{message}" if title else message)
        parent = self.parentWidget()
        self.setMaximumWidth(max(240, (parent.width() if parent is not None else 420) - 48))
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        if duration_ms is None:
            duration_ms = self.WARN_MS if severity == "warning" else self.INFO_MS
        if duration_ms > 0:
            self._timer.start(duration_ms)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = max(16, (parent.width() - self.width()) // 2)
        y = max(16, parent.height() - self.height() - 28)
        self.move(x, y)

    def mousePressEvent(self, event: object) -> None:
        if isinstance(event, QMouseEvent):
            self._timer.stop()
            self._fade_out()
            event.accept()
            return
        super().mousePressEvent(event)  # type: ignore[arg-type]

    def _fade_out(self) -> None:
        if not self.isVisible():
            return
        self._hiding = True
        self._fade.stop()
        self._fade.setDuration(160)
        self._fade.setStartValue(max(0.0, self._effect.opacity()))
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self) -> None:
        if not self._hiding:
            return
        self.hide()
        self._hiding = False
        self._effect.setOpacity(1.0)


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
        self._toast.show_message(message, title, severity=severity)

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

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self._toast.isVisible():
            self._toast.reposition()

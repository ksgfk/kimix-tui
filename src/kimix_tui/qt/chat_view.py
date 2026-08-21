"""Chat view: transcript, composer, history toolbar, and Kimix bridge wiring."""

from __future__ import annotations

from contextlib import suppress

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from kimix_tui.qt.bridge import HistoryPage, KimixBridge, TranscriptDelta
from kimix_tui.qt.composer import Composer
from kimix_tui.qt.transcript import Transcript


class ChatView(QWidget):
    """Run one SDK session inside a full-window chat interface."""

    leave_requested = Signal()
    open_settings = Signal()
    approval_asked = Signal(object)
    question_asked = Signal(object)
    notify = Signal(str, str, str)

    def __init__(self, bridge: KimixBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chat-view")
        self.bridge = bridge
        self._epoch = 0
        self._pending_config_label: str | None = None
        self._history_total = 0
        self._history_loading = False
        self._session_label = "connecting…"
        self._context_text = ""
        self._build()
        self._connect_bridge()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("chat-toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        title = QLabel("CHAT")
        title.setObjectName("chat-title")
        self._status = QLabel("connecting…")
        self._status.setObjectName("status")
        settings = QPushButton("Settings")
        settings.setObjectName("open-settings")
        home = QPushButton("Home")
        home.setObjectName("leave-session")
        toolbar_layout.addWidget(title)
        toolbar_layout.addWidget(self._status, 1)
        toolbar_layout.addWidget(settings)
        toolbar_layout.addWidget(home)
        root.addWidget(toolbar)

        history = QFrame()
        history.setObjectName("history-toolbar")
        history_layout = QHBoxLayout(history)
        self._history_info = QLabel("History · connecting…")
        self._history_info.setObjectName("history-info")
        self._older = QPushButton("←")
        self._older.setObjectName("load-older")
        self._older.setToolTip("Previous turn")
        self._turn_input = QLineEdit()
        self._turn_input.setObjectName("history-turn")
        self._turn_input.setPlaceholderText("Turn #")
        self._turn_input.setValidator(QIntValidator(1, 1_000_000, self))
        self._turn_input.setEnabled(False)
        self._turn_input.setFixedWidth(72)
        self._turn_input.setToolTip("Seek to turn")
        self._newer = QPushButton("→")
        self._newer.setObjectName("load-newer")
        self._newer.setEnabled(False)
        self._newer.setToolTip("Next turn")
        self._latest = QPushButton("↓")
        self._latest.setObjectName("jump-latest")
        self._latest.setEnabled(False)
        self._latest.setToolTip("Jump to latest")
        history_layout.addWidget(self._history_info, 1)
        history_layout.addWidget(self._older)
        history_layout.addWidget(self._turn_input)
        history_layout.addWidget(self._newer)
        history_layout.addWidget(self._latest)
        root.addWidget(history)

        self.transcript = Transcript()
        root.addWidget(self.transcript, 1)

        self.prompt = Composer("Ask AI, or type /help")
        self.prompt.setEnabled(False)
        root.addWidget(self.prompt)

        footer = QFrame()
        footer.setObjectName("chat-footer")
        footer_layout = QHBoxLayout(footer)
        hint = QLabel("Ctrl+G cancel")
        hint.setObjectName("chat-hint")
        self._context = QLabel("")
        self._context.setObjectName("context")
        self._context.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(hint)
        footer_layout.addWidget(self._context, 1)
        root.addWidget(footer)

        settings.clicked.connect(self.open_settings.emit)
        home.clicked.connect(self.leave_requested.emit)
        self._older.clicked.connect(self.load_older_history)
        self._newer.clicked.connect(self.load_newer_history)
        self._latest.clicked.connect(self.jump_to_latest)
        self._turn_input.returnPressed.connect(self._submit_turn)
        self.prompt.submitted.connect(self._submit_prompt)
        self.transcript.reached_top.connect(self._on_reached_top)
        self.transcript.reached_bottom.connect(self._on_reached_bottom)
        self.transcript.viewport_turn_changed.connect(lambda _turn: self._update_history_toolbar())
        self.transcript.record_copied.connect(
            lambda label: self.notify.emit(f"{label} message copied", "information", "")
        )

    def _connect_bridge(self) -> None:
        bridge = self.bridge
        bridge.session_opened.connect(self._on_session_opened)
        bridge.session_failed.connect(self._on_session_failed)
        bridge.transcript_delta.connect(self._on_delta)
        bridge.history_page.connect(self._on_history_page)
        bridge.history_loading.connect(self._on_history_loading)
        bridge.input_enabled.connect(self._on_input_enabled)
        bridge.approval_asked.connect(self._forward_approval)
        bridge.question_asked.connect(self._forward_question)
        bridge.generation_started.connect(self._on_generation_started)
        bridge.generation_finished.connect(self._on_generation_finished)

    def disconnect_bridge(self) -> None:
        bridge = self.bridge
        for signal, slot in (
            (bridge.session_opened, self._on_session_opened),
            (bridge.session_failed, self._on_session_failed),
            (bridge.transcript_delta, self._on_delta),
            (bridge.history_page, self._on_history_page),
            (bridge.history_loading, self._on_history_loading),
            (bridge.input_enabled, self._on_input_enabled),
            (bridge.approval_asked, self._forward_approval),
            (bridge.question_asked, self._forward_question),
            (bridge.generation_started, self._on_generation_started),
            (bridge.generation_finished, self._on_generation_finished),
        ):
            with suppress(RuntimeError, TypeError):
                signal.disconnect(slot)

    def _forward_approval(self, ask: object) -> None:
        self.approval_asked.emit(ask)

    def _forward_question(self, ask: object) -> None:
        self.question_asked.emit(ask)

    def keyPressEvent(self, event: object) -> None:
        from PySide6.QtGui import QKeyEvent

        if not isinstance(event, QKeyEvent):
            super().keyPressEvent(event)  # type: ignore[arg-type]
            return
        if event.key() == Qt.Key.Key_Escape:
            self.leave_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F4:
            self.open_settings.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F2:
            self.focus_prompt()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F3:
            self.focus_history_turn()
            event.accept()
            return
        if event.key() == Qt.Key.Key_G and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.cancel_prompt()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Up and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.load_older_history()
            event.accept()
            return
        if event.key() == Qt.Key.Key_End and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.jump_to_latest()
            event.accept()
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    @property
    def busy(self) -> bool:
        return self.bridge.busy

    @property
    def session_id(self) -> str | None:
        return self.bridge.session_id

    @property
    def prompt_enabled(self) -> bool:
        return self.prompt.isEnabled()

    def set_pending_config(self, label: str) -> None:
        self._pending_config_label = label
        self._set_status(self._session_label)

    def focus_prompt(self) -> None:
        if not self.busy:
            self.prompt.setFocus()

    def focus_history_turn(self) -> None:
        if self._turn_input.isEnabled():
            self._turn_input.setFocus()

    def cancel_prompt(self) -> None:
        if self.busy:
            self.bridge.cancel_prompt()
            self._set_status(f"{self._session_label} · cancelling…")
        else:
            self.focus_prompt()

    def load_older_history(self) -> None:
        self.bridge.load_older(self._display_turn())

    def load_newer_history(self) -> None:
        self.bridge.load_newer(self._display_turn())

    def jump_to_latest(self) -> None:
        self.transcript.jump_to_latest()
        self.bridge.jump_to_latest()

    def jump_to_history_turn(self, turn: int) -> None:
        self.bridge.jump_to_turn(turn)

    def _submit_prompt(self, text: str) -> None:
        stripped = text.strip()
        if not stripped or self.session_id is None or self.busy:
            return
        self.prompt.clear()
        if stripped.startswith("/"):
            if stripped.partition(" ")[0] == "/quit":
                self.leave_requested.emit()
                return
            self.bridge.run_command(stripped)
            return
        self.transcript.append_block("user", stripped)
        self.bridge.run_prompt(stripped)

    def _submit_turn(self) -> None:
        value = self._turn_input.text().strip()
        if not value:
            return
        try:
            turn = int(value)
        except ValueError:
            self.notify.emit("Enter a numeric turn", "warning", "")
            return
        if turn < 1 or turn > self._history_total:
            self.notify.emit(
                f"Turn must be between 1 and {self._history_total}",
                "warning",
                "",
            )
            return
        self.jump_to_history_turn(turn)

    def _on_session_opened(self, session_id: str, status: str, epoch: int) -> None:
        self._epoch = epoch
        self._session_label = f"session {session_id}"
        self._context_text = status
        self._set_status(self._session_label)
        self._context.setText(status)

    def _on_session_failed(self, message: str, epoch: int) -> None:
        self._epoch = epoch
        self._append_delta(TranscriptDelta("error", message, epoch=epoch))
        self._set_status("session unavailable")

    def _on_delta(self, delta: object) -> None:
        if not isinstance(delta, TranscriptDelta) or delta.epoch != self.bridge.epoch:
            return
        self._append_delta(delta)

    def _append_delta(self, delta: TranscriptDelta) -> None:
        if delta.kind == "status":
            self._context_text = delta.text
            self._context.setText(delta.text)
            return
        if delta.text == "__clear__":
            self.transcript.clear_messages()
            self._history_total = 0
            self._update_history_toolbar()
            return
        if delta.starts_stream:
            self.transcript.finish_stream()
        if delta.streaming:
            self.transcript.append_stream(
                delta.kind,
                delta.text,
                replace=delta.replaces_stream,
            )
        else:
            self.transcript.append_block(delta.kind, delta.text)

    def _on_history_page(self, page: object) -> None:
        if not isinstance(page, HistoryPage) or page.epoch != self.bridge.epoch:
            return
        if page.items:
            self.transcript.replace_history_blocks(list(page.items))
            self._history_total = page.total_turns
            if page.pin_latest:
                self.transcript.jump_to_latest()
            elif page.target_turn is not None:
                self.transcript.jump_to_turn(page.target_turn)
        elif page.pin_latest:
            self.transcript.jump_to_latest()
            self._history_total = page.total_turns
        else:
            self.transcript.mark_history_window()
            self._history_total = page.total_turns
        self._update_history_toolbar()

    def _on_history_loading(self, loading: bool, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self._history_loading = loading
        self._update_history_toolbar()

    def _on_generation_started(self, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        if self.session_id:
            self._set_status(f"session {self.session_id} · running")

    def _on_input_enabled(self, enabled: bool, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self.prompt.setEnabled(enabled)
        if enabled:
            self.prompt.setFocus()
        if enabled and self.session_id:
            self._set_status(f"session {self.session_id}")
            if self._context_text:
                self._context.setText(self._context_text)

    def _on_generation_finished(self, epoch: int) -> None:
        if epoch != self.bridge.epoch:
            return
        self.transcript.finish_stream()

    def _on_reached_top(self) -> None:
        if self._history_total <= 0 or self._history_loading:
            return
        self.bridge.prefetch_older()

    def _on_reached_bottom(self) -> None:
        if self._history_total <= 0 or self._history_loading:
            return
        self.bridge.prefetch_newer()

    def _display_turn(self) -> int:
        total = self._history_total
        if self.transcript.pinned_to_latest:
            return total
        viewport = self.transcript.viewport_turn()
        if viewport is not None:
            return viewport + 1
        return total

    def _set_status(self, text: str) -> None:
        if self._pending_config_label:
            text += f" · next: {self._pending_config_label}"
        self._status.setText(text)

    def _update_history_toolbar(self) -> None:
        if self._history_loading:
            self._history_info.setText("History · loading…")
        if self._history_total <= 0:
            if not self._history_loading:
                self._history_info.setText("History · no turns yet")
            self._older.setEnabled(False)
            self._turn_input.setEnabled(False)
            self._newer.setEnabled(False)
            self._latest.setEnabled(False)
            return
        current = self._display_turn()
        if not self._history_loading:
            self._history_info.setText(f"History · Turn {current} of {self._history_total}")
        self._older.setEnabled(not self._history_loading and current > 1)
        self._turn_input.setPlaceholderText(f"Turn 1-{self._history_total}")
        self._turn_input.setEnabled(not self._history_loading)
        self._newer.setEnabled(not self._history_loading and current < self._history_total)
        self._latest.setEnabled(
            not self._history_loading
            and not (
                current >= self._history_total and self.transcript.pinned_to_latest
            )
        )

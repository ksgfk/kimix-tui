"""Auto-growing chat composer: Enter sends, Ctrl/Shift+Enter inserts a newline."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy, QWidget


class Composer(QPlainTextEdit):
    """Chat input matching the previous PromptInput key bindings."""

    submitted = Signal(str)
    MIN_HEIGHT = 72
    MAX_HEIGHT = 192

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("prompt")
        self.setPlaceholderText(placeholder)
        self.setTabChangesFocus(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.MIN_HEIGHT)
        self.textChanged.connect(self._sync_height)

    @property
    def text(self) -> str:
        return self.toPlainText()

    @text.setter
    def text(self, value: str) -> None:
        self.setPlainText(value)

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._sync_height()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
            ):
                self.insertPlainText("\n")
                self._sync_height()
                return
            self.submitted.emit(self.toPlainText())
            return
        super().keyPressEvent(event)

    def _sync_height(self) -> None:
        lines = max(1, self.toPlainText().count("\n") + 1)
        line_height = max(18, self.fontMetrics().lineSpacing())
        document_height = lines * line_height + 16
        target = min(self.MAX_HEIGHT, max(self.MIN_HEIGHT, document_height))
        if target != self.height():
            self.setFixedHeight(target)

    def sizeHint(self) -> QSize:
        return QSize(200, self.height())

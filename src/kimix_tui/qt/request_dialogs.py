"""Modal dialogs for approvals, questions, and session deletion."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ApprovalDialog(QDialog):
    """Resolve a SDK approval or hook request."""

    decided = Signal(str)

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("approval-dialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 280)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setObjectName("dialog-title")
        heading.setWordWrap(True)
        body = QTextEdit()
        body.setObjectName("dialog-body")
        body.setReadOnly(True)
        body.setPlainText(description)
        layout.addWidget(heading)
        layout.addWidget(body)
        actions = QHBoxLayout()
        actions.addStretch()
        reject = QPushButton("Reject")
        reject.setObjectName("reject")
        session = QPushButton("Approve session")
        session.setObjectName("approve-for-session")
        approve = QPushButton("Approve")
        approve.setObjectName("approve")
        actions.addWidget(reject)
        actions.addWidget(session)
        actions.addWidget(approve)
        layout.addLayout(actions)
        reject.clicked.connect(lambda: self._choose("reject"))
        session.clicked.connect(lambda: self._choose("approve_for_session"))
        approve.clicked.connect(lambda: self._choose("approve"))
        QShortcut(QKeySequence("A"), self, lambda: self._choose("approve"))
        QShortcut(QKeySequence("S"), self, lambda: self._choose("approve_for_session"))
        QShortcut(QKeySequence("R"), self, lambda: self._choose("reject"))
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, lambda: self._choose("reject"))
        approve.setDefault(True)

    def _choose(self, decision: str) -> None:
        self.decided.emit(decision)
        self.accept()


class QuestionDialog(QDialog):
    """Collect a free-form answer for a public SDK question request."""

    answered = Signal(object)

    def __init__(self, prompt: str, body: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("question-dialog")
        self.setWindowTitle("Question")
        self.setModal(True)
        self.resize(520, 240)
        layout = QVBoxLayout(self)
        heading = QLabel(prompt)
        heading.setObjectName("dialog-title")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        if body:
            detail = QLabel(body)
            detail.setObjectName("dialog-body")
            detail.setWordWrap(True)
            layout.addWidget(detail)
        self._answer = QLineEdit()
        self._answer.setObjectName("answer")
        self._answer.setPlaceholderText("Type an option label or a free-form answer")
        layout.addWidget(self._answer)
        self._resolved = False
        self._answer.returnPressed.connect(self._submit)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.reject)
        self._answer.setFocus()

    def _submit(self) -> None:
        text = self._answer.text().strip()
        if not text:
            return
        self._emit_answer(text)
        self.accept()

    def reject(self) -> None:  # type: ignore[override]
        self._emit_answer(None)
        super().reject()

    def _emit_answer(self, value: object) -> None:
        if self._resolved:
            return
        self._resolved = True
        self.answered.emit(value)


class DeleteSessionsDialog(QDialog):
    """Confirm permanent deletion of one or more sessions."""

    def __init__(self, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("delete-dialog")
        self.setModal(True)
        noun = "session" if count == 1 else "sessions"
        self.setWindowTitle(f"Delete {count} {noun}?")
        layout = QVBoxLayout(self)
        title = QLabel(f"Delete {count} {noun}?")
        title.setObjectName("delete-title")
        copy = QLabel("Conversation history and session files will be permanently removed.")
        copy.setObjectName("delete-copy")
        copy.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(copy)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel-delete")
        confirm = QPushButton("Delete")
        confirm.setObjectName("confirm-delete")
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        layout.addLayout(actions)
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.reject)

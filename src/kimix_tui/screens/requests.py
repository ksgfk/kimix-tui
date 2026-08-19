"""Modal screens that resolve SDK interaction requests."""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, Static


class RequestModalScreen[ScreenResult](ModalScreen[ScreenResult]):
    """Shared presentation for blocking SDK request screens."""

    DEFAULT_CSS = """
    RequestModalScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }

    #decision-dialog, #question-dialog {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }

    .dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .dialog-body { margin-bottom: 1; }
    """


class ApprovalScreen(RequestModalScreen[str]):
    """Resolve a SDK approval or hook request without mouse interaction."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("a", "choose('approve')", "Approve"),
        Binding("s", "choose('approve_for_session')", "Approve session"),
        Binding("r,escape", "choose('reject')", "Reject"),
    ]

    def __init__(self, title: str, description: str) -> None:
        super().__init__()
        self._title = title
        self._description = description

    def compose(self) -> ComposeResult:
        with Vertical(id="decision-dialog"):
            yield Label(self._title, classes="dialog-title")
            yield Static(self._description, classes="dialog-body")
            yield Static("[a] approve   [s] approve for session   [r/Esc] reject")

    def action_choose(self, decision: str) -> None:
        self.dismiss(decision)


class QuestionScreen(RequestModalScreen[str | None]):
    """Collect a free-form answer for a public SDK question request."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, question: object) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        prompt = str(getattr(self._question, "question", "Question"))
        options = getattr(self._question, "options", [])
        option_lines = [
            f"- {getattr(option, 'label', option)}"
            + (
                f": {getattr(option, 'description', '')}"
                if getattr(option, "description", "")
                else ""
            )
            for option in options
        ]
        with Vertical(id="question-dialog"):
            yield Label(prompt, classes="dialog-title")
            if option_lines:
                yield Static("\n".join(option_lines), classes="dialog-body")
            yield Input(placeholder="Type an option label or a free-form answer", id="answer")

    def on_mount(self) -> None:
        self.query_one("#answer", Input).focus()

    @on(Input.Submitted, "#answer")
    def submit_answer(self, event: Input.Submitted) -> None:
        answer = event.value.strip()
        if answer:
            self.dismiss(answer)

    def action_cancel(self) -> None:
        self.dismiss(None)

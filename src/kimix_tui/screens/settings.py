"""Modal screen for selecting and inspecting Kimix provider JSON files."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from kimix_tui.llm_config import LLMConfigError, LLMConfigReference, inspect_llm_config

ConfigInspector = Callable[[Path], LLMConfigReference]
ConfigRegistrar = Callable[[LLMConfigReference], None]
ConfigRemover = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class LLMSettingsResult:
    """Validated configuration selected in the settings modal."""

    reference: LLMConfigReference
    use_project_default: bool = False


class OpenLLMSettings(Message):
    """Request the application to configure a default or saved session."""

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__()
        self.session_id = session_id


class ConfigPathItem(ListItem):
    """A configuration saved in the library."""

    def __init__(self, reference: LLMConfigReference) -> None:
        super().__init__()
        self.reference = reference

    def compose(self) -> ComposeResult:
        name, status = self._content()
        yield Label(name, classes="config-name", markup=False)
        yield Static(status, classes="config-summary", markup=False)

    def update_reference(self, reference: LLMConfigReference) -> None:
        self.reference = reference
        name, status = self._content()
        self.query_one(".config-name", Label).update(name)
        self.query_one(".config-summary", Static).update(status)

    def _content(self) -> tuple[str, str]:
        status = (
            f"Missing · {self.reference.provider_type} · {self.reference.model_name}"
            if self.reference.error
            else f"{self.reference.provider_type} · {self.reference.model_name}"
        )
        return self.reference.label, status


class ProjectDefaultItem(ListItem):
    """Select the work directory default without creating a session override."""

    def __init__(self, reference: LLMConfigReference) -> None:
        super().__init__()
        self.reference = reference

    def compose(self) -> ComposeResult:
        status = (
            f"Missing · {self.reference.label}"
            if self.reference.error
            else f"{self.reference.label} · {self.reference.provider_type}"
        )
        yield Label("Project default", classes="config-name", markup=False)
        yield Static(status, classes="config-summary", markup=False)


class LLMSettingsScreen(ModalScreen[LLMSettingsResult | None]):
    """Choose a config path and preview its redacted LLM settings."""

    CSS = """
    LLMSettingsScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }

    #settings-dialog {
        width: 112;
        max-width: 96%;
        height: 33;
        max-height: 94%;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    #settings-header {
        width: 100%;
        height: 2;
        border-bottom: tall $panel;
    }

    #settings-title {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    #settings-scope {
        width: auto;
        height: 1;
        color: $text-muted;
        text-align: right;
    }

    #config-path-label {
        width: 100%;
        height: 1;
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }

    #config-path-row {
        width: 100%;
        height: 3;
    }

    #config-path {
        width: 1fr;
        height: 3;
    }

    #load-config {
        width: 12;
        min-width: 12;
        height: 3;
        margin-left: 1;
    }

    #settings-error {
        width: 100%;
        height: 1;
        color: $error;
        text-overflow: ellipsis;
        text-wrap: nowrap;
    }

    #settings-body {
        width: 100%;
        height: 1fr;
        margin-top: 1;
        layout: horizontal;
    }

    #config-sources {
        width: 34;
        min-width: 28;
        height: 100%;
        padding-right: 1;
        border-right: tall $panel;
    }

    #config-sources-title {
        height: 1;
        margin-bottom: 1;
        color: $text-muted;
        text-style: bold;
    }

    #config-list {
        height: 1fr;
        background: $surface;
        scrollbar-size: 1 1;
    }

    #config-list > ConfigPathItem,
    #config-list > ProjectDefaultItem {
        height: 2;
        padding: 0 1;
        color: $text-muted;
    }

    #config-list > ConfigPathItem:hover,
    #config-list > ConfigPathItem.-highlight,
    #config-list > ProjectDefaultItem:hover,
    #config-list > ProjectDefaultItem.-highlight {
        background: $boost;
        color: $text;
    }

    #config-list > ConfigPathItem.-highlight,
    #config-list > ProjectDefaultItem.-highlight {
        padding-left: 0;
        border-left: thick $accent;
    }

    .config-name, .config-summary {
        width: 100%;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    .config-name { text-style: bold; }
    .config-summary { color: $text-muted; }

    #config-details {
        width: 1fr;
        height: 100%;
        padding-left: 2;
        scrollbar-size: 1 1;
    }

    #config-details-title {
        height: 1;
        margin-bottom: 1;
        color: $accent;
        text-style: bold;
    }

    .config-detail-row {
        width: 100%;
        height: 2;
    }

    .config-detail-key {
        width: 18;
        height: 1;
        color: $text-muted;
    }

    .config-detail-value {
        width: 1fr;
        height: 2;
        max-height: 2;
        text-overflow: ellipsis;
    }

    #settings-actions {
        width: 100%;
        height: 3;
        align: right middle;
        margin-top: 1;
    }

    #delete-config, #cancel-settings, #apply-settings {
        width: 14;
        min-width: 14;
        height: 3;
        margin-left: 1;
    }

    LLMSettingsScreen.-narrow #settings-dialog {
        height: 94%;
    }

    LLMSettingsScreen.-narrow #settings-body {
        layout: vertical;
    }

    LLMSettingsScreen.-narrow #config-sources {
        width: 100%;
        min-width: 0;
        height: 8;
        padding-right: 0;
        padding-bottom: 1;
        border-right: none;
        border-bottom: tall $panel;
    }

    LLMSettingsScreen.-narrow #config-details {
        width: 100%;
        height: 1fr;
        padding: 1 0 0 0;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    _DETAIL_FIELDS: ClassVar[dict[str, Callable[[LLMConfigReference], str]]] = {
        "#config-model": lambda item: item.label,
        "#config-format": lambda item: item.file_format,
        "#config-model-id": lambda item: item.model_name,
        "#config-provider": lambda item: item.provider_type,
        "#config-endpoint": lambda item: item.base_url,
        "#config-credential": lambda item: item.credential,
        "#config-context": lambda item: _format_tokens(item.max_context_size),
        "#config-output": lambda item: _format_tokens(item.max_tokens),
        "#config-capabilities": lambda item: ", ".join(item.capabilities) or "Not specified",
        "#config-thinking": lambda item: (
            f"effort {item.thinking_effort or 'not specified'} · "
            f"stream {'on' if item.show_thinking_stream else 'off'}"
        ),
    }

    def __init__(
        self,
        *,
        current: LLMConfigReference,
        references: Iterable[LLMConfigReference],
        scope_label: str,
        project_default: LLMConfigReference | None = None,
        inherits_project_default: bool = False,
        manage_library: bool = False,
        inspector: ConfigInspector = inspect_llm_config,
        registrar: ConfigRegistrar | None = None,
        remover: ConfigRemover | None = None,
    ) -> None:
        super().__init__()
        self._current = current
        self._references: dict[Path, LLMConfigReference] = {}
        for reference in [*references, current]:
            self._references[reference.path.resolve(strict=False)] = reference
        self._scope_label = scope_label
        self._project_default = project_default
        self._inherits_project_default = inherits_project_default
        self._manage_library = manage_library
        self._inspector = inspector
        self._registrar = registrar
        self._remover = remover
        self._preview: LLMConfigReference | None = None
        self._selected_reference: LLMConfigReference | None = None
        self._use_project_default = False

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-dialog"):
            with Horizontal(id="settings-header"):
                yield Label("LLM configuration", id="settings-title")
                yield Static(self._scope_label, id="settings-scope", markup=False)
            if self._manage_library:
                yield Static("KIMIX PROVIDER CONFIG (.JSON)", id="config-path-label")
                with Horizontal(id="config-path-row"):
                    yield Input(
                        str(self._current.path),
                        placeholder=r"C:\path\to\provider.json",
                        id="config-path",
                    )
                    yield Button("Add config", id="load-config")
            yield Static("", id="settings-error", markup=False)
            with Horizontal(id="settings-body"):
                with Vertical(id="config-sources"):
                    yield Static("AVAILABLE CONFIGS", id="config-sources-title")
                    yield ListView(
                        *(
                            [ProjectDefaultItem(self._project_default)]
                            if self._project_default is not None
                            else []
                        ),
                        *[ConfigPathItem(reference) for reference in self._references.values()],
                        id="config-list",
                    )
                with VerticalScroll(id="config-details"):
                    yield Static("CONFIG DETAILS", id="config-details-title")
                    yield from self._detail_row("Name", "config-model")
                    yield from self._detail_row("Format", "config-format")
                    yield from self._detail_row("Model ID", "config-model-id")
                    yield from self._detail_row("Provider", "config-provider")
                    yield from self._detail_row("Endpoint", "config-endpoint")
                    yield from self._detail_row("Credential", "config-credential")
                    yield from self._detail_row("Context", "config-context")
                    yield from self._detail_row("Max output", "config-output")
                    yield from self._detail_row("Capabilities", "config-capabilities")
                    yield from self._detail_row("Thinking", "config-thinking")
            with Horizontal(id="settings-actions"):
                if self._manage_library:
                    yield Button("Remove", id="delete-config", disabled=True)
                yield Button("Cancel", id="cancel-settings")
                yield Button("Use config", id="apply-settings", variant="primary", disabled=True)

    @staticmethod
    def _detail_row(label: str, value_id: str) -> Iterable[Horizontal]:
        yield Horizontal(
            Static(label, classes="config-detail-key"),
            Static("-", id=value_id, classes="config-detail-value", markup=False),
            classes="config-detail-row",
        )

    def on_mount(self) -> None:
        list_view = self.query_one("#config-list", ListView)
        if self._project_default is not None and self._inherits_project_default:
            list_view.index = 0
            self._select_project_default(self._project_default)
        else:
            paths = list(self._references)
            offset = 1 if self._project_default is not None else 0
            list_view.index = paths.index(self._current.path.resolve(strict=False)) + offset
            self._show_reference(self._current, error=self._current.error)
        list_view.focus()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 88, "-narrow")

    @on(ListView.Highlighted, "#config-list")
    def highlight_path(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, ConfigPathItem):
            self._select_reference(event.item.reference)
        elif isinstance(event.item, ProjectDefaultItem):
            self._select_project_default(event.item.reference)

    @on(ListView.Selected, "#config-list")
    def select_path(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ConfigPathItem):
            self._select_reference(event.item.reference)
        elif isinstance(event.item, ProjectDefaultItem):
            self._select_project_default(event.item.reference)

    @on(Input.Submitted, "#config-path")
    def submit_path(self, event: Input.Submitted) -> None:
        self._add_path(Path(event.value))

    @on(Button.Pressed, "#load-config")
    def press_load(self) -> None:
        self._add_path(Path(self.query_one("#config-path", Input).value))

    @on(Button.Pressed, "#apply-settings")
    def press_apply(self) -> None:
        if self._preview is not None:
            self.dismiss(LLMSettingsResult(self._preview, self._use_project_default))

    @on(Button.Pressed, "#delete-config")
    async def press_delete_config(self) -> None:
        reference = self._selected_reference
        if (
            not self._manage_library
            or reference is None
            or reference.path.resolve(strict=False) == self._current.path.resolve(strict=False)
        ):
            return
        try:
            if self._remover is not None:
                self._remover(reference.path)
        except OSError as exc:
            self.query_one("#settings-error", Static).update(
                f"Failed to remove configuration reference: {exc}"
            )
            return
        key = reference.path.resolve(strict=False)
        self._references.pop(key, None)
        for item in self.query(ConfigPathItem):
            if item.reference.path.resolve(strict=False) == key:
                await item.remove()
                break
        current_path = self._current.path.resolve(strict=False)
        remaining = list(self.query(ConfigPathItem))
        for index, item in enumerate(remaining):
            if item.reference.path.resolve(strict=False) == current_path:
                self.query_one("#config-list", ListView).index = index
                break
        self._select_reference(self._current)

    @on(Button.Pressed, "#cancel-settings")
    def press_cancel(self) -> None:
        self.action_cancel()

    def _add_path(self, path: Path) -> None:
        try:
            reference = self._inspector(path)
        except LLMConfigError as exc:
            self._preview = None
            self.query_one("#settings-error", Static).update(str(exc))
            self.query_one("#apply-settings", Button).disabled = True
            return
        try:
            if self._registrar is not None:
                self._registrar(reference)
        except OSError as exc:
            self._preview = None
            self.query_one("#settings-error", Static).update(
                f"Failed to save configuration metadata: {exc}"
            )
            self.query_one("#apply-settings", Button).disabled = True
            return
        key = reference.path.resolve(strict=False)
        if key not in self._references:
            self._references[key] = reference
            self.query_one("#config-list", ListView).append(ConfigPathItem(reference))
        else:
            self._references[key] = reference
            for item in self.query(ConfigPathItem):
                if item.reference.path.resolve(strict=False) == key:
                    item.update_reference(reference)
                    break
        self.query_one("#config-path", Input).value = str(reference.path)
        self._show_reference(reference)

    def _select_reference(self, reference: LLMConfigReference) -> None:
        self._update_path_input(reference.path)
        self._show_reference(reference, error=reference.error)

    def _select_project_default(self, reference: LLMConfigReference) -> None:
        self._update_path_input(reference.path)
        self._show_reference(reference, error=reference.error, use_project_default=True)

    def _update_path_input(self, path: Path) -> None:
        path_inputs = list(self.query("#config-path"))
        if path_inputs and isinstance(path_inputs[0], Input):
            path_inputs[0].value = str(path)

    def _show_reference(
        self,
        reference: LLMConfigReference,
        *,
        error: str | None = None,
        use_project_default: bool = False,
    ) -> None:
        self._selected_reference = reference
        self._preview = reference if error is None else None
        self._use_project_default = use_project_default and error is None
        self.query_one("#settings-error", Static).update(error or "")
        for selector, value in self._DETAIL_FIELDS.items():
            self.query_one(selector, Static).update(value(reference))
        self.query_one("#apply-settings", Button).disabled = error is not None
        if self._manage_library:
            self.query_one("#delete-config", Button).disabled = (
                use_project_default
                or reference.path.resolve(strict=False)
                == self._current.path.resolve(strict=False)
            )

    def action_cancel(self) -> None:
        self.dismiss(None)


def _format_tokens(value: int | None) -> str:
    return f"{value:,} tokens" if value is not None else "Not specified"

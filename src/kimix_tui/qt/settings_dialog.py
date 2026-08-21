"""LLM configuration dialog: library on the left, redacted details on the right."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kimix_tui.llm_config import LLMConfigError, LLMConfigReference, inspect_llm_config

ConfigInspector = Callable[[Path], LLMConfigReference]
ConfigRegistrar = Callable[[LLMConfigReference], None]
ConfigRemover = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class LLMSettingsResult:
    reference: LLMConfigReference
    use_project_default: bool = False


class ConfigListItem(QListWidgetItem):
    def __init__(self, reference: LLMConfigReference, *, project_default: bool = False) -> None:
        super().__init__()
        self.reference = reference
        self.project_default = project_default
        self._refresh()

    def update_reference(self, reference: LLMConfigReference) -> None:
        self.reference = reference
        self._refresh()

    def _refresh(self) -> None:
        if self.project_default:
            name = "Project default"
            status = (
                f"Missing · {self.reference.label}"
                if self.reference.error
                else f"{self.reference.label} · {self.reference.provider_type}"
            )
        else:
            name = self.reference.label
            status = (
                f"Missing · {self.reference.provider_type} · {self.reference.model_name}"
                if self.reference.error
                else f"{self.reference.provider_type} · {self.reference.model_name}"
            )
        self.setText(f"{name}\n{status}")
        self.setSizeHint(QSize(240, 48))


class LLMSettingsDialog(QDialog):
    """Choose a config path and preview its redacted LLM settings."""

    applied = Signal(object)

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settings-dialog")
        self.setWindowTitle("LLM configuration")
        self.setModal(True)
        self.resize(920, 560)
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
        self._narrow = False
        self._build()
        self._select_initial()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("LLM configuration")
        title.setObjectName("settings-title")
        scope = QLabel(self._scope_label)
        scope.setObjectName("settings-scope")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(scope)
        root.addLayout(header)

        if self._manage_library:
            path_label = QLabel("KIMIX PROVIDER CONFIG (.JSON)")
            path_label.setObjectName("config-path-label")
            root.addWidget(path_label)
            path_row = QHBoxLayout()
            self._path_input = QLineEdit(str(self._current.path))
            self._path_input.setObjectName("config-path")
            self._path_input.setPlaceholderText(r"C:\path\to\provider.json")
            load = QPushButton("Add config")
            load.setObjectName("load-config")
            path_row.addWidget(self._path_input)
            path_row.addWidget(load)
            root.addLayout(path_row)
            self._path_input.returnPressed.connect(lambda: self._add_path(Path(self._path_input.text())))
            load.clicked.connect(lambda: self._add_path(Path(self._path_input.text())))
        else:
            self._path_input = None

        self._error = QLabel("")
        self._error.setObjectName("settings-error")
        root.addWidget(self._error)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("settings-body")
        sources = QWidget()
        sources.setObjectName("config-sources")
        sources_layout = QVBoxLayout(sources)
        sources_title = QLabel("AVAILABLE CONFIGS")
        sources_title.setObjectName("config-sources-title")
        self._list = QListWidget()
        self._list.setObjectName("config-list")
        sources_layout.addWidget(sources_title)
        sources_layout.addWidget(self._list)
        self._splitter.addWidget(sources)

        details = QWidget()
        details.setObjectName("config-details")
        details_layout = QVBoxLayout(details)
        details_title = QLabel("CONFIG DETAILS")
        details_title.setObjectName("config-details-title")
        details_layout.addWidget(details_title)
        form = QFormLayout()
        self._fields: dict[str, QLabel] = {}
        for key, label in (
            ("config-model", "Name"),
            ("config-format", "Format"),
            ("config-model-id", "Model ID"),
            ("config-provider", "Provider"),
            ("config-endpoint", "Endpoint"),
            ("config-credential", "Credential"),
            ("config-context", "Context"),
            ("config-output", "Max output"),
            ("config-capabilities", "Capabilities"),
            ("config-thinking", "Thinking"),
        ):
            value = QLabel("-")
            value.setObjectName(key)
            value.setWordWrap(True)
            self._fields[key] = value
            form.addRow(label, value)
        details_layout.addLayout(form)
        details_layout.addStretch()
        self._splitter.addWidget(details)
        self._splitter.setSizes([320, 560])
        root.addWidget(self._splitter, 1)

        actions_row = QWidget()
        actions_row.setObjectName("settings-actions")
        actions = QHBoxLayout(actions_row)
        actions.addStretch()
        self._delete = None
        if self._manage_library:
            self._delete = QPushButton("Remove")
            self._delete.setObjectName("delete-config")
            self._delete.setEnabled(False)
            self._delete.clicked.connect(self._remove_selected)
            actions.addWidget(self._delete)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel-settings")
        apply_btn = QPushButton("Use config")
        apply_btn.setObjectName("apply-settings")
        apply_btn.setEnabled(False)
        self._apply = apply_btn
        actions.addWidget(cancel)
        actions.addWidget(apply_btn)
        root.addWidget(actions_row)

        self._populate_list()
        self._list.currentItemChanged.connect(self._on_current_changed)
        apply_btn.clicked.connect(self._apply_clicked)
        cancel.clicked.connect(self.reject)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.reject)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        narrow = self.width() < 880
        if narrow == self._narrow:
            return
        self._narrow = narrow
        self._splitter.setOrientation(
            Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        )

    def _populate_list(self) -> None:
        self._list.clear()
        if self._project_default is not None:
            self._list.addItem(ConfigListItem(self._project_default, project_default=True))
        for reference in self._references.values():
            self._list.addItem(ConfigListItem(reference))

    def _select_initial(self) -> None:
        if self._project_default is not None and self._inherits_project_default:
            self._list.setCurrentRow(0)
            self._select_project_default(self._project_default)
            return
        offset = 1 if self._project_default is not None else 0
        paths = list(self._references)
        try:
            index = paths.index(self._current.path.resolve(strict=False)) + offset
        except ValueError:
            index = offset
        self._list.setCurrentRow(index)
        self._show_reference(self._current, error=self._current.error)

    def _on_current_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if not isinstance(current, ConfigListItem):
            return
        if current.project_default:
            self._select_project_default(current.reference)
        else:
            self._select_reference(current.reference)

    def _select_reference(self, reference: LLMConfigReference) -> None:
        self._update_path_input(reference.path)
        self._show_reference(reference, error=reference.error)

    def _select_project_default(self, reference: LLMConfigReference) -> None:
        self._update_path_input(reference.path)
        self._show_reference(reference, error=reference.error, use_project_default=True)

    def _update_path_input(self, path: Path) -> None:
        if self._path_input is not None:
            self._path_input.setText(str(path))

    def _add_path(self, path: Path) -> None:
        try:
            reference = self._inspector(path)
        except LLMConfigError as exc:
            self._preview = None
            self._error.setText(str(exc))
            self._apply.setEnabled(False)
            return
        try:
            if self._registrar is not None:
                self._registrar(reference)
        except OSError as exc:
            self._preview = None
            self._error.setText(f"Failed to save configuration metadata: {exc}")
            self._apply.setEnabled(False)
            return
        key = reference.path.resolve(strict=False)
        if key not in self._references:
            self._references[key] = reference
            self._list.addItem(ConfigListItem(reference))
        else:
            self._references[key] = reference
            for row in range(self._list.count()):
                item = self._list.item(row)
                if (
                    isinstance(item, ConfigListItem)
                    and not item.project_default
                    and item.reference.path.resolve(strict=False) == key
                ):
                    item.update_reference(reference)
                    break
        self._update_path_input(reference.path)
        self._show_reference(reference)

    def _remove_selected(self) -> None:
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
            self._error.setText(f"Failed to remove configuration reference: {exc}")
            return
        key = reference.path.resolve(strict=False)
        self._references.pop(key, None)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if (
                isinstance(item, ConfigListItem)
                and not item.project_default
                and item.reference.path.resolve(strict=False) == key
            ):
                self._list.takeItem(row)
                break
        current_path = self._current.path.resolve(strict=False)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if (
                isinstance(item, ConfigListItem)
                and not item.project_default
                and item.reference.path.resolve(strict=False) == current_path
            ):
                self._list.setCurrentRow(row)
                break
        self._select_reference(self._current)

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
        self._error.setText(error or "")
        values = {
            "config-model": reference.label,
            "config-format": reference.file_format,
            "config-model-id": reference.model_name,
            "config-provider": reference.provider_type,
            "config-endpoint": reference.base_url,
            "config-credential": reference.credential,
            "config-context": _format_tokens(reference.max_context_size),
            "config-output": _format_tokens(reference.max_tokens),
            "config-capabilities": ", ".join(reference.capabilities) or "Not specified",
            "config-thinking": (
                f"effort {reference.thinking_effort or 'not specified'} · "
                f"stream {'on' if reference.show_thinking_stream else 'off'}"
            ),
        }
        for key, value in values.items():
            self._fields[key].setText(value)
        self._apply.setEnabled(error is None)
        if self._delete is not None:
            self._delete.setEnabled(
                not use_project_default
                and reference.path.resolve(strict=False)
                != self._current.path.resolve(strict=False)
            )

    def _apply_clicked(self) -> None:
        if self._preview is None:
            return
        result = LLMSettingsResult(self._preview, self._use_project_default)
        self.applied.emit(result)
        self.accept()

    def config_items(self) -> list[ConfigListItem]:
        return [
            item
            for row in range(self._list.count())
            if isinstance((item := self._list.item(row)), ConfigListItem)
        ]


def _format_tokens(value: int | None) -> str:
    return f"{value:,} tokens" if value is not None else "Not specified"

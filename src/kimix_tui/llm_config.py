"""Per-session references to Kimix provider JSON files."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import kimix
import orjson
from kimi_agent_sdk import Config
from kimi_cli.metadata import WorkDirMeta
from kimi_cli.share import get_share_dir
from kimix.utils.config import (
    _create_config,
    _inherit_sub_provider_defaults,
    _pick_main_from_sub_providers,
)

from kimix_tui.kimi_workdir import resolve_kimi_work_dir


class LLMConfigError(ValueError):
    """Raised when an LLM configuration cannot be loaded safely."""


def default_config_path() -> Path:
    """Return Kimix's package-level default provider JSON path."""

    package_file = kimix.__file__
    if package_file is None:
        return Path("default_config.json").resolve(strict=False)
    return (Path(package_file).resolve().parent / "default_config.json").resolve(strict=False)


@dataclass(frozen=True, slots=True)
class LLMConfigReference:
    """A reloadable config path plus a redacted summary for display."""

    path: Path
    model_name: str
    provider_type: str
    base_url: str
    credential: str
    file_format: str = "Unknown"
    display_name: str | None = None
    model_override: str | None = None
    max_context_size: int | None = None
    max_tokens: int | None = None
    capabilities: tuple[str, ...] = ()
    thinking_effort: str | None = None
    show_thinking_stream: bool | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.model_name or self.path.name

def inspect_llm_config(
    config_file: Path,
    *,
    model_override: str | None = None,
) -> LLMConfigReference:
    """Validate a Kimix provider JSON and return a secret-free summary."""

    path, provider_dict = _load_kimix_provider_json(config_file)
    preview_dict = dict(provider_dict)
    preview_dict.pop("env", None)
    config = _build_sdk_config(preview_dict, path)
    model = config.model
    provider = config.provider
    assert model is not None
    assert provider is not None
    model_name = model_override or model.model
    profile_name = provider_dict.get("name")
    display_name = str(profile_name) if profile_name else model.display_name
    provider_type = str(provider.type)
    base_url = _redact_url(provider.base_url) if provider.base_url else "Not set in file"
    if provider_dict.get("oauth"):
        credential = "OAuth"
    elif provider_dict.get("api_key"):
        credential = "API key configured"
    elif _has_environment_credential(provider_dict):
        credential = "Environment"
    else:
        credential = "Not stored in file"

    return LLMConfigReference(
        path=path,
        model_name=model_name,
        provider_type=provider_type,
        base_url=base_url,
        credential=credential,
        file_format=_config_format(path),
        display_name=display_name,
        model_override=model_override,
        max_context_size=model.max_context_size,
        max_tokens=config.max_tokens,
        capabilities=tuple(sorted(str(item) for item in (model.capabilities or ()))),
        thinking_effort=(
            str(provider_dict["thinking_effort"])
            if provider_dict.get("thinking_effort") is not None
            else None
        ),
        show_thinking_stream=(
            bool(provider_dict["show_thinking_stream"])
            if provider_dict.get("show_thinking_stream") is not None
            else None
        ),
    )


def load_kimix_sdk_config(config_file: Path) -> Config:
    """Load a Kimix flat provider JSON as the SDK Config used for a session."""

    path, provider_dict = _load_kimix_provider_json(config_file)
    return _build_sdk_config(provider_dict, path)


def load_kimix_provider_dict(config_file: Path) -> dict[str, Any]:
    """Load the normalized provider mapping consumed by Kimix session factories."""

    _path, provider_dict = _load_kimix_provider_json(config_file)
    return provider_dict


def _load_kimix_provider_json(config_file: Path) -> tuple[Path, dict[str, Any]]:
    path = config_file.expanduser().resolve(strict=False)
    if path.suffix.lower() != ".json":
        raise LLMConfigError(f"Kimix configuration must be a JSON file: {path}")
    if not path.is_file():
        raise LLMConfigError(f"Configuration file does not exist: {path}")
    try:
        loaded = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise LLMConfigError(f"Invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise LLMConfigError(f"Kimix configuration must contain a JSON object: {path}")

    provider_dict = deepcopy(loaded)
    sub_provider = provider_dict.pop("sub_provider", None)
    sub_providers = provider_dict.pop("sub_providers", None)
    _inherit_sub_provider_defaults(provider_dict, sub_provider, sub_providers)
    _pick_main_from_sub_providers(provider_dict, sub_provider, sub_providers)
    return path, provider_dict


def _build_sdk_config(provider_dict: dict[str, Any], path: Path) -> Config:
    try:
        config, _normalized = _create_config(provider_dict)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise LLMConfigError(f"Invalid Kimix configuration {path}: {exc}") from exc
    return config


def _has_environment_credential(provider_dict: dict[str, Any]) -> bool:
    environment = provider_dict.get("env")
    if isinstance(environment, dict) and any(
        key in environment for key in ("KIMI_API_KEY", "KIMIX_API_KEY")
    ):
        return True
    return bool(os.environ.get("KIMI_API_KEY") or os.environ.get("KIMIX_API_KEY"))


def _redact_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "Configured endpoint"


def unavailable_config_reference(
    path: Path,
    *,
    model_override: str | None = None,
) -> LLMConfigReference:
    """Build a displayable reference when startup configuration is invalid."""

    return LLMConfigReference(
        path=path.expanduser().resolve(strict=False),
        model_name=model_override or "Configuration unavailable",
        provider_type="Unavailable",
        base_url="Unavailable",
        credential="Unavailable",
        file_format="JSON",
        model_override=model_override,
        error=f"Configuration file is unavailable: {path.expanduser().resolve(strict=False)}",
    )


def _config_format(path: Path) -> str:
    return "JSON" if path.suffix.lower() == ".json" else "Unsupported"


def config_file_available(reference: LLMConfigReference) -> bool:
    """Return whether a reference can be loaded without an external missing file."""

    return reference.error is None and reference.provider_type != "Unavailable" and reference.path.is_file()


SessionConfigFileResolver = Callable[[Path, str], Path]

STORE_FILENAME = "kimix-gui.json"


def default_store_file() -> Path:
    """Return the share-dir file that stores config library paths."""

    return get_share_dir() / STORE_FILENAME


def session_config_file(work_dir: Path, session_id: str) -> Path:
    """Return the config reference file inside a Kimi session."""

    if not session_id or Path(session_id).name != session_id:
        raise ValueError(f"Invalid session id: {session_id!r}")
    resolved = resolve_kimi_work_dir(work_dir)
    return WorkDirMeta(path=str(resolved)).sessions_dir / session_id / STORE_FILENAME


class LLMConfigStore:
    """Persist path-only global defaults and session-local config references."""

    VERSION = 3
    SESSION_VERSION = 1

    def __init__(
        self,
        metadata_file: Path | None = None,
        *,
        session_file_resolver: SessionConfigFileResolver = session_config_file,
    ) -> None:
        self.metadata_file = metadata_file or default_store_file()
        self._session_file_resolver = session_file_resolver
        self._data = self._load()

    def default_for(self, work_dir: Path) -> LLMConfigReference | None:
        entry = self._work_dir_entry(work_dir, create=False)
        return self._reference_from_value(entry.get("default")) if entry else None

    def set_default(self, work_dir: Path, reference: LLMConfigReference) -> None:
        self._add_config(reference)
        self._work_dir_entry(work_dir, create=True)["default"] = self._path_text(reference.path)
        self._save()

    def configs(self) -> list[LLMConfigReference]:
        configs = self._data.setdefault("configs", [])
        assert isinstance(configs, list)
        return [
            reference
            for value in configs
            if (reference := self._reference_from_value(value)) is not None
        ]

    def add_config(self, reference: LLMConfigReference) -> None:
        self._add_config(reference)
        self._save()

    def remove_config(self, path: Path) -> None:
        configs = self._data.setdefault("configs", [])
        assert isinstance(configs, list)
        path_text = self._path_text(path)
        if path_text in configs:
            configs.remove(path_text)
            self._save()

    def session_for(self, work_dir: Path, session_id: str) -> LLMConfigReference | None:
        metadata_file = self._session_file_resolver(work_dir, session_id)
        try:
            data = json.loads(metadata_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            return replace(
                unavailable_config_reference(metadata_file),
                error=f"Invalid session configuration reference {metadata_file}: {exc}",
            )
        config_value = (
            data.get("config")
            if isinstance(data, dict) and data.get("version") == self.SESSION_VERSION
            else None
        )
        reference = self._reference_from_value(config_value)
        if reference is not None:
            return reference
        return replace(
            unavailable_config_reference(metadata_file),
            error=f"Invalid session configuration reference: {metadata_file}",
        )

    def set_session(
        self,
        work_dir: Path,
        session_id: str,
        reference: LLMConfigReference,
    ) -> None:
        self._add_config(reference)
        self._write_session_path(work_dir, session_id, reference.path)
        self._save()

    def clear_session(self, work_dir: Path, session_id: str) -> None:
        try:
            self._session_file_resolver(work_dir, session_id).unlink()
        except FileNotFoundError:
            pass

    def references_for(self, work_dir: Path) -> list[LLMConfigReference]:
        entry = self._work_dir_entry(work_dir, create=False)
        references = self.configs()
        if entry:
            default = self._reference_from_value(entry.get("default"))
            if default is not None:
                references.append(default)
        default_path = default_config_path()
        if default_path.is_file():
            try:
                references.append(inspect_llm_config(default_path))
            except LLMConfigError:
                pass
        deduplicated: dict[Path, LLMConfigReference] = {}
        for reference in references:
            path = reference.path.expanduser().resolve(strict=False)
            deduplicated[path] = reference
        return list(deduplicated.values())

    def _add_config(self, reference: LLMConfigReference) -> None:
        configs = self._data.setdefault("configs", [])
        assert isinstance(configs, list)
        path = self._path_text(reference.path)
        if path not in configs:
            configs.append(path)

    @staticmethod
    def _path_text(path: Path) -> str:
        return str(path.expanduser().resolve(strict=False))

    @staticmethod
    def _path_from_value(value: object) -> Path | None:
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve(strict=False)
        return None

    @classmethod
    def _reference_from_value(cls, value: object) -> LLMConfigReference | None:
        path = cls._path_from_value(value)
        if path is None:
            return None
        try:
            return inspect_llm_config(path)
        except LLMConfigError as exc:
            return replace(unavailable_config_reference(path), error=str(exc))

    def _write_session_path(self, work_dir: Path, session_id: str, path: Path) -> None:
        self._write_json(
            self._session_file_resolver(work_dir, session_id),
            {"version": self.SESSION_VERSION, "config": self._path_text(path)},
        )

    def _work_dir_entry(self, work_dir: Path, *, create: bool) -> dict[str, Any]:
        work_dirs = self._data.setdefault("work_dirs", {})
        assert isinstance(work_dirs, dict)
        key = str(work_dir.expanduser().resolve(strict=False))
        if create:
            entry = work_dirs.setdefault(key, {})
            assert isinstance(entry, dict)
            return entry
        entry = work_dirs.get(key)
        return entry if isinstance(entry, dict) else {}

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            return self._empty_data()
        if not isinstance(data, dict) or data.get("version") != self.VERSION:
            return self._empty_data()
        if not isinstance(data.get("configs"), list):
            return self._empty_data()
        if not isinstance(data.get("work_dirs"), dict):
            return self._empty_data()
        return data

    def _empty_data(self) -> dict[str, Any]:
        return {"version": self.VERSION, "configs": [], "work_dirs": {}}

    def _save(self) -> None:
        self._write_json(self.metadata_file, self._data)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

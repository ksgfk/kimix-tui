from __future__ import annotations

import json
from pathlib import Path

import pytest

from kimix_tui.llm_config import (
    LLMConfigStore,
    config_file_available,
    default_store_file,
    inspect_llm_config,
    session_config_file,
    unavailable_config_reference,
)


def config_store(tmp_path: Path, metadata_file: Path | None = None) -> LLMConfigStore:
    return LLMConfigStore(
        metadata_file or (tmp_path / "metadata.json"),
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-gui.json"
        ),
    )


def write_llm_config(
    path: Path,
    *,
    model: str,
    display_name: str,
    provider: str = "openai_legacy",
    base_url: str = "https://user:password@example.com/v1?api_key=query-secret",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_name": model,
                "name": display_name,
                "model": model,
                "max_context_size": 131_072,
                "max_tokens": 8192,
                "capabilities": ["thinking"],
                "url": base_url,
                "type": provider,
                "api_key": "top-secret-key",
                "show_thinking_stream": True,
                "thinking_effort": "high",
            }
        ),
        encoding="utf-8",
    )
    return path


def write_json_config(path: Path, *, model: str, display_name: str) -> Path:
    return write_llm_config(
        path,
        model=model,
        display_name=display_name,
        provider="anthropic",
        base_url="https://api.anthropic.test/v1",
    )


def test_inspect_config_returns_redacted_details(tmp_path: Path) -> None:
    config_file = write_llm_config(
        tmp_path / "model.json",
        model="gpt-test",
        display_name="Test Model",
    )

    reference = inspect_llm_config(config_file)

    assert reference.path == config_file.resolve()
    assert reference.label == "Test Model"
    assert reference.model_name == "gpt-test"
    assert reference.provider_type == "openai_legacy"
    assert reference.base_url == "https://example.com/v1"
    assert reference.credential == "API key configured"
    assert reference.max_context_size == 131_072
    assert reference.max_tokens == 8192
    assert reference.capabilities == ("thinking",)
    assert reference.thinking_effort == "high"
    assert reference.show_thinking_stream is True
    assert reference.file_format == "JSON"


def test_inspect_json_config(tmp_path: Path) -> None:
    config_file = write_json_config(
        tmp_path / "model.json",
        model="claude-test",
        display_name="Claude Test",
    )

    reference = inspect_llm_config(config_file)

    assert reference.file_format == "JSON"
    assert reference.label == "Claude Test"
    assert reference.model_name == "claude-test"
    assert reference.provider_type == "anthropic"
    assert reference.max_context_size == 131_072
    assert reference.max_tokens == 8192


def test_store_persists_project_and_session_references_without_secrets(tmp_path: Path) -> None:
    config_file = write_llm_config(
        tmp_path / "model.json",
        model="gpt-test",
        display_name="Test Model",
    )
    reference = inspect_llm_config(config_file)
    metadata_file = tmp_path / "kimix-gui.json"
    work_dir = tmp_path / "project"
    store = config_store(tmp_path, metadata_file)

    store.set_default(work_dir, reference)
    store.set_session(work_dir, "session-1", reference)

    reloaded = config_store(tmp_path, metadata_file)
    saved_default = reloaded.default_for(work_dir)
    saved_session = reloaded.session_for(work_dir, "session-1")
    assert saved_default is not None
    assert saved_session is not None
    assert saved_default.path == reference.path
    assert saved_session.path == reference.path
    persisted = metadata_file.read_text(encoding="utf-8")
    assert "top-secret-key" not in persisted
    assert "password" not in persisted
    assert "query-secret" not in persisted
    data = json.loads(persisted)
    assert data == {
        "version": 3,
        "configs": [str(config_file.resolve())],
        "work_dirs": {str(work_dir.resolve()): {"default": str(config_file.resolve())}},
    }
    session_data = json.loads(
        (tmp_path / "sessions" / "session-1" / "kimix-gui.json").read_text(encoding="utf-8")
    )
    assert session_data == {"version": 1, "config": str(config_file.resolve())}


def test_missing_external_config_is_not_available(tmp_path: Path) -> None:
    reference = unavailable_config_reference(tmp_path / "missing.json")

    assert config_file_available(reference) is False


def test_session_config_file_lives_inside_kimi_session_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    path = session_config_file(tmp_path / "project", "session-1")

    assert path.name == "kimix-gui.json"
    assert path.parent.name == "session-1"
    assert path.parents[2].name == "sessions"


def test_default_store_file_uses_kimix_gui_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path / "share"))

    assert default_store_file() == tmp_path / "share" / "kimix-gui.json"


def test_config_library_is_shared_across_work_dirs(tmp_path: Path) -> None:
    first = inspect_llm_config(
        write_llm_config(tmp_path / "first.json", model="first", display_name="First")
    )
    second = inspect_llm_config(
        write_llm_config(tmp_path / "second.json", model="second", display_name="Second")
    )
    metadata_file = tmp_path / "metadata.json"
    store = config_store(tmp_path, metadata_file)

    store.add_config(first)
    store.add_config(second)

    paths = {
        reference.path
        for reference in config_store(tmp_path, metadata_file).references_for(tmp_path)
    }
    assert {first.path, second.path}.issubset(paths)

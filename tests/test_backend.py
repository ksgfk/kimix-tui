from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from kimix_tui import backend
from kimix_tui.backend import (
    SessionOptions,
    close_sdk_session,
    create_sdk_session,
    new_session_id,
)


def test_session_options_are_standalone_values(tmp_path: Path) -> None:
    options = SessionOptions(work_dir=tmp_path, model="kimi", thinking=True)

    assert options.work_dir == tmp_path
    assert options.model == "kimi"
    assert options.thinking is True
    assert options.yolo is False


def test_new_session_id_is_compact_and_unique() -> None:
    first = new_session_id()
    second = new_session_id()

    assert first.startswith("tui_")
    assert len(first) == 16
    assert first != second


@pytest.mark.asyncio
async def test_session_factory_passes_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 100_000,
                "type": "openai_legacy",
                "url": "https://example.test/v1",
                "api_key": "secret",
            }
        ),
        encoding="utf-8",
    )
    created = object()
    received: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(backend, "create_session_async", fake_create)

    result = await create_sdk_session(
        SessionOptions(tmp_path, config_file=config_file, model="override-model")
    )

    assert result is created
    provider_dict = received["provider_dict"]
    assert isinstance(provider_dict, dict)
    assert provider_dict["model"] == "override-model"
    assert provider_dict["url"] == "https://example.test/v1"
    assert received["resume"] is False
    assert received["model"] == "override-model"


@pytest.mark.asyncio
async def test_session_factory_resumes_through_kimix_worker_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = object()
    received: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> object:
        received.update(kwargs)
        return created

    monkeypatch.setattr(backend, "create_session_async", fake_create)

    result = await create_sdk_session(
        SessionOptions(
            tmp_path,
            session_id="existing-session",
            model="model-override",
            thinking=True,
            yolo=True,
        )
    )

    assert result is created
    assert received["session_id"] == "existing-session"
    assert received["resume"] is True
    assert received["model"] == "model-override"
    assert received["thinking"] is True
    assert received["yolo"] is True


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "vendor" / "KimiX" / "pyproject.toml").exists(),
    reason="vendor/KimiX submodule is not checked out",
)
async def test_session_factory_loads_worker_execution_tools(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 131_072,
                "capabilities": [],
                "type": "openai_legacy",
                "url": "http://127.0.0.1",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )

    session = await create_sdk_session(SessionOptions(tmp_path, config_file=config_file))
    try:
        runtime_session = cast(Any, session)
        tool_names = {
            tool.name for tool in runtime_session._cli.soul.agent.toolset.tools
        }

        assert "python" in tool_names
        assert "job_output" in tool_names
        assert {"bash", "pwsh"} & tool_names
        assert session.status.yolo_enabled is False
    finally:
        await close_sdk_session(session)

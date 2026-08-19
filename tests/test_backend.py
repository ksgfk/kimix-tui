from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from kimi_agent_sdk import Config

from kimix_tui import backend
from kimix_tui.backend import SessionOptions, create_sdk_session, new_session_id


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

    monkeypatch.setattr(backend.Session, "create", fake_create)

    result = await create_sdk_session(SessionOptions(tmp_path, config_file=config_file))

    assert result is created
    config = cast(Config, received["config"])
    assert isinstance(config, Config)
    assert config.model is not None
    assert config.provider is not None
    assert config.model.model == "test-model"
    assert config.provider.base_url == "https://example.test/v1"

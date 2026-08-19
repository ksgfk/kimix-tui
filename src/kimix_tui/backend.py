"""Thin adapter around the public :mod:`kimi_agent_sdk` session API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from kimi_agent_sdk import Session

from kimix_tui.kimi_workdir import resolve_kimi_work_dir
from kimix_tui.llm_config import load_kimix_sdk_config


class SdkSession(Protocol):
    """The public subset of ``kimi_agent_sdk.Session`` used by the UI."""

    @property
    def id(self) -> str: ...

    @property
    def status(self) -> Any: ...

    def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]: ...

    def cancel(self) -> None: ...

    async def clear(self, **custom_arguments: Any) -> None: ...

    async def compact(self, *, custom_instruction: str = "") -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionOptions:
    """Configuration for creating or resuming a public SDK session."""

    work_dir: Path
    session_id: str | None = None
    config_file: Path | None = None
    model: str | None = None
    thinking: bool = False
    yolo: bool = False


def new_session_id() -> str:
    """Return a compact persistent session id owned by this TUI."""

    return f"tui_{uuid4().hex[:12]}"


async def create_sdk_session(options: SessionOptions) -> SdkSession:
    """Create or resume a session without importing Kimix internals."""

    work_dir = resolve_kimi_work_dir(options.work_dir)
    config = load_kimix_sdk_config(options.config_file) if options.config_file else None
    common: dict[str, Any] = {
        "config": config,
        "model": options.model,
        "thinking": options.thinking,
        "yolo": options.yolo,
    }

    if options.session_id:
        resumed = await Session.resume(
            work_dir=work_dir,
            session_id=options.session_id,
            **common,
        )
        if resumed is not None:
            return resumed

    return await Session.create(
        work_dir=work_dir,
        session_id=options.session_id or new_session_id(),
        **common,
    )

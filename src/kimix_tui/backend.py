"""Thin adapter around the public :mod:`kimi_agent_sdk` session API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from kimix import close_session_async

try:
    from kimix import create_session_async
except ImportError:  # Local Kimix-CLI-X revisions export the worker factory privately.
    from kimix import base as _kimix_base
    from kimix.utils import _create_session_async

    async def create_session_async(**kwargs: Any) -> Any:
        kwargs.pop("model", None)
        thinking = bool(kwargs.pop("thinking", False))
        yolo = bool(kwargs.pop("yolo", False))
        previous_thinking = _kimix_base._default_thinking
        previous_yolo = _kimix_base._default_yolo
        _kimix_base._default_thinking = thinking
        _kimix_base._default_yolo = yolo
        try:
            return await _create_session_async(**kwargs)
        finally:
            _kimix_base._default_thinking = previous_thinking
            _kimix_base._default_yolo = previous_yolo

from kimix_tui.kimi_workdir import resolve_kimi_work_dir
from kimix_tui.llm_config import load_kimix_provider_dict


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
    """Create or resume a session through Kimix's public Worker factory."""

    work_dir = resolve_kimi_work_dir(options.work_dir)
    provider_dict = (
        load_kimix_provider_dict(options.config_file) if options.config_file else None
    )
    if provider_dict is not None and options.model is not None:
        provider_dict["model"] = options.model
    common: dict[str, Any] = {
        "provider_dict": provider_dict,
        "model": options.model,
        "thinking": options.thinking,
        "yolo": options.yolo,
    }

    return await create_session_async(
        work_dir=work_dir,
        session_id=options.session_id or new_session_id(),
        resume=options.session_id is not None,
        **common,
    )


async def close_sdk_session(session: SdkSession) -> None:
    """Close a Kimix-created session and remove it from the live-session registry."""

    await close_session_async(session)

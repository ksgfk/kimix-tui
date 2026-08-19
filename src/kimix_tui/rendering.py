"""Pure rendering decisions for public Kimi Agent SDK wire messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import orjson
from kimi_agent_sdk import (
    CompactionBegin,
    CompactionEnd,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolCallPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
)

RenderKind = Literal[
    "assistant",
    "thinking",
    "tool",
    "tool_result",
    "error",
    "system",
    "status",
]


@dataclass(frozen=True, slots=True)
class RenderEvent:
    """A framework-neutral display event consumed by the Textual layer."""

    kind: RenderKind
    text: str
    streaming: bool = False


DISPLAY_CHAR_LIMIT = 4_000


def truncate_display(text: str, limit: int = DISPLAY_CHAR_LIMIT) -> str:
    """Keep at most ``limit`` characters without retaining the dropped tail."""

    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n… output truncated ({omitted} more chars)"


def bounded_concat(existing: str, fragment: str, limit: int = DISPLAY_CHAR_LIMIT) -> str:
    """Append ``fragment`` without ever allocating ``existing + fragment`` past ``limit``."""

    if not fragment:
        return existing
    if not existing:
        return truncate_display(fragment, limit)
    if len(existing) >= limit:
        return existing
    room = limit - len(existing)
    if len(fragment) <= room:
        return existing + fragment
    omitted = len(fragment) - room
    return f"{existing}{fragment[:room]}\n… output truncated ({omitted} more chars)"


def _pretty_json(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError:
        return value
    return orjson.dumps(parsed, option=orjson.OPT_INDENT_2).decode("utf-8")


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, TextPart):
                parts.append(item.text)
            elif isinstance(item, ThinkPart):
                parts.append(item.think)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def user_input_text(user_input: object) -> str:
    """Normalize a TurnBegin/SteerInput payload into visible user text."""

    return _content_text(user_input).strip()


def _tool_result_text(message: ToolResult) -> str:
    result = message.return_value
    detail = result.brief or result.message or _content_text(result.output)
    return truncate_display(detail or "(no visible output)")


def render_wire_message(message: object) -> RenderEvent | None:
    """Map a public SDK wire message to a small, stable UI event."""

    if isinstance(message, TextPart):
        return RenderEvent("assistant", message.text, streaming=True)
    if isinstance(message, ThinkPart):
        return RenderEvent("thinking", message.think, streaming=True)
    if isinstance(message, ToolCall):
        arguments = _pretty_json(message.function.arguments)
        body = f"{message.function.name}"
        if arguments:
            body = f"{body}\n{truncate_display(arguments)}"
        return RenderEvent("tool", body)
    if isinstance(message, ToolCallPart):
        # Argument fragments are represented by the surrounding ToolCall block.
        return None
    if isinstance(message, ToolResult):
        kind: RenderKind = "error" if message.return_value.is_error else "tool_result"
        return RenderEvent(kind, _tool_result_text(message))
    if isinstance(message, StepBegin):
        return RenderEvent("system", f"Step {message.n}")
    if isinstance(message, StepInterrupted):
        return RenderEvent("error", "Current step was interrupted")
    if isinstance(message, CompactionBegin):
        return RenderEvent("system", "Compacting context…")
    if isinstance(message, CompactionEnd):
        suffix = (
            f" (~{message.estimated_token_count} tokens)"
            if message.estimated_token_count is not None
            else ""
        )
        return RenderEvent("system", f"Context compacted{suffix}")
    if isinstance(message, StatusUpdate):
        return RenderEvent("status", format_status(message))
    if isinstance(message, TurnBegin | TurnEnd):
        return None
    return None


def format_status(status: object) -> str:
    """Format either a ``StatusUpdate`` or a SDK status snapshot."""

    tokens = getattr(status, "context_tokens", None)
    maximum = getattr(status, "max_context_tokens", None)
    usage = getattr(status, "context_usage", None)

    fields: list[str] = []
    if tokens is not None:
        fields.append(f"context {tokens:,}")
        if maximum:
            fields[-1] += f"/{maximum:,}"
    if usage is not None:
        percentage = usage * 100 if 0 <= usage <= 1 else usage
        fields.append(f"{percentage:.1f}%")
    return " · ".join(fields) or "ready"

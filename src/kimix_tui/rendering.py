"""Pure rendering decisions for public Kimi Agent SDK wire messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal

import orjson
from kimi_agent_sdk import (
    ApprovalRequest,
    ApprovalResponse,
    AudioURLPart,
    BriefDisplayBlock,
    CompactionBegin,
    CompactionEnd,
    DiffDisplayBlock,
    DisplayBlock,
    ImageURLPart,
    ShellDisplayBlock,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    SubagentEvent,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    ToolCall,
    ToolCallPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
    UnknownDisplayBlock,
    VideoURLPart,
)

from kimix_tui.tool_display import format_tool_call_text, format_tool_result_text

RenderKind = Literal[
    "assistant",
    "thinking",
    "tool",
    "tool_result",
    "approval",
    "error",
    "system",
    "status",
]


@dataclass(frozen=True, slots=True)
class RenderEvent:
    """A framework-neutral display event consumed by the Qt UI layer."""

    kind: RenderKind
    text: str
    streaming: bool = False
    starts_stream: bool = False
    replaces_stream: bool = False


@dataclass(slots=True)
class RenderState:
    """Small amount of state required to render related wire events."""

    tool_names: dict[str, str] = field(default_factory=dict)
    tool_arguments: dict[str, str] = field(default_factory=dict)
    tool_extras: dict[str, object] = field(default_factory=dict)
    active_tool_call_id: str | None = None
    subagents: dict[str, RenderState] = field(default_factory=dict)
    scope_stream_kind: RenderKind | None = None
    status_values: dict[str, object] = field(default_factory=dict)


_OBSERVABILITY_EVENTS = {"LLMRequest", "LLMToolsSnapshot", "MCPToolsDiscovered"}


def _model_data(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    try:
        return model_dump(mode="json", exclude_none=True)
    except TypeError:
        return model_dump()


def _json_text(value: object) -> str:
    try:
        return orjson.dumps(
            _model_data(value),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=lambda item: str(item),
        ).decode("utf-8")
    except TypeError, ValueError:
        return str(value)


def _pretty_json(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError:
        return value
    return _json_text(parsed)


def _media_text(kind: str, media: object) -> str:
    url = str(getattr(media, "url", ""))
    media_id = getattr(media, "id", None)
    suffix = f"\nID: {media_id}" if media_id else ""
    return f"[{kind}]\nURL: {url}{suffix}"


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, TextPart):
        return value.text
    if isinstance(value, ThinkPart):
        return value.think
    if isinstance(value, ImageURLPart):
        return _media_text("Image", value.image_url)
    if isinstance(value, AudioURLPart):
        return _media_text("Audio", value.audio_url)
    if isinstance(value, VideoURLPart):
        return _media_text("Video", value.video_url)
    if isinstance(value, list | tuple):
        return "\n".join(filter(None, (_content_text(item) for item in value)))
    data = _model_data(value)
    return _json_text(data) if data is not value else str(value)


def user_input_text(user_input: object) -> str:
    """Normalize a TurnBegin/SteerInput payload into visible user text."""

    return _content_text(user_input).strip()


def format_display_blocks(display: Sequence[object]) -> str:
    """Render the structured display blocks used by native Kimix tools."""

    parts: list[str] = []
    for block in display:
        if isinstance(block, BriefDisplayBlock):
            if block.text:
                parts.append(block.text)
            continue
        if isinstance(block, DiffDisplayBlock):
            summary = " · summary" if block.is_summary else ""
            lines = [
                f"Diff: {block.path}{summary}",
                f"@@ -{block.old_start} +{block.new_start} @@",
            ]
            lines.extend(f"- {line}" for line in block.old_text.splitlines())
            lines.extend(f"+ {line}" for line in block.new_text.splitlines())
            parts.append("\n".join(lines))
            continue
        if isinstance(block, TodoDisplayBlock):
            lines = ["Todos:"]
            for item in block.items:
                marker = {"done": "[x]", "in_progress": "[>]"}.get(item.status, "[ ]")
                indent = "  " * max(0, item.depth)
                note = f" — {item.notes}" if item.notes else ""
                lines.append(f"{indent}{marker} {item.title}{note}")
            parts.append("\n".join(lines))
            continue
        if isinstance(block, ShellDisplayBlock):
            parts.append(f"Shell output · {block.language}")
            continue
        if type(block).__name__ == "BackgroundTaskDisplayBlock":
            parts.append(
                f"[{getattr(block, 'status', '?')}] "
                f"{getattr(block, 'task_id', '?')} · {getattr(block, 'kind', 'task')}: "
                f"{getattr(block, 'description', '')}"
            )
            continue
        if isinstance(block, UnknownDisplayBlock):
            parts.append(f"{block.type}:\n{_json_text(block.data)}")
            continue
        if isinstance(block, DisplayBlock):
            data = _model_data(block)
            label = type(block).__name__.removesuffix("DisplayBlock") or block.type
            parts.append(f"{label}:\n{_json_text(data)}")
            continue
        parts.append(f"{type(block).__name__}:\n{_json_text(block)}")
    return "\n\n".join(parts)


def _tool_call_text(
    name: str,
    _call_id: str,
    arguments: str | None,
    extras: object = None,
) -> str:
    return format_tool_call_text(name, arguments, extras)


def _tool_result_text(message: ToolResult, tool_name: str | None) -> str:
    result = message.return_value
    return format_tool_result_text(
        tool_name,
        is_error=result.is_error,
        message=result.message or "",
        display=format_display_blocks(result.display),
        output=_content_text(result.output),
        extras=result.extras,
    )


def _approval_text(message: ApprovalRequest) -> str:
    lines = [
        f"{message.sender} requests: {message.action}",
        message.description,
        f"Request ID: {message.id}",
        f"Tool call ID: {message.tool_call_id}",
    ]
    source = [message.source_kind, message.source_id, message.subagent_type, message.agent_id]
    if any(source):
        lines.append("Source: " + " · ".join(str(value) for value in source if value))
    if message.source_description:
        lines.append(f"Source detail: {message.source_description}")
    display = format_display_blocks(message.display)
    if display:
        lines.append(f"Display:\n{display}")
    return "\n".join(filter(None, lines))


def _field_lines(title: str, fields: Sequence[tuple[str, object]]) -> str:
    lines = [title]
    lines.extend(
        f"{label}: {value}" for label, value in fields if value is not None and value != ""
    )
    return "\n".join(lines)


def _render_subagent(message: SubagentEvent, state: RenderState) -> RenderEvent:
    state.scope_stream_kind = None
    identity = message.agent_id or message.parent_tool_call_id or "unknown"
    label = f"Subagent {message.subagent_type or 'agent'} · {identity}"
    nested_state = state.subagents.setdefault(identity, RenderState())
    nested_message = message.event
    if isinstance(nested_message, TurnBegin):
        nested_state.scope_stream_kind = None
        text = user_input_text(nested_message.user_input)
        return RenderEvent("system", f"{label}\nTurn started\nInput:\n{text}")
    if isinstance(nested_message, TurnEnd):
        nested_state.scope_stream_kind = None
        return RenderEvent("system", f"{label}\nTurn completed")

    rendered = render_wire_message(nested_message, state=nested_state)
    if rendered is None:
        return RenderEvent(
            "system",
            f"{label}\n{type(nested_message).__name__}:\n{_json_text(nested_message)}",
        )
    prefix = ""
    if not rendered.streaming:
        nested_state.scope_stream_kind = None
        prefix = f"{label}\n"
    elif rendered.starts_stream or nested_state.scope_stream_kind != rendered.kind:
        nested_state.scope_stream_kind = rendered.kind
        prefix = f"{label}\n"
    return RenderEvent(
        rendered.kind,
        prefix + rendered.text,
        streaming=rendered.streaming,
        starts_stream=rendered.starts_stream,
        replaces_stream=rendered.replaces_stream,
    )


def _render_named_event(message: object) -> RenderEvent | None:
    name = type(message).__name__
    if name in _OBSERVABILITY_EVENTS:
        return None
    if name == "StepRetry":
        return RenderEvent(
            "error",
            _field_lines(
                f"Retrying step {getattr(message, 'n', '?')}",
                (
                    ("Next attempt", getattr(message, "next_attempt", None)),
                    ("Maximum attempts", getattr(message, "max_attempts", None)),
                    ("Wait", f"{getattr(message, 'wait_s', 0):g}s"),
                    ("Error", getattr(message, "error_type", None)),
                    ("HTTP status", getattr(message, "status_code", None)),
                ),
            ),
        )
    if name == "HookTriggered":
        return RenderEvent(
            "system",
            _field_lines(
                "Hook triggered",
                (
                    ("Event", getattr(message, "event", None)),
                    ("Target", getattr(message, "target", None)),
                    ("Hooks", getattr(message, "hook_count", None)),
                ),
            ),
        )
    if name == "HookResolved":
        action = getattr(message, "action", "allow")
        return RenderEvent(
            "error" if action == "block" else "system",
            _field_lines(
                "Hook resolved",
                (
                    ("Event", getattr(message, "event", None)),
                    ("Target", getattr(message, "target", None)),
                    ("Action", action),
                    ("Reason", getattr(message, "reason", None)),
                    ("Duration", f"{getattr(message, 'duration_ms', 0)}ms"),
                ),
            ),
        )
    if name in {"MCPLoadingBegin", "MCPLoadingEnd"}:
        text = "Loading MCP servers" if name.endswith("Begin") else "MCP loading completed"
        return RenderEvent("system", text)
    if name == "Notification":
        severity = str(getattr(message, "severity", "info"))
        kind: RenderKind = "error" if severity == "error" else "system"
        return RenderEvent(
            kind,
            _field_lines(
                f"Notification · {getattr(message, 'title', '')}",
                (
                    ("Body", getattr(message, "body", None)),
                    ("Severity", severity),
                    ("Category", getattr(message, "category", None)),
                    ("Type", getattr(message, "type", None)),
                    ("Source", getattr(message, "source_id", None)),
                    ("Payload", _json_text(getattr(message, "payload", {}))),
                ),
            ),
        )
    if name == "BtwBegin":
        return RenderEvent("system", f"Side question:\n{getattr(message, 'question', '')}")
    if name == "BtwEnd":
        error = getattr(message, "error", None)
        text = error or getattr(message, "response", None) or "(no response)"
        return RenderEvent("error" if error else "assistant", f"Side answer:\n{text}")
    if name == "QuestionRequest":
        lines = [
            "Question request",
            f"Request ID: {getattr(message, 'id', '')}",
            f"Tool call ID: {getattr(message, 'tool_call_id', '')}",
        ]
        for index, question in enumerate(getattr(message, "questions", ()), start=1):
            header = f" [{question.header}]" if getattr(question, "header", "") else ""
            lines.append(f"{index}.{header} {question.question}")
            if getattr(question, "body", ""):
                lines.append(str(question.body))
            for option in getattr(question, "options", ()):
                detail = f" — {option.description}" if option.description else ""
                lines.append(f"  - {option.label}{detail}")
            if getattr(question, "multi_select", False):
                lines.append("  Multiple selections allowed")
        return RenderEvent("approval", "\n".join(lines))
    if name == "HookRequest":
        return RenderEvent(
            "approval",
            _field_lines(
                "Hook request",
                (
                    ("Request ID", getattr(message, "id", None)),
                    ("Subscription", getattr(message, "subscription_id", None)),
                    ("Event", getattr(message, "event", None)),
                    ("Target", getattr(message, "target", None)),
                    ("Input", _json_text(getattr(message, "input_data", {}))),
                ),
            ),
        )
    if name == "ToolCallRequest":
        return RenderEvent(
            "approval",
            _field_lines(
                "External tool call request",
                (
                    ("Call ID", getattr(message, "id", None)),
                    ("Name", getattr(message, "name", None)),
                    ("Arguments", _pretty_json(getattr(message, "arguments", None))),
                ),
            ),
        )

    data = _model_data(message)
    if data is message:
        return None
    return RenderEvent("system", f"{name}:\n{_json_text(data)}")


def render_wire_message(
    message: object,
    *,
    state: RenderState | None = None,
) -> RenderEvent | None:
    """Map a public SDK wire message to a detailed, stable UI event."""

    state = state or RenderState()
    if isinstance(message, TextPart):
        starts_stream = state.scope_stream_kind != "assistant"
        state.scope_stream_kind = "assistant"
        return RenderEvent("assistant", message.text, streaming=True, starts_stream=starts_stream)
    if isinstance(message, ThinkPart):
        starts_stream = state.scope_stream_kind != "thinking"
        state.scope_stream_kind = "thinking"
        return RenderEvent("thinking", message.think, streaming=True, starts_stream=starts_stream)
    if isinstance(message, ImageURLPart):
        state.scope_stream_kind = None
        return RenderEvent("system", _media_text("Image", message.image_url))
    if isinstance(message, AudioURLPart):
        state.scope_stream_kind = None
        return RenderEvent("system", _media_text("Audio", message.audio_url))
    if isinstance(message, VideoURLPart):
        state.scope_stream_kind = None
        return RenderEvent("system", _media_text("Video", message.video_url))
    if isinstance(message, ToolCall):
        state.tool_names[message.id] = message.function.name
        state.tool_arguments[message.id] = message.function.arguments or ""
        if message.extras:
            state.tool_extras[message.id] = message.extras
        state.active_tool_call_id = message.id
        state.scope_stream_kind = "tool"
        return RenderEvent(
            "tool",
            _tool_call_text(
                message.function.name,
                message.id,
                message.function.arguments,
                message.extras,
            ),
            streaming=True,
            starts_stream=True,
        )
    if isinstance(message, ToolCallPart):
        fragment = message.arguments_part or ""
        starts_stream = state.active_tool_call_id is None or state.scope_stream_kind != "tool"
        state.scope_stream_kind = "tool"
        call_id = state.active_tool_call_id
        if call_id is None:
            return RenderEvent(
                "tool",
                f"Tool arguments fragment:\n{fragment}",
                streaming=True,
                starts_stream=True,
            )
        arguments = state.tool_arguments.get(call_id, "") + fragment
        state.tool_arguments[call_id] = arguments
        return RenderEvent(
            "tool",
            _tool_call_text(
                state.tool_names.get(call_id, "Tool"),
                call_id,
                arguments,
                state.tool_extras.get(call_id),
            ),
            streaming=True,
            starts_stream=starts_stream,
            replaces_stream=True,
        )
    if isinstance(message, ToolResult):
        tool_name = state.tool_names.pop(message.tool_call_id, None)
        state.tool_arguments.pop(message.tool_call_id, None)
        state.tool_extras.pop(message.tool_call_id, None)
        if state.active_tool_call_id == message.tool_call_id:
            state.active_tool_call_id = None
        state.scope_stream_kind = None
        kind: RenderKind = "error" if message.return_value.is_error else "tool_result"
        return RenderEvent(kind, _tool_result_text(message, tool_name))
    if isinstance(message, ApprovalRequest):
        state.scope_stream_kind = None
        return RenderEvent("approval", _approval_text(message))
    if isinstance(message, ApprovalResponse):
        state.scope_stream_kind = None
        feedback = f"\nFeedback: {message.feedback}" if message.feedback else ""
        return RenderEvent(
            "system" if message.response != "reject" else "error",
            f"Approval {message.response}\nRequest ID: {message.request_id}{feedback}",
        )
    if isinstance(message, SubagentEvent):
        return _render_subagent(message, state)
    if isinstance(message, StepBegin):
        state.scope_stream_kind = None
        return RenderEvent("system", f"Step {message.n}")
    if isinstance(message, StepInterrupted):
        state.scope_stream_kind = None
        return RenderEvent("error", "Current step was interrupted")
    if isinstance(message, CompactionBegin):
        state.scope_stream_kind = None
        return RenderEvent(
            "system",
            _field_lines(
                "Compacting context",
                (
                    ("Trigger", message.trigger),
                    ("Shadowed tokens", message.shadowed_tokens),
                    ("Compaction ID", message.compaction_id),
                ),
            ),
        )
    if isinstance(message, CompactionEnd):
        state.scope_stream_kind = None
        return RenderEvent(
            "error" if message.error else "system",
            _field_lines(
                "Context compaction failed" if message.error else "Context compacted",
                (
                    ("Trigger", message.trigger),
                    ("Shadowed tokens", message.shadowed_tokens),
                    ("Estimated tokens", message.estimated_token_count),
                    ("Compaction ID", message.compaction_id),
                    ("Error", message.error),
                ),
            ),
        )
    if isinstance(message, StatusUpdate):
        for name in (
            "context_usage",
            "context_tokens",
            "max_context_tokens",
            "token_usage",
            "message_id",
            "mcp_status",
        ):
            value = getattr(message, name, None)
            if value is not None:
                state.status_values[name] = value
        return RenderEvent("status", format_status(SimpleNamespace(**state.status_values)))
    if isinstance(message, TurnBegin | TurnEnd):
        state.scope_stream_kind = None
        return None
    state.scope_stream_kind = None
    return _render_named_event(message)


def format_status(status: object) -> str:
    """Format context, token, and MCP status information."""

    tokens = getattr(status, "context_tokens", None)
    maximum = getattr(status, "max_context_tokens", None)
    usage = getattr(status, "context_usage", None)

    fields: list[str] = []
    if tokens is not None:
        context = f"context {tokens:,}"
        if maximum:
            context += f"/{maximum:,}"
        fields.append(context)
    if usage is not None:
        percentage = usage * 100 if 0 <= usage <= 1 else usage
        fields.append(f"{percentage:.1f}%")

    token_usage = getattr(status, "token_usage", None)
    if token_usage is not None:
        input_other = getattr(token_usage, "input_other", 0)
        cache_read = getattr(token_usage, "input_cache_read", 0)
        cache_creation = getattr(token_usage, "input_cache_creation", 0)
        total_input = input_other + cache_read + cache_creation
        fields.append(
            f"tokens in {total_input:,} "
            f"(new {input_other:,}, cache read {cache_read:,}, cache write {cache_creation:,})"
        )
        fields.append(f"out {getattr(token_usage, 'output', 0):,}")

    mcp = getattr(status, "mcp_status", None)
    if mcp is not None:
        loading = "loading" if getattr(mcp, "loading", False) else "ready"
        mcp_text = (
            f"MCP {getattr(mcp, 'connected', 0)}/{getattr(mcp, 'total', 0)} "
            f"{loading}, {getattr(mcp, 'tools', 0)} tools"
        )
        servers = getattr(mcp, "servers", ())
        if servers:
            server_text = ", ".join(
                f"{getattr(server, 'name', '?')}:{getattr(server, 'status', '?')}"
                for server in servers
            )
            mcp_text += f" [{server_text}]"
        fields.append(mcp_text)
    if getattr(status, "yolo_enabled", False):
        fields.append("YOLO enabled")
    if getattr(status, "afk_enabled", False):
        fields.append("AFK enabled")
    return " · ".join(fields) or "ready"

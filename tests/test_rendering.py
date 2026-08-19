from __future__ import annotations

from kimi_agent_sdk import (
    CompactionEnd,
    StatusUpdate,
    StepBegin,
    TextPart,
    ThinkPart,
    ToolCall,
    ToolOk,
    ToolResult,
)

from kimix_tui.rendering import bounded_concat, format_status, render_wire_message, truncate_display


def test_text_and_thinking_are_streaming_events() -> None:
    text = render_wire_message(TextPart(text="hello"))
    thinking = render_wire_message(ThinkPart(think="considering"))

    assert text is not None
    assert (text.kind, text.text, text.streaming) == ("assistant", "hello", True)
    assert thinking is not None
    assert (thinking.kind, thinking.text, thinking.streaming) == (
        "thinking",
        "considering",
        True,
    )


def test_tool_call_pretty_prints_json_arguments() -> None:
    event = render_wire_message(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="read", arguments='{"path":"a.py"}'),
        )
    )

    assert event is not None
    assert event.kind == "tool"
    assert '"path": "a.py"' in event.text


def test_tool_result_uses_public_brief_display() -> None:
    event = render_wire_message(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolOk(output="long internal output", brief="read a.py"),
        )
    )

    assert event is not None
    assert (event.kind, event.text) == ("tool_result", "read a.py")


def test_status_and_compaction_are_human_readable() -> None:
    status = StatusUpdate(
        context_tokens=1_000,
        max_context_tokens=10_000,
        context_usage=0.1,
    )
    rendered = render_wire_message(status)
    compacted = render_wire_message(CompactionEnd(estimated_token_count=500))

    assert format_status(status) == "context 1,000/10,000 · 10.0%"
    assert rendered is not None and rendered.kind == "status"
    assert compacted is not None and "500" in compacted.text


def test_step_event_is_visible() -> None:
    event = render_wire_message(StepBegin(n=3))

    assert event is not None
    assert event.text == "Step 3"


def test_truncate_display_keeps_prefix_only() -> None:
    assert truncate_display("short") == "short"
    clipped = truncate_display("x" * 50, limit=10)
    assert clipped.startswith("xxxxxxxxxx")
    assert "40 more chars" in clipped
    assert len(clipped) < 50


def test_bounded_concat_does_not_grow_past_limit() -> None:
    assert bounded_concat("", "hello") == "hello"
    assert bounded_concat("he", "llo") == "hello"
    clipped = bounded_concat("abc", "defghij", limit=5)
    assert clipped.startswith("abcde")
    assert "truncated" in clipped
    already = bounded_concat("abcde", "zzz", limit=5)
    assert already == "abcde"

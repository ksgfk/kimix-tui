from __future__ import annotations

from types import SimpleNamespace

from kimi_agent_sdk import (
    BriefDisplayBlock,
    CompactionEnd,
    DiffDisplayBlock,
    StatusUpdate,
    StepBegin,
    SubagentEvent,
    TextPart,
    ThinkPart,
    TodoDisplayBlock,
    TodoDisplayItem,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
)

from kimix_tui.rendering import (
    RenderState,
    format_display_blocks,
    format_status,
    render_wire_message,
)


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
    assert event.starts_stream is True
    assert event.text.startswith("read  a.py")
    assert "Call ID" not in event.text
    assert '"path"' not in event.text


def test_tool_stream_and_result_keep_all_public_details() -> None:
    state = RenderState()
    call = render_wire_message(
        ToolCall(
            id="call-1",
            function=ToolCall.FunctionBody(name="read", arguments=""),
            extras={"provider": "test"},
        ),
        state=state,
    )
    part = render_wire_message(ToolCallPart(arguments_part='{"path":"a.py"}'), state=state)
    result = render_wire_message(
        ToolResult(
            tool_call_id="call-1",
            return_value=ToolReturnValue(
                is_error=False,
                output="long internal output",
                message="success",
                display=[BriefDisplayBlock(text="read a.py")],
                extras={"bytes": 20},
            ),
        ),
        state=state,
    )

    assert call is not None and "read" in call.text
    assert part is not None and part.streaming is True
    assert part.replaces_stream is True
    assert part.text.startswith("read  a.py")
    assert "Call ID" not in part.text
    assert result is not None and result.kind == "tool_result"
    assert "a.py" in result.text
    assert "long internal output" in result.text
    assert "Call ID" not in result.text
    assert "Message:" not in result.text
    assert "Display:" not in result.text
    assert "Output:" not in result.text


def test_grep_and_todo_calls_are_human_readable() -> None:
    grep = render_wire_message(
        ToolCall(
            id="g1",
            function=ToolCall.FunctionBody(
                name="grep",
                arguments='{"pattern":"def foo","path":"src","include":"*.py"}',
            ),
        )
    )
    todo = render_wire_message(
        ToolCall(
            id="t1",
            function=ToolCall.FunctionBody(
                name="todo_write",
                arguments=(
                    '{"todos":[{"content":"Implement","status":"in_progress"},'
                    '{"content":"Verify","status":"pending"}]}'
                ),
            ),
        )
    )

    assert grep is not None
    assert grep.text.startswith("grep  def foo")
    assert "src" in grep.text
    assert "*.py" in grep.text
    assert '"pattern"' not in grep.text
    assert todo is not None
    assert "[>] Implement" in todo.text
    assert "[ ] Verify" in todo.text
    assert '"status"' not in todo.text


def test_native_display_blocks_include_diff_and_todo_details() -> None:
    text = format_display_blocks(
        [
            BriefDisplayBlock(text="Updated files"),
            DiffDisplayBlock(
                path="a.py",
                old_text="old",
                new_text="new",
                old_start=4,
                new_start=5,
            ),
            TodoDisplayBlock(
                items=[
                    TodoDisplayItem(title="Implement", status="in_progress", notes="now"),
                    TodoDisplayItem(title="Verify", status="done", depth=1),
                ]
            ),
        ]
    )

    for detail in (
        "Updated files",
        "Diff: a.py",
        "@@ -4 +5 @@",
        "- old",
        "+ new",
        "[>] Implement — now",
        "  [x] Verify",
    ):
        assert detail in text


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


def test_status_includes_tokens_message_and_mcp_details() -> None:
    status = SimpleNamespace(
        context_tokens=2_000,
        max_context_tokens=20_000,
        context_usage=0.1,
        token_usage=TokenUsage(
            input_other=100,
            input_cache_read=800,
            input_cache_creation=50,
            output=75,
        ),
        message_id="msg-1",
        mcp_status=SimpleNamespace(
            loading=False,
            connected=1,
            total=2,
            tools=8,
            servers=(SimpleNamespace(name="github", status="connected"),),
        ),
    )

    text = format_status(status)

    assert "context 2,000/20,000" in text
    assert "tokens in 950 (new 100, cache read 800, cache write 50)" in text
    assert "out 75" in text
    assert "message msg-1" not in text
    assert "message" not in text
    assert "MCP 1/2 ready, 8 tools [github:connected]" in text


def test_subagent_streams_do_not_merge_with_main_assistant() -> None:
    state = RenderState()
    main_before = render_wire_message(TextPart(text="main"), state=state)
    child_first = render_wire_message(
        SubagentEvent(agent_id="child-1", subagent_type="explore", event=TextPart(text="one")),
        state=state,
    )
    child_second = render_wire_message(
        SubagentEvent(agent_id="child-1", subagent_type="explore", event=TextPart(text="two")),
        state=state,
    )
    main_after = render_wire_message(TextPart(text="back"), state=state)

    assert main_before is not None and main_before.starts_stream is True
    assert child_first is not None and child_first.starts_stream is True
    assert child_first.text.startswith("Subagent explore · child-1\n")
    assert child_second is not None and child_second.starts_stream is False
    assert child_second.text == "two"
    assert main_after is not None and main_after.starts_stream is True


def test_step_event_is_visible() -> None:
    event = render_wire_message(StepBegin(n=3))

    assert event is not None
    assert event.text == "Step 3"

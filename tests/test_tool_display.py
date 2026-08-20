from __future__ import annotations

import orjson

from kimix_tui.tool_display import format_tool_call_text, format_tool_result_text


def test_write_call_summary_line_stays_short_and_body_keeps_content() -> None:
    content = "hello\nworld\n" + ("block\n" * 30)
    text = format_tool_call_text(
        "write",
        orjson.dumps({"path": "notes.txt", "content": content}).decode("utf-8"),
    )

    first, *rest = text.splitlines()
    assert first == "write  notes.txt"
    assert "content:" in rest
    assert "hello" in rest
    assert "world" in rest
    assert rest.count("block") == 30
    assert content.rstrip("\n") in text


def test_python_call_keeps_full_code_under_code_label() -> None:
    code = "def run():\n    return 1 + 2\n"
    text = format_tool_call_text("python", orjson.dumps({"code": code}).decode("utf-8"))

    assert text.startswith("python  def run():")
    assert "\ncode:\n" in text
    assert code in text


def test_shell_call_keeps_full_command_and_cwd() -> None:
    command = "pytest -q tests/test_tool_display.py"
    text = format_tool_call_text(
        "bash",
        orjson.dumps({"command": command, "cwd": "/tmp/work"}).decode("utf-8"),
    )

    assert text.startswith(f"bash  {command}")
    assert "command:" in text
    assert command in text
    assert "cwd: /tmp/work" in text


def test_argument_aliases_decode_like_cli() -> None:
    text = format_tool_call_text(
        "edit",
        orjson.dumps(
            {
                "file_path": "app.py",
                "old_string": "foo = 1",
                "new_string": "foo = 2",
            }
        ).decode("utf-8"),
    )

    assert text.startswith("edit  app.py")
    assert "\nold:\nfoo = 1" in text
    assert "\nnew:\nfoo = 2" in text


def test_generic_tool_keeps_pretty_original_arguments() -> None:
    text = format_tool_call_text(
        "custom_mcp",
        orjson.dumps({"query": "abc", "limit": 5}).decode("utf-8"),
    )

    assert text.startswith("custom_mcp  query:abc limit:5")
    assert '"query": "abc"' in text
    assert '"limit": 5' in text


def test_tool_result_keeps_full_output_below_headline() -> None:
    output = "match 1\nmatch 2\nmatch 3"
    text = format_tool_result_text(
        "grep",
        is_error=False,
        message="success",
        display="3 matches",
        output=output,
    )

    assert text.startswith("grep  3 matches")
    assert output in text


def test_tool_result_keeps_message_body_when_first_line_matches_headline() -> None:
    message = "succeeded\nfull command output\nmore lines"
    text = format_tool_result_text(
        "bash",
        is_error=False,
        message=message,
        display="",
        output="",
    )

    assert text.startswith("bash  succeeded")
    assert "full command output" in text
    assert "more lines" in text


def test_native_tool_calls_and_results_keep_extras() -> None:
    call = format_tool_call_text("read", '{"path":"a.py"}', extras={"provider": "test"})
    result = format_tool_result_text(
        "read",
        is_error=False,
        message="success",
        display="a.py",
        output="file body",
        extras={"bytes": 20},
    )

    assert '"provider": "test"' in call
    assert '"bytes": 20' in result
    assert "file body" in result

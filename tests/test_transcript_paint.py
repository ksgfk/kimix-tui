from __future__ import annotations

from kimix_tui.qt.paint import layout_plain, layout_record, markdown_plain_text
from kimix_tui.transcript_layout import bar_color_name, compact_summary, record_label


def test_user_keeps_cyan_label_and_raw_markdown() -> None:
    layout = layout_record("user", "please use **bold** here", width=48)
    plain = layout_plain("user", "please use **bold** here", width=48)
    assert layout.label == "You"
    assert "**bold**" in layout.body
    assert "⧉" in plain
    assert layout.bar_color == "cyan"
    assert layout.markdown is False


def test_assistant_renders_simple_markdown(qapp) -> None:
    layout = layout_record("assistant", "## Title\n\nUse **bold** and `code`.", width=48)
    rendered = markdown_plain_text(layout.body)
    assert layout.label == "AI"
    assert layout.markdown is True
    assert "Title" in rendered
    assert "bold" in rendered
    assert "**bold**" not in rendered
    assert "code" in rendered
    assert layout.bar_color == "green"


def test_thinking_uses_muted_italic() -> None:
    layout = layout_record("thinking", "considering options", width=48, expanded=True)
    assert "▾ Think" in layout.header
    assert "considering options" not in layout.header
    assert "considering options" in layout.body
    assert layout.italic_body is True
    assert layout.bar_color == "muted"


def test_non_dialogue_records_are_compacted_to_one_line() -> None:
    layout = layout_record(
        "tool_result",
        "Read file succeeded\nCall ID: call-123\nOutput: a very detailed result",
        width=48,
    )
    assert layout.compact is True
    assert "▸ Read" in layout.header
    assert "file succeeded" in layout.header
    assert "Call ID" not in layout.header
    assert "very detailed result" not in layout.header
    assert layout.body == ""


def test_tool_summary_uses_compact_headline_not_payload() -> None:
    layout = layout_record(
        "tool",
        "write  demo.py\npath: demo.py\ncontent:\nprint('hello')\nprint('world')",
        width=48,
    )
    assert layout.compact is True
    assert "▸ Write" in layout.header
    assert "demo.py" in layout.header
    assert "print('hello')" not in layout.header
    assert "print('world')" not in layout.header


def test_expanded_non_dialogue_records_show_the_full_message() -> None:
    layout = layout_record(
        "tool_result",
        "Read file succeeded\nCall ID: call-123\nOutput: all details",
        width=48,
        expanded=True,
    )
    assert layout.compact is False
    assert "▾ Read" in layout.header
    assert "Read file succeeded" not in layout.header
    assert "Call ID: call-123" in layout.body
    assert "Output: all details" in layout.body


def test_expanded_tool_call_shows_original_cli_style_payload() -> None:
    content = "line one\nline two\nline three"
    layout = layout_record(
        "tool",
        f"write  demo.py\npath: demo.py\ncontent:\n{content}",
        width=48,
        expanded=True,
    )
    assert "▾ Write" in layout.header
    assert "content:" in layout.body
    assert "line one" in layout.body
    assert "line two" in layout.body
    assert "line three" in layout.body


def test_non_dialogue_body_is_muted_not_vivid() -> None:
    samples = (
        ("tool", "custom_mcp"),
        ("tool_result", "succeeded"),
        ("thinking", "hmm"),
        ("system", "ready"),
        ("approval", "allow this"),
    )
    for kind, text in samples:
        assert bar_color_name(kind, text) == "muted"


def test_known_tools_use_specialized_titles_and_colors() -> None:
    cases = (
        ("tool", "grep  def foo", "Grep", "cyan"),
        ("tool_result", "grep  3 matches", "Grep", "cyan"),
        ("tool", "read  a.py", "Read", "cyan"),
        ("tool_result", "read  a.py", "Read", "cyan"),
        ("tool", "todo_write  2 items", "Todo", "yellow"),
        ("tool", "bash  ls -la", "Bash", "green"),
    )
    for kind, text, title, color in cases:
        layout = layout_record(kind, text, width=48)
        assert f"▸ {title}" in layout.header
        assert layout.bar_color == color
        assert record_label(kind, text) == title


def test_compacted_records_use_ascii_ellipsis_when_clipped() -> None:
    layout = layout_record(
        "system",
        "This status message is much too long for the available space",
        width=24,
    )
    plain = layout_plain("system", "This status message is much too long for the available space", width=24)
    assert layout.compact is True
    assert "This..." in layout.header or compact_summary(
        "This status message is much too long for the available space", 8
    )
    assert plain.rstrip().endswith("⧉")


def test_assistant_markdown_is_opt_in() -> None:
    plain = layout_record("assistant", "no markup here", width=48)
    marked = layout_record("assistant", "Use **bold**", width=48)
    assert plain.markdown is False
    assert marked.markdown is True

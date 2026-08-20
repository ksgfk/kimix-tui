from __future__ import annotations

from io import StringIO

from rich.console import Console
from textual.color import Color
from textual.filter import Monochrome

from kimix_tui.transcript_paint import render_record_strips, sanitize_strip


def _console(width: int = 48) -> Console:
    return Console(
        file=StringIO(),
        width=width,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
    )


def _plain(strips: object) -> str:
    lines: list[str] = []
    for strip in strips:  # type: ignore[union-attr]
        lines.append("".join(segment.text for segment in strip._segments))
    return "\n".join(lines)


def _styles(strips: object) -> list[object]:
    styles: list[object] = []
    for strip in strips:  # type: ignore[union-attr]
        styles.extend(segment.style for segment in strip._segments)
    return styles


def test_user_keeps_cyan_label_and_raw_markdown() -> None:
    strips = render_record_strips(
        "user",
        "please use **bold** here",
        width=48,
        console=_console(),
    )
    plain = _plain(strips)
    styles = _styles(strips)

    assert "You" in plain
    assert "**bold**" in plain
    assert "⧉" in plain
    assert any(style is not None and "cyan" in str(style).lower() for style in styles)


def test_assistant_renders_simple_markdown() -> None:
    strips = render_record_strips(
        "assistant",
        "## Title\n\nUse **bold** and `code`.",
        width=48,
        console=_console(),
    )
    plain = _plain(strips)
    styles = _styles(strips)

    assert "AI" in plain
    assert "Title" in plain
    assert "bold" in plain
    assert "**bold**" not in plain
    assert "code" in plain
    assert any(style is not None and style.bold for style in styles)
    assert any(style is not None and "green" in str(style).lower() for style in styles)


def test_thinking_uses_muted_italic() -> None:
    strips = render_record_strips(
        "thinking",
        "considering options",
        width=48,
        console=_console(),
        expanded=True,
    )
    header = "".join(segment.text for segment in strips[0]._segments)
    plain = _plain(strips)
    styles = _styles(strips)
    assert "▾ Think" in header
    assert "considering options" not in header
    assert "considering options" in plain
    assert any(style is not None and style.italic for style in styles)


def test_non_dialogue_records_are_compacted_to_one_line() -> None:
    strips = render_record_strips(
        "tool_result",
        "Read file succeeded\nCall ID: call-123\nOutput: a very detailed result",
        width=48,
        console=_console(),
    )

    assert len(strips) == 1
    assert "▸ Read" in _plain(strips)
    assert "file succeeded" in _plain(strips)
    assert "Call ID" not in _plain(strips)
    assert "very detailed result" not in _plain(strips)


def test_tool_summary_uses_compact_headline_not_payload() -> None:
    strips = render_record_strips(
        "tool",
        "write  demo.py\npath: demo.py\ncontent:\nprint('hello')\nprint('world')",
        width=48,
        console=_console(),
    )

    plain = _plain(strips)
    assert len(strips) == 1
    assert "▸ Write" in plain
    assert "demo.py" in plain
    assert "print('hello')" not in plain
    assert "print('world')" not in plain


def test_expanded_non_dialogue_records_show_the_full_message() -> None:
    strips = render_record_strips(
        "tool_result",
        "Read file succeeded\nCall ID: call-123\nOutput: all details",
        width=48,
        console=_console(),
        expanded=True,
    )

    plain = _plain(strips)
    header = "".join(segment.text for segment in strips[0]._segments)
    assert len(strips) > 1
    assert "▾ Read" in header
    assert "Read file succeeded" not in header
    assert "Call ID: call-123" in plain
    assert "Output: all details" in plain


def test_expanded_tool_call_shows_original_cli_style_payload() -> None:
    content = "line one\nline two\nline three"
    strips = render_record_strips(
        "tool",
        f"write  demo.py\npath: demo.py\ncontent:\n{content}",
        width=48,
        console=_console(),
        expanded=True,
    )

    plain = _plain(strips)
    header = "".join(segment.text for segment in strips[0]._segments)
    assert "▾ Write" in header
    assert "content:" in plain
    assert "line one" in plain
    assert "line two" in plain
    assert "line three" in plain


def test_non_dialogue_body_is_gray_not_vivid() -> None:
    samples = (
        ("tool", "custom_mcp"),
        ("tool_result", "succeeded"),
        ("thinking", "hmm"),
        ("system", "ready"),
        ("approval", "allow this"),
    )
    for kind, text in samples:
        styles = _styles(
            render_record_strips(
                kind,
                text,
                width=48,
                console=_console(),
                expanded=True,
            )
        )
        joined = " ".join(str(style).lower() for style in styles if style)
        assert "magenta" not in joined
        assert "red" not in joined
        assert "yellow" not in joined
        assert "blue" not in joined
        assert "black" in joined


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
        strips = render_record_strips(kind, text, width=48, console=_console())
        plain = _plain(strips)
        joined = " ".join(str(style).lower() for style in _styles(strips) if style)
        assert f"▸ {title}" in plain
        assert color in joined


def test_compacted_records_use_ascii_ellipsis_when_clipped() -> None:
    strips = render_record_strips(
        "system",
        "This status message is much too long for the available space",
        width=24,
        console=_console(),
    )

    plain = _plain(strips)
    assert len(strips) == 1
    assert "This..." in plain
    assert plain.rstrip().endswith("⧉")


def test_transcript_text_has_no_background_colors() -> None:
    samples = (
        ("user", "plain text"),
        ("assistant", "Inline `code`\n\n```python\nprint(1)\n```"),
        ("tool", 'read_file {"path": "demo.py"}'),
        ("tool_result", "done"),
    )

    for kind, text in samples:
        styles = _styles(render_record_strips(kind, text, width=48, console=_console()))
        assert all(style is None or style.bgcolor is None for style in styles)


def test_painted_strips_survive_monochrome_filter() -> None:
    mono = Monochrome()
    background = Color.parse("#000000")
    samples = (
        ("user", "please use **bold**"),
        ("assistant", "## Title\n\nUse **bold** and `code`."),
        ("tool", "read_file"),
        ("thinking", "hmm"),
    )
    for kind, text in samples:
        strips = render_record_strips(kind, text, width=48, console=_console())
        for strip in strips:
            assert all(segment.style is not None for segment in strip._segments)
            cropped = strip.crop_extend(0, 80, None)
            sanitize_strip(cropped).apply_filter(mono, background)

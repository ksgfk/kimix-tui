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
    )
    styles = _styles(strips)
    assert any(style is not None and style.italic for style in styles)


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

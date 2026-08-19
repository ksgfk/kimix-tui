"""Paint transcript records as colored, optionally markdown-styled strips."""

from __future__ import annotations

from collections.abc import Iterable

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip

_NULL = Style.null()
_BAR = "▎ "

_LABELS: dict[str, tuple[str, str]] = {
    "user": ("You", "bold bright_cyan"),
    "assistant": ("AI", "bold bright_green"),
    "thinking": ("Think", "italic bright_black"),
    "tool": ("Tool", "bold bright_magenta"),
    "tool_result": ("Result", "bright_blue"),
    "approval": ("Approval", "bold yellow"),
    "system": ("System", "bright_black"),
    "error": ("Error", "bold red"),
}

_BAR_COLOR: dict[str, str] = {
    "user": "bright_cyan",
    "assistant": "bright_green",
    "thinking": "bright_black",
    "tool": "magenta",
    "tool_result": "bright_blue",
    "approval": "yellow",
    "system": "bright_black",
    "error": "red",
}

_BODY_STYLE: dict[str, str] = {
    "thinking": "italic bright_black",
    "system": "bright_black",
    "error": "red",
}


def record_label(kind: str) -> str:
    """Return the label shown for a transcript record."""

    return _LABELS.get(kind, (kind.title(), "bold"))[0]


def _without_background(style: Style) -> Style:
    if style.bgcolor is None:
        return style
    return Style(
        color=style.color,
        bold=style.bold,
        dim=style.dim,
        italic=style.italic,
        underline=style.underline,
        blink=style.blink,
        blink2=style.blink2,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        underline2=style.underline2,
        frame=style.frame,
        encircle=style.encircle,
        overline=style.overline,
        link=style.link,
        meta=style.meta,
    )


def _fill_none(segments: Iterable[Segment], fallback: Style = _NULL) -> list[Segment]:
    return [
        Segment(
            text,
            _without_background(style if style is not None else fallback),
            control,
        )
        for text, style, control in segments
    ]


def sanitize_strip(strip: Strip, fallback: Style = _NULL) -> Strip:
    """Ensure every segment has a Style so Textual filters do not crash."""

    return Strip(_fill_none(strip._segments, fallback), strip.cell_length)


def _cell_length(segments: list[Segment]) -> int:
    return sum(cell_len(text) for text, _style, control in segments if not control)


def _with_bar(
    segments: list[Segment],
    bar_style: Style,
) -> list[Segment]:
    return [Segment(_BAR, bar_style), *_fill_none(segments)]


def _strip_from_text(
    line: Text,
    console: Console,
    bar_style: Style,
) -> Strip:
    segments = _with_bar(list(line.render(console)), bar_style)
    return Strip(segments, _cell_length(segments))


def _markdown_lines(text: str, width: int, console: Console) -> list[list[Segment]]:
    markdown = Markdown(text, justify="left", hyperlinks=False)
    options = console.options.update(width=width, highlight=False)
    return console.render_lines(markdown, options, pad=False)


def render_record_strips(
    kind: str,
    text: str,
    *,
    width: int,
    console: Console,
) -> list[Strip]:
    """Turn one transcript row into wrapped strips with kind colors."""

    label, label_style = _LABELS.get(kind, (kind.title(), "bold"))
    bar_style = Style.parse(_BAR_COLOR.get(kind, "bright_black"))
    body_style = _BODY_STYLE.get(kind, "")
    inner_width = max(8, width - cell_len(_BAR))
    strips: list[Strip] = []

    if kind == "assistant":
        header = Text(label, style=label_style)
        strips.append(_strip_from_text(header, console, bar_style))
        try:
            for line in _markdown_lines(text, inner_width, console):
                segments = _with_bar(line, bar_style)
                strips.append(Strip(segments, _cell_length(segments)))
        except Exception:  # noqa: BLE001 - fall back to plain body
            body = Text(text, style=body_style)
            for wrapped in body.wrap(console, inner_width, overflow="fold"):
                strips.append(_strip_from_text(wrapped, console, bar_style))
    else:
        assembled = Text.assemble(
            (f"{label}  ", label_style),
            (text, body_style),
        )
        for wrapped in assembled.wrap(console, inner_width, overflow="fold"):
            strips.append(_strip_from_text(wrapped, console, bar_style))

    if not strips:
        strips.append(Strip.blank(width, _NULL))
    strips.append(Strip.blank(width, _NULL))
    return [sanitize_strip(strip) for strip in strips]

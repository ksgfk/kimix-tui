"""Paint transcript records as colored, optionally markdown-styled strips."""

from __future__ import annotations

from collections.abc import Iterable

from rich.cells import cell_len, set_cell_size
from rich.console import Console
from rich.markdown import Markdown
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip

_NULL = Style.null()
_HEADER_MARK = "▌ "
_BODY_INDENT = "  "
_COPY_SUFFIX = "  ⧉  "

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

_MARKDOWN_HINTS = ("`", "*", "_", "#", ">", "[", "~~")
_DIALOGUE_KINDS = frozenset({"user", "assistant"})
_SUMMARY_METADATA_PREFIXES = (
    "Call ID:",
    "Request ID:",
    "Tool call ID:",
    "Compaction ID:",
)
_SUMMARY_SECTION_LABELS = frozenset(
    {"Arguments:", "Message:", "Display:", "Output:", "Extras:", "Input:", "Payload:"}
)


def record_label(kind: str) -> str:
    """Return the label shown for a transcript record."""

    return _LABELS.get(kind, (kind.title(), "bold"))[0]


def is_dialogue_record(kind: str) -> bool:
    """Return whether a record is a user or assistant chat message."""

    return kind in _DIALOGUE_KINDS


def is_compact_record(kind: str, *, expanded: bool = False) -> bool:
    """Return whether a transcript record is shown as a one-line summary."""

    return not is_dialogue_record(kind) and not expanded


def copy_hit_start(width: int) -> int:
    """Return the first column occupied by the header copy action."""

    return max(0, width - cell_len(_COPY_SUFFIX))


def _compact_summary(text: str, width: int) -> str:
    """Build a useful one-line summary while preserving wide characters."""

    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        summary = "(no details)"
    else:
        details = [
            line
            for line in lines[1:]
            if line not in _SUMMARY_SECTION_LABELS
            and not line.startswith(_SUMMARY_METADATA_PREFIXES)
        ]
        summary = lines[0]
        if details:
            summary += " · " + " ".join(details)
    if cell_len(summary) <= width:
        return summary
    if width <= 3:
        return "." * width
    return f"{set_cell_size(summary, width - 3)}..."


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


def _with_prefix(
    segments: list[Segment],
    prefix: str,
    prefix_style: Style,
) -> list[Segment]:
    return [Segment(prefix, prefix_style), *_fill_none(segments)]


def _strip_from_text(
    line: Text,
    console: Console,
    prefix: str = _BODY_INDENT,
    prefix_style: Style = _NULL,
) -> Strip:
    segments = _with_prefix(list(line.render(console)), prefix, prefix_style)
    return Strip(segments, _cell_length(segments))


def _header_strip(
    content: Text,
    *,
    width: int,
    console: Console,
    mark_style: Style,
) -> Strip:
    """Render a clean header with a stable, right-aligned copy action."""

    content = content.copy()
    content.truncate(
        max(0, width - cell_len(_HEADER_MARK) - cell_len(_COPY_SUFFIX)),
        overflow="ellipsis",
    )
    segments = _with_prefix(list(content.render(console)), _HEADER_MARK, mark_style)
    gap = max(0, copy_hit_start(width) - _cell_length(segments))
    segments.extend(
        [
            Segment(" " * gap, _NULL),
            Segment(_COPY_SUFFIX, Style.parse("dim")),
        ]
    )
    return Strip(segments, _cell_length(segments))


def _markdown_lines(text: str, width: int, console: Console) -> list[list[Segment]]:
    markdown = Markdown(text, justify="left", hyperlinks=False)
    options = console.options.update(width=width, highlight=False)
    return console.render_lines(markdown, options, pad=False)


def _plain_text_lines(text: str, width: int, console: Console) -> list[list[Segment]]:
    """Wrap plain assistant output without constructing Rich's Markdown tree."""

    body = Text(text)
    return [list(wrapped.render(console)) for wrapped in body.wrap(console, width, overflow="fold")]


def render_record_strips(
    kind: str,
    text: str,
    *,
    width: int,
    console: Console,
    expanded: bool = False,
) -> list[Strip]:
    """Turn one transcript row into wrapped strips with kind colors."""

    label, label_style = _LABELS.get(kind, (kind.title(), "bold"))
    mark_style = Style.parse(_BAR_COLOR.get(kind, "bright_black"))
    body_style = _BODY_STYLE.get(kind, "")
    body_width = max(8, width - cell_len(_BODY_INDENT))
    header_width = max(0, width - cell_len(_HEADER_MARK) - cell_len(_COPY_SUFFIX))
    strips: list[Strip] = []

    compact_record = is_compact_record(kind, expanded=expanded)
    auxiliary_record = not is_dialogue_record(kind)

    if compact_record:
        prefix = f"▸ {label}  "
        summary_width = max(0, header_width - cell_len(prefix))
        header = Text.assemble(
            (prefix, label_style),
            (_compact_summary(text, summary_width), body_style),
        )
        strips.append(
            _header_strip(header, width=width, console=console, mark_style=mark_style)
        )
    elif auxiliary_record:
        prefix = f"▾ {label}  "
        summary_width = max(0, header_width - cell_len(prefix))
        strips.append(
            _header_strip(
                Text.assemble(
                    (prefix, label_style),
                    (_compact_summary(text, summary_width), body_style),
                ),
                width=width,
                console=console,
                mark_style=mark_style,
            )
        )
        body = Text(text or "(no details)", style=body_style)
        for wrapped in body.wrap(console, body_width, overflow="fold"):
            strips.append(_strip_from_text(wrapped, console))
    elif kind == "assistant":
        header = Text(label, style=label_style)
        strips.append(
            _header_strip(header, width=width, console=console, mark_style=mark_style)
        )
        try:
            lines = (
                _markdown_lines(text, body_width, console)
                if any(marker in text for marker in _MARKDOWN_HINTS)
                else _plain_text_lines(text, body_width, console)
            )
            for line in lines:
                segments = _with_prefix(line, _BODY_INDENT, _NULL)
                strips.append(Strip(segments, _cell_length(segments)))
        except Exception:  # noqa: BLE001 - fall back to plain body
            body = Text(text, style=body_style)
            for wrapped in body.wrap(console, body_width, overflow="fold"):
                strips.append(_strip_from_text(wrapped, console))
    else:
        strips.append(
            _header_strip(
                Text(label, style=label_style),
                width=width,
                console=console,
                mark_style=mark_style,
            )
        )
        body = Text(text, style=body_style)
        for wrapped in body.wrap(console, body_width, overflow="fold"):
            strips.append(_strip_from_text(wrapped, console))

    if not strips:
        strips.append(Strip.blank(width, _NULL))
    if not compact_record:
        strips.append(Strip.blank(width, _NULL))
    return [sanitize_strip(strip) for strip in strips]

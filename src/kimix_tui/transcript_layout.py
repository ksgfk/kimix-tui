"""Framework-neutral transcript labels, compact summaries, and kind rules."""

from __future__ import annotations

from unicodedata import east_asian_width

from kimix_tui.tool_display import (
    strip_tool_name,
    tool_family,
    tool_label,
    tool_name_from_text,
    tool_styles,
)

COPY_SUFFIX = "  ⧉  "
HEADER_MARK = "▌ "
BODY_INDENT = "  "

LABELS: dict[str, str] = {
    "user": "You",
    "assistant": "AI",
    "thinking": "Think",
    "tool": "Tool",
    "tool_result": "Result",
    "approval": "Approval",
    "system": "System",
    "error": "Error",
}

BAR_COLOR_NAME: dict[str, str] = {
    "user": "cyan",
    "assistant": "green",
    "thinking": "muted",
    "tool": "muted",
    "tool_result": "muted",
    "approval": "muted",
    "system": "muted",
    "error": "red",
}

FAMILY_BAR_NAME: dict[str, str] = {
    "read": "cyan",
    "grep": "cyan",
    "glob": "cyan",
    "write": "yellow",
    "edit": "yellow",
    "shell": "green",
    "python": "green",
    "todo": "yellow",
    "search": "blue",
    "fetch": "blue",
    "agent": "magenta",
}

_DEFAULT_EXPANDED_KINDS = frozenset({"thinking"})
_DIALOGUE_KINDS = frozenset({"user", "assistant"})
_MARKDOWN_HINTS = ("`", "*", "_", "#", ">", "[", "~~")
_SUMMARY_METADATA_PREFIXES = (
    "Call ID:",
    "Request ID:",
    "Tool call ID:",
    "Compaction ID:",
)
_SUMMARY_SECTION_LABELS = frozenset(
    {"Arguments:", "Message:", "Display:", "Output:", "Extras:", "Input:", "Payload:"}
)


def cell_len(text: str) -> int:
    """Return a terminal-style cell width (wide East-Asian glyphs count as two)."""

    width = 0
    for char in text:
        width += 2 if east_asian_width(char) in {"F", "W"} else 1
    return width


def set_cell_size(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` cells without adding a pad suffix."""

    if width <= 0:
        return ""
    used = 0
    chars: list[str] = []
    for char in text:
        size = 2 if east_asian_width(char) in {"F", "W"} else 1
        if used + size > width:
            break
        chars.append(char)
        used += size
    return "".join(chars)


def record_label(kind: str, text: str = "") -> str:
    """Return the label shown for a transcript record."""

    if kind in {"tool", "tool_result", "error"} and text:
        name = tool_name_from_text(text)
        if name and (kind != "error" or tool_styles(name) is not None):
            return tool_label(name, kind=kind)
    return LABELS.get(kind, kind.title())


def is_dialogue_record(kind: str) -> bool:
    """Return whether a record is a user or assistant chat message."""

    return kind in _DIALOGUE_KINDS


def is_compact_record(kind: str, *, expanded: bool = False) -> bool:
    """Return whether a transcript record is shown as a one-line summary."""

    return not is_dialogue_record(kind) and not expanded


def default_expanded(kind: str) -> bool:
    """Return whether a record kind starts expanded."""

    return kind in _DEFAULT_EXPANDED_KINDS


def has_markdown_hints(text: str) -> bool:
    """Return whether assistant text should be parsed as Markdown."""

    return any(marker in text for marker in _MARKDOWN_HINTS)


def copy_hit_start(width: int) -> int:
    """Return the first column occupied by the header copy action."""

    return max(0, width - cell_len(COPY_SUFFIX))


def bar_color_name(kind: str, text: str = "") -> str:
    """Return a stable color name used by tests and the Qt painter."""

    if kind in {"tool", "tool_result"}:
        name = tool_name_from_text(text)
        family = tool_family(name)
        if family in FAMILY_BAR_NAME:
            return FAMILY_BAR_NAME[family]
    if kind == "error":
        name = tool_name_from_text(text)
        if name and tool_styles(name) is not None:
            return "red"
    return BAR_COLOR_NAME.get(kind, "muted")


def compact_summary(text: str, width: int, *, lead_name: str | None = None) -> str:
    """Build a useful one-line summary from the compact headline."""

    summary = "(no details)"
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        candidate = strip_tool_name(line, lead_name) if lead_name else line
        if not candidate:
            continue
        if candidate in _SUMMARY_SECTION_LABELS or candidate.startswith(
            _SUMMARY_METADATA_PREFIXES
        ):
            continue
        if candidate.endswith(":") and cell_len(candidate) < 24:
            continue
        summary = candidate
        break
    if cell_len(summary) <= width:
        return summary
    if width <= 3:
        return "." * width
    return f"{set_cell_size(summary, width - 3)}..."


def expanded_body(kind: str, text: str, tool_name: str | None) -> str:
    """Return the original payload for an expanded auxiliary record."""

    body_text = text or "(no details)"
    if not tool_name:
        return body_text
    body_lines = body_text.splitlines() or [body_text]
    body_lines[0] = strip_tool_name(body_lines[0], tool_name)
    if kind in {"tool", "tool_result", "error"} and len(body_lines) > 1:
        details = "\n".join(body_lines[1:]).strip()
        if details:
            return details
    return "\n".join(body_lines).strip() or "(no details)"


def tool_name_for(kind: str, text: str) -> str | None:
    """Return the wire tool name when the kind carries one."""

    if kind in {"tool", "tool_result", "error"}:
        return tool_name_from_text(text)
    return None

"""Classify native tools and format call/result text for the transcript."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import orjson

_BORING_MESSAGES = frozenset({"success", "succeeded", "ok", "done", "completed"})

_FAMILY_ALIASES: dict[str, str] = {
    "read": "read",
    "read_file": "read",
    "readfile": "read",
    "read_image": "read",
    "readimage": "read",
    "read_media": "read",
    "readmediafile": "read",
    "grep": "grep",
    "grep_local": "grep",
    "find_str": "grep",
    "findstr": "grep",
    "glob": "glob",
    "write": "write",
    "write_file": "write",
    "writefile": "write",
    "edit": "edit",
    "edit_file": "edit",
    "editfile": "edit",
    "replace": "edit",
    "str_replace": "edit",
    "bash": "shell",
    "shell": "shell",
    "pwsh": "shell",
    "powershell": "shell",
    "run": "shell",
    "python": "python",
    "todo": "todo",
    "todo_write": "todo",
    "todowrite": "todo",
    "todo_update": "todo",
    "todoupdate": "todo",
    "web_search": "search",
    "websearch": "search",
    "search": "search",
    "fetch_url": "fetch",
    "fetchurl": "fetch",
    "web_extract": "fetch",
    "webextract": "fetch",
    "subagent": "agent",
    "agent": "agent",
}

_FAMILY_LABEL: dict[str, str] = {
    "read": "Read",
    "grep": "Grep",
    "glob": "Glob",
    "write": "Write",
    "edit": "Edit",
    "python": "Python",
    "todo": "Todo",
    "search": "Search",
    "fetch": "Fetch",
    "agent": "Agent",
}

# Header / bar colors. Bodies stay muted gray in the painter.
_FAMILY_LABEL_STYLE: dict[str, str] = {
    "read": "bold cyan",
    "grep": "bold bright_cyan",
    "glob": "bold cyan",
    "write": "bold yellow",
    "edit": "bold yellow",
    "shell": "bold green",
    "python": "bold green",
    "todo": "bold bright_yellow",
    "search": "bold blue",
    "fetch": "bold blue",
    "agent": "bold magenta",
}

_FAMILY_BAR: dict[str, str] = {
    "read": "cyan",
    "grep": "bright_cyan",
    "glob": "cyan",
    "write": "yellow",
    "edit": "yellow",
    "shell": "green",
    "python": "green",
    "todo": "bright_yellow",
    "search": "blue",
    "fetch": "blue",
    "agent": "magenta",
}


def normalize_tool_name(name: str) -> str:
    """Canonicalize a wire tool name for family lookup."""

    return name.strip().lower().replace("-", "_")


def tool_family(name: str | None) -> str:
    """Return a display family such as ``read`` or ``generic``."""

    if not name:
        return "generic"
    key = normalize_tool_name(name)
    if key in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[key]
    return _FAMILY_ALIASES.get(key.replace("_", ""), "generic")


def tool_name_from_text(text: str) -> str | None:
    """Extract the wire tool name from a formatted call/result first line."""

    if not text:
        return None
    first = text.splitlines()[0].strip()
    if not first:
        return None
    token = first.split()[0].rstrip("·").strip()
    return token or None


def tool_label(name: str | None, *, kind: str = "tool") -> str:
    """Return the specialized header title for a tool call or result."""

    family = tool_family(name)
    if family in _FAMILY_LABEL:
        return _FAMILY_LABEL[family]
    if family == "shell" and name:
        return _title_name(name)
    if name:
        return _title_name(name)
    return "Result" if kind == "tool_result" else "Tool"


def tool_styles(name: str | None) -> tuple[str, str] | None:
    """Return ``(label_style, bar_color)`` for a known family, else None."""

    family = tool_family(name)
    label_style = _FAMILY_LABEL_STYLE.get(family)
    bar = _FAMILY_BAR.get(family)
    if not label_style or not bar:
        return None
    return label_style, bar


def strip_tool_name(line: str, name: str | None) -> str:
    """Remove a leading wire tool name from a summary line."""

    if not name or not line:
        return line
    prefixes = (f"{name}  ", f"{name} · ", f"{name} ", name)
    for prefix in prefixes:
        if line.startswith(prefix):
            return line[len(prefix) :].lstrip("· ").strip()
    lowered = line.lower()
    key = name.lower()
    prefixes = (f"{key}  ", f"{key} · ", f"{key} ", key)
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return line[len(prefix) :].lstrip("· ").strip()
    return line


def format_tool_call_text(
    name: str,
    arguments: str | None,
    extras: object = None,
) -> str:
    """Human-readable tool-call body. First line always starts with ``name``."""

    parsed = _parse_object(arguments)
    family = tool_family(name)
    headline, extra_lines = _call_details(family, parsed, arguments)
    lines = [f"{name}  {headline}" if headline else name]
    lines.extend(extra_lines)
    if family == "generic" and extras:
        dumped = _pretty_value(extras)
        if dumped:
            lines.append(dumped)
    return "\n".join(lines)


def format_tool_result_text(
    tool_name: str | None,
    *,
    is_error: bool,
    message: str = "",
    display: str = "",
    output: str = "",
    extras: object = None,
) -> str:
    """Human-readable tool-result body. First line starts with the tool name."""

    name = tool_name or "Tool"
    family = tool_family(tool_name)
    outcome = "failed" if is_error else "succeeded"
    headline = _result_headline(name, message, display, outcome)
    lines = [f"{name}  {headline}" if headline else name]

    for part in (display, output):
        if not part:
            continue
        if _is_redundant_part(part, name, headline):
            continue
        lines.append(part)

    if message and message.strip().lower() not in _BORING_MESSAGES:
        first = message.splitlines()[0].strip()
        if first and first != headline and message not in lines:
            lines.append(message)

    if family == "generic" and extras:
        dumped = _pretty_value(extras)
        if dumped:
            lines.append(dumped)

    if len(lines) == 1 and not headline:
        lines.append("(no visible output)")
    return "\n".join(lines)


def _title_name(name: str) -> str:
    token = name.strip().split(".")[-1]
    if not token:
        return "Tool"
    return token.replace("_", " ").replace("-", " ").title()


def _parse_object(value: str | None) -> dict[str, Any] | None:
    if not value or not value.strip():
        return None
    try:
        parsed = orjson.loads(value)
    except orjson.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _pretty_value(value: object) -> str:
    try:
        return orjson.dumps(
            value,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
            default=lambda item: str(item),
        ).decode("utf-8")
    except (TypeError, ValueError):
        return str(value)


def _one_line(value: object, limit: int = 72) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    return f"{text[: limit - 3]}..."


def _short_path(value: object) -> str:
    if isinstance(value, list | tuple):
        if not value:
            return ""
        if len(value) == 1:
            return str(value[0])
        return f"{value[0]} +{len(value) - 1}"
    return str(value) if value is not None else ""


def _first(parsed: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if key in parsed and parsed[key] not in (None, ""):
            return parsed[key]
    return None


def _todo_marker(status: object) -> str:
    key = str(status or "").lower()
    if key in {"done", "completed"}:
        return "[x]"
    if key in {"in_progress", "in-progress", "doing"}:
        return "[>]"
    return "[ ]"


def _format_todo_item(item: object, depth: int = 0) -> list[str]:
    indent = "  " * depth
    if isinstance(item, str):
        return [f"{indent}[ ] {item}"]
    if not isinstance(item, Mapping):
        return [f"{indent}{item}"]
    title = item.get("content") or item.get("title") or item.get("name") or ""
    note = f" — {item['notes']}" if item.get("notes") else ""
    lines = [f"{indent}{_todo_marker(item.get('status'))} {title}{note}".rstrip()]
    children = item.get("children") or item.get("items") or ()
    if isinstance(children, Sequence) and not isinstance(children, str | bytes):
        for child in children:
            lines.extend(_format_todo_item(child, depth + 1))
    return lines


def _call_details(
    family: str,
    parsed: dict[str, Any] | None,
    raw_arguments: str | None,
) -> tuple[str, list[str]]:
    if parsed is None:
        leftover = (raw_arguments or "").strip()
        return leftover, []

    if family == "read":
        path = _short_path(_first(parsed, "file_path", "path"))
        bits: list[str] = []
        offset = parsed.get("offset")
        if offset not in (None, 0, 1):
            bits.append(f"offset {offset}")
        limit = _first(parsed, "limit", "n_lines")
        if limit not in (None, ""):
            bits.append(f"{limit} lines")
        if parsed.get("glob"):
            bits.append("glob")
        return path, ([" · ".join(bits)] if bits else [])

    if family == "grep":
        pattern = str(_first(parsed, "pattern") or "")
        bits = []
        path = parsed.get("path")
        if path not in (None, "", "."):
            bits.append(str(path))
        include = _first(parsed, "include", "glob")
        if include:
            bits.append(str(include))
        mode = parsed.get("output_mode")
        if mode not in (None, "", "files_with_matches"):
            bits.append(str(mode))
        if parsed.get("-i") or parsed.get("case_insensitive"):
            bits.append("ignore-case")
        return pattern, ([" · ".join(bits)] if bits else [])

    if family == "glob":
        pattern = str(_first(parsed, "pattern") or "")
        path = parsed.get("path")
        extra = [str(path)] if path not in (None, "", ".") else []
        return pattern, extra

    if family == "write":
        path = _short_path(_first(parsed, "file_path", "path"))
        extra = []
        mode = parsed.get("mode")
        if mode not in (None, "", "overwrite"):
            extra.append(str(mode))
        return path, extra

    if family == "edit":
        path = _short_path(_first(parsed, "file_path", "path"))
        extra = []
        edits = parsed.get("edits")
        old = _first(parsed, "old", "old_string")
        new = _first(parsed, "new", "new_string")
        if isinstance(edits, Sequence) and not isinstance(edits, str | bytes) and edits:
            extra.append(f"{len(edits)} edit" + ("s" if len(edits) != 1 else ""))
            first = edits[0]
            if isinstance(first, Mapping):
                old = first.get("old") or first.get("old_string") or old
        if old:
            extra.append(_one_line(old, 60))
        if new and not old:
            extra.append(_one_line(new, 60))
        return path, extra

    if family == "shell":
        command = str(_first(parsed, "command", "cmd") or "")
        extra = []
        cwd = _first(parsed, "working_directory", "cwd", "workdir")
        if cwd:
            extra.append(str(cwd))
        return command, extra

    if family == "python":
        code = str(_first(parsed, "code", "file") or "")
        first_line = next((line.strip() for line in code.splitlines() if line.strip()), "")
        return _one_line(first_line or code, 72), []

    if family == "todo":
        todos = parsed.get("todos") or parsed.get("updates") or parsed.get("items")
        if isinstance(todos, Sequence) and not isinstance(todos, str | bytes):
            lines: list[str] = []
            for item in todos:
                lines.extend(_format_todo_item(item))
            headline = f"{len(todos)} item" + ("s" if len(todos) != 1 else "")
            return headline, lines
        title = _first(parsed, "title", "content")
        if title:
            status = parsed.get("status")
            rename = parsed.get("rename_to")
            extra = []
            if rename:
                extra.append(f"rename → {rename}")
            return f"{_todo_marker(status)} {title}", extra
        return "", [_pretty_value(parsed)]

    if family == "search":
        return str(_first(parsed, "query", "search_term") or ""), []

    if family == "fetch":
        url = _first(parsed, "url")
        if not url:
            urls = parsed.get("urls")
            if isinstance(urls, Sequence) and urls:
                first = urls[0]
                if isinstance(first, Mapping):
                    url = first.get("url") or first.get("href")
                else:
                    url = first
        return str(url or ""), []

    if family == "agent":
        return _one_line(_first(parsed, "description", "prompt") or "", 72), []

    dumped = _pretty_value(parsed)
    return "", [dumped] if dumped else []


def _result_headline(name: str, message: str, display: str, outcome: str) -> str:
    if display:
        first = display.splitlines()[0].strip()
        stripped = strip_tool_name(first, name)
        if stripped:
            return stripped
        if first:
            return first
    if message:
        first = message.splitlines()[0].strip()
        if first.lower() not in _BORING_MESSAGES:
            stripped = strip_tool_name(first, name)
            return stripped or first
    return outcome


def _is_redundant_part(part: str, name: str, headline: str) -> bool:
    text = part.strip()
    if not text:
        return True
    if text == headline:
        return True
    if strip_tool_name(text, name) == headline:
        return True
    first = text.splitlines()[0].strip()
    if len(text.splitlines()) == 1 and strip_tool_name(first, name) == headline:
        return True
    return False

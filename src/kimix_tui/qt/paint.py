"""Qt painting helpers for transcript rows (no Rich/Textual)."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextDocument
from PySide6.QtWidgets import QApplication

from kimix_tui.qt.theme import COLORS
from kimix_tui.transcript_layout import (
    COPY_SUFFIX,
    bar_color_name,
    cell_len,
    compact_summary,
    copy_hit_start,
    expanded_body,
    has_markdown_hints,
    is_compact_record,
    is_dialogue_record,
    record_label,
    tool_name_for,
)

COLOR_HEX: dict[str, str] = {
    "cyan": COLORS["cyan"],
    "green": COLORS["green"],
    "red": COLORS["red"],
    "yellow": COLORS["yellow"],
    "blue": COLORS["blue"],
    "magenta": COLORS["magenta"],
    "muted": COLORS["muted"],
}


@dataclass(frozen=True, slots=True)
class RecordLayout:
    """Plain-text layout of one transcript row, used by tests and the delegate."""

    header: str
    body: str
    compact: bool
    label: str
    bar_color: str
    italic_body: bool
    markdown: bool
    copy_suffix: str = COPY_SUFFIX


def qcolor(name: str) -> QColor:
    """Map a layout color name to a QColor."""

    return QColor(COLOR_HEX.get(name, COLORS["muted"]))


def layout_record(
    kind: str,
    text: str,
    *,
    width: int,
    expanded: bool = False,
) -> RecordLayout:
    """Build the header/body strings that the TUI painter used to emit as strips."""

    label = record_label(kind, text)
    tool_name = tool_name_for(kind, text)
    compact = is_compact_record(kind, expanded=expanded)
    bar = bar_color_name(kind, text)
    header_width = max(0, width - cell_len("▌ ") - cell_len(COPY_SUFFIX))
    if compact:
        prefix = f"▸ {label}  "
        summary_width = max(0, header_width - cell_len(prefix))
        header = prefix + compact_summary(text, summary_width, lead_name=tool_name)
        return RecordLayout(
            header=header,
            body="",
            compact=True,
            label=label,
            bar_color=bar,
            italic_body=False,
            markdown=False,
        )
    if not is_dialogue_record(kind):
        return RecordLayout(
            header=f"▾ {label}",
            body=expanded_body(kind, text, tool_name),
            compact=False,
            label=label,
            bar_color=bar,
            italic_body=kind == "thinking",
            markdown=False,
        )
    return RecordLayout(
        header=label,
        body=text,
        compact=False,
        label=label,
        bar_color=bar,
        italic_body=False,
        markdown=kind == "assistant" and has_markdown_hints(text),
    )


def markdown_plain_text(text: str) -> str:
    """Render CommonMark to plain text using Qt's document engine."""

    if QApplication.instance() is None:
        return text
    document = QTextDocument()
    document.setMarkdown(text)
    return document.toPlainText()


def layout_plain(kind: str, text: str, *, width: int, expanded: bool = False) -> str:
    """Return the full visible text of a laid-out record."""

    layout = layout_record(kind, text, width=width, expanded=expanded)
    header = f"{layout.header}{layout.copy_suffix}"
    if layout.compact:
        return header
    body = markdown_plain_text(layout.body) if layout.markdown else layout.body
    return f"{header}\n{body}"


def copy_region_left(width: int) -> int:
    """Pixel-agnostic copy hit column used by tests; delegate maps it to pixels."""

    return copy_hit_start(width)


def header_font(*, bold: bool = True, italic: bool = False) -> QFont:
    font = QFont(QApplication.font()) if QApplication.instance() else QFont()
    font.setBold(bold)
    font.setItalic(italic)
    return font


def body_font(*, italic: bool = False, monospace: bool = False) -> QFont:
    font = QFont(QApplication.font()) if QApplication.instance() else QFont()
    font.setItalic(italic)
    if monospace:
        font.setFamily("Cascadia Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def alignment_left() -> Qt.AlignmentFlag:
    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

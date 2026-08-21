"""Dark, compact desktop palette and stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

COLORS = {
    "bg": "#0f1115",
    "surface": "#161a22",
    "panel": "#1c212c",
    "boost": "#252b38",
    "border": "#2c3342",
    "text": "#e8edf5",
    "muted": "#8b95a8",
    "accent": "#5eead4",
    "cyan": "#22d3ee",
    "green": "#4ade80",
    "red": "#f87171",
    "yellow": "#facc15",
    "blue": "#60a5fa",
    "magenta": "#e879f9",
    "error": "#f87171",
}

APP_STYLE = f"""
QWidget {{
    background: {COLORS["bg"]};
    color: {COLORS["text"]};
    font-size: 13px;
}}
QMainWindow, QDialog, QStackedWidget {{
    background: {COLORS["bg"]};
}}
QLabel#home-title, QLabel#chat-title, QLabel#detail-overline,
QLabel#settings-title, QLabel#config-details-title {{
    color: {COLORS["accent"]};
    font-weight: 600;
    letter-spacing: 0.4px;
}}
QLabel#home-path, QLabel#home-model, QLabel#status, QLabel#context,
QLabel#history-info, QLabel#session-count, QLabel#selection-count,
QLabel#home-status, QLabel#detail-state, QLabel#settings-scope {{
    color: {COLORS["muted"]};
}}
QLineEdit, QPlainTextEdit, QSpinBox {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: {COLORS["boost"]};
}}
QLineEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {COLORS["accent"]};
}}
QPushButton {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 6px 12px;
    color: {COLORS["text"]};
}}
QPushButton:hover {{
    background: {COLORS["boost"]};
}}
QPushButton:disabled {{
    color: {COLORS["muted"]};
}}
QPushButton#start-new-session, QPushButton#open-session,
QPushButton#apply-settings, QPushButton#approve {{
    background: {COLORS["accent"]};
    color: #042f2e;
    border: none;
    font-weight: 600;
}}
QPushButton#start-new-session:hover, QPushButton#open-session:hover,
QPushButton#apply-settings:hover, QPushButton#approve:hover {{
    background: #2dd4bf;
}}
QPushButton#delete-sessions, QPushButton#confirm-delete, QPushButton#reject {{
    background: #3f1d22;
    color: {COLORS["red"]};
    border: 1px solid #7f1d1d;
}}
QListWidget, QListView {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    outline: none;
    padding: 4px;
}}
QListWidget::item, QListView::item {{
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {COLORS["boost"]};
    border-left: 3px solid {COLORS["accent"]};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 4px 2px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS["border"]};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background: {COLORS["border"]};
}}
QFrame#home-toolbar, QFrame#chat-toolbar, QFrame#history-toolbar,
QFrame#chat-footer {{
    background: {COLORS["surface"]};
    border: none;
}}
QFrame#session-detail {{
    background: {COLORS["panel"]};
    border-radius: 12px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the dark palette and stylesheet to ``app``."""

    font = QFont("Segoe UI")
    font.setPixelSize(13)
    app.setFont(font)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["panel"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["panel"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["boost"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS["muted"]))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLE)

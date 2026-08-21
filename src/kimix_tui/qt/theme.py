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
    background: transparent;
    color: {COLORS["text"]};
    font-size: 13px;
}}
QMainWindow, QDialog, QStackedWidget,
QWidget#home-view, QWidget#chat-view {{
    background: {COLORS["bg"]};
}}
QLabel {{
    background: transparent;
}}
QMenu, QToolTip {{
    background: {COLORS["panel"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
}}
QLabel#home-title, QLabel#chat-title, QLabel#detail-overline,
QLabel#settings-title, QLabel#config-details-title {{
    color: {COLORS["accent"]};
    font-weight: 600;
    letter-spacing: 0.4px;
}}
QLabel#home-title {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0;
}}
QLabel#history-title, QLabel#selection-count {{
    color: {COLORS["text"]};
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0;
}}
QLabel#detail-overline {{
    color: {COLORS["muted"]};
    font-size: 11px;
    letter-spacing: 0.8px;
}}
QLabel#detail-title {{
    font-size: 18px;
    font-weight: 600;
}}
QLabel#home-path, QLabel#home-model, QLabel#status, QLabel#context,
QLabel#history-info, QLabel#session-count,
QLabel#home-status, QLabel#detail-state, QLabel#settings-scope,
QLabel#session-meta, QLabel#detail-key {{
    color: {COLORS["muted"]};
}}
QLabel#session-title {{
    font-weight: 600;
}}
QLabel#session-meta {{
    font-size: 12px;
}}
QLabel#session-badge {{
    font-size: 11px;
    font-weight: 600;
    border-radius: 8px;
    padding: 2px 8px;
}}
QLabel#session-badge[kind="last"] {{
    background: {COLORS["accent"]};
    color: #042f2e;
}}
QLabel#session-badge[kind="archived"] {{
    background: {COLORS["boost"]};
    color: {COLORS["muted"]};
}}
QWidget#session-row {{
    background: transparent;
}}
QCheckBox#session-check {{
    background: transparent;
    spacing: 0;
    border: none;
}}
QWidget#history-header {{
    background: transparent;
}}
QPushButton#select-shown {{
    background: transparent;
    border: none;
    color: {COLORS["muted"]};
    padding: 4px 8px;
}}
QPushButton#select-shown:hover {{
    color: {COLORS["text"]};
    background: {COLORS["boost"]};
}}
QPushButton#select-shown:disabled {{
    background: transparent;
}}
QWidget#home-view QPushButton#delete-sessions {{
    padding: 4px 10px;
}}
QLineEdit, QPlainTextEdit, QSpinBox, QTextEdit {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: {COLORS["boost"]};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {COLORS["accent"]};
}}
QPlainTextEdit#prompt {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
    padding: 8px 12px;
}}
QPlainTextEdit#prompt:focus {{
    border: 1px solid {COLORS["accent"]};
}}
QPushButton#expand-prompt, QPushButton#close-composer-pad {{
    background: transparent;
    border: none;
    padding: 0;
    min-width: 22px;
    border-radius: 6px;
    color: {COLORS["muted"]};
}}
QPushButton#expand-prompt:hover, QPushButton#close-composer-pad:hover {{
    background: {COLORS["boost"]};
    color: {COLORS["text"]};
}}
QDialog#composer-pad {{
    background: transparent;
}}
QFrame#composer-pad-card {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 16px;
}}
QFrame#composer-pad-header {{
    background: transparent;
}}
QLabel#composer-pad-title {{
    font-size: 16px;
    font-weight: 600;
}}
QLabel#composer-pad-hint, QLabel#composer-pad-count {{
    color: {COLORS["muted"]};
    font-size: 12px;
}}
QPlainTextEdit#prompt-pad {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 14px;
}}
QPlainTextEdit#prompt-pad:focus {{
    border: 1px solid {COLORS["accent"]};
}}
QFrame#composer-dock {{
    background: {COLORS["surface"]};
}}
QPushButton#send-prompt, QPushButton#cancel-prompt,
QPushButton#send-pad, QPushButton#cancel-pad {{
    min-width: 64px;
    padding: 4px 12px;
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
QPushButton#apply-settings, QPushButton#approve, QPushButton#send-prompt,
QPushButton#send-pad {{
    background: {COLORS["accent"]};
    color: #042f2e;
    border: none;
    font-weight: 600;
}}
QPushButton#start-new-session:hover, QPushButton#open-session:hover,
QPushButton#apply-settings:hover, QPushButton#approve:hover,
QPushButton#send-prompt:hover, QPushButton#send-pad:hover {{
    background: #2dd4bf;
}}
QPushButton#send-prompt:disabled, QPushButton#send-pad:disabled {{
    background: {COLORS["panel"]};
    color: {COLORS["muted"]};
    border: 1px solid {COLORS["border"]};
}}
QPushButton#delete-sessions, QPushButton#confirm-delete, QPushButton#reject,
QPushButton#cancel-prompt, QPushButton#cancel-pad {{
    background: #3f1d22;
    color: {COLORS["red"]};
    border: 1px solid #7f1d1d;
}}
QPushButton#cancel-prompt:disabled, QPushButton#cancel-pad:disabled {{
    background: {COLORS["panel"]};
    color: {COLORS["muted"]};
    border: 1px solid {COLORS["border"]};
}}
QListWidget, QListView {{
    background: transparent;
    border: none;
    outline: none;
    padding: 4px;
}}
QListView#transcript {{
    background: transparent;
    border: none;
    padding: 0;
}}
QListView#transcript::item {{
    background: transparent;
    border: none;
    padding: 0;
}}
QListWidget::item, QListView::item {{
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item:selected, QListView::item:selected {{
    background: {COLORS["boost"]};
    border-left: 3px solid {COLORS["accent"]};
}}
QListWidget#session-list {{
    background: transparent;
    border: none;
    outline: none;
    padding: 2px 0;
}}
QListWidget#session-list::item {{
    padding: 0;
    margin: 1px 0;
    border: none;
    background: transparent;
}}
QListWidget#session-list::item:selected {{
    background: transparent;
    border: none;
}}
QListWidget#session-list::item:hover {{
    background: transparent;
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
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
}}
QLabel#toast {{
    background: {COLORS["boost"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 18px;
    padding: 10px 18px;
    font-size: 13px;
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

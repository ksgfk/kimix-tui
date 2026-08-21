from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

from kimix_tui.app import KimixTuiApp
from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.home_view import HomeView


def find[T: QWidget](root: QWidget, name: str, cls: type[T] = QWidget) -> T:
    widget = root.findChild(cls, name)
    if widget is None:
        raise AssertionError(f"widget #{name} not found under {type(root).__name__}")
    return widget


def widget_text(root: QWidget, name: str) -> str:
    return find(root, name).text()


def wait_idle(qtbot, app: KimixTuiApp, timeout: int = 10_000) -> None:
    qtbot.waitUntil(lambda: app.bridge.is_idle(), timeout=timeout)
    for _ in range(8):
        QApplication.processEvents()
        if app.bridge.is_idle():
            return
        qtbot.wait(20)
    QApplication.processEvents()


def wait_chat_ready(qtbot, app: KimixTuiApp, timeout: int = 10_000) -> ChatView:
    qtbot.waitUntil(
        lambda: isinstance(app.screen, ChatView) and app.screen.prompt_enabled,
        timeout=timeout,
    )
    wait_idle(qtbot, app, timeout=timeout)
    chat = app.screen
    assert isinstance(chat, ChatView)
    return chat


def wait_home(qtbot, app: KimixTuiApp, timeout: int = 10_000) -> HomeView:
    qtbot.waitUntil(lambda: isinstance(app.screen, HomeView), timeout=timeout)
    home = app.screen
    assert isinstance(home, HomeView)

    def _loaded() -> bool:
        status = home.findChild(QWidget, "home-status")
        if status is None:
            return False
        if not status.isVisible():
            return True
        return "Loading sessions" not in status.text()

    qtbot.waitUntil(_loaded, timeout=timeout)
    wait_idle(qtbot, app, timeout=timeout)
    return home


def launch_app(qtbot, app: KimixTuiApp, size: tuple[int, int] = (1100, 720)):
    window = app.create_window()
    window.resize(*size)
    qtbot.addWidget(window)
    window.show()
    wait_idle(qtbot, app)
    return window

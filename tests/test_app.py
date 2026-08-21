from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from kimi_agent_sdk import (
    ApprovalRequest,
    BriefDisplayBlock,
    StatusUpdate,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolReturnValue,
    TurnEnd,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton
from qtutil import find, launch_app, wait_chat_ready, wait_home, wait_idle, widget_text

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.history import HistoryBlock, SessionHistory, Timeline
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.composer import Composer
from kimix_tui.qt.home_view import HomeView
from kimix_tui.qt.request_dialogs import ApprovalDialog


def _fake_timeline(turn_count: int) -> Timeline:
    return Timeline.from_turn_blocks(
        [
            [HistoryBlock("user", f"q{index}"), HistoryBlock("assistant", f"a{index}")]
            for index in range(turn_count)
        ]
    )


def _config_store(tmp_path: Path) -> LLMConfigStore:
    config_file = tmp_path / "provider.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 131_072,
                "url": "https://example.test/v1",
                "type": "openai_legacy",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    store = LLMConfigStore(
        tmp_path / "kimix-tui.json",
        session_file_resolver=lambda _work_dir, session_id: (
            tmp_path / "sessions" / session_id / "kimix-tui.json"
        ),
    )
    store.set_default(tmp_path, inspect_llm_config(config_file))
    return store


class FakeSession:
    def __init__(self, messages: list[object], *, hang_prompt: bool = False) -> None:
        self.id = "fake-session"
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self._messages = messages
        self._hang_prompt = hang_prompt
        self._hang = asyncio.Event()
        self.prompt_started = asyncio.Event()
        self.prompts: list[str] = []
        self.cancelled = False
        self.closed = False

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        self.prompts.append(user_input)
        assert merge_wire_messages is False
        self.prompt_started.set()
        for message in self._messages:
            yield message
        if self._hang_prompt:
            await self._hang.wait()

    def cancel(self) -> None:
        self.cancelled = True
        self._hang.set()

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self._hang.set()
        self.closed = True


def _submit(qtbot, chat: ChatView, text: str) -> None:
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText(text)
    prompt.submitted.emit(text)


def test_keyboard_submit_streams_into_transcript(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="hello "), TextPart(text="world"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "hi")
    qtbot.waitUntil(lambda: session.prompts == ["hi"], timeout=10_000)
    wait_idle(qtbot, app)

    records = [(record.kind, record.text) for record in chat.transcript.records]
    assert session.prompts == ["hi"]
    assert ("user", "hi") in records
    assert ("assistant", "hello world") in records


def test_keyboard_submit_sends_multiline_prompt(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    prompt.setFocus()
    prompt.setPlainText("hi\nthere")
    prompt.submitted.emit("hi\nthere")
    qtbot.waitUntil(lambda: session.prompts == ["hi\nthere"], timeout=10_000)
    wait_idle(qtbot, app)

    records = [(record.kind, record.text) for record in chat.transcript.records]
    assert session.prompts == ["hi\nthere"]
    assert ("user", "hi\nthere") in records
    assert prompt.text == ""


def test_chat_prompt_stays_within_screen(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app, size=(800, 600))
    chat = wait_chat_ready(qtbot, app)
    prompt = chat.prompt
    assert prompt.x() >= 0
    assert prompt.geometry().right() <= chat.width()
    prompt.setFocus()
    for _ in range(5):
        qtbot.keyClick(prompt, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert prompt.x() >= 0
    assert prompt.geometry().right() <= chat.width()


def test_send_button_submits_prompt(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    send = find(chat, "send-prompt", QPushButton)
    cancel = find(chat, "cancel-prompt", QPushButton)
    assert send.text() == "Send"
    assert cancel.text() == "Cancel"
    assert send.isVisible() is True
    assert send.isEnabled() is False
    assert cancel.isVisible() is False
    assert send.x() >= chat.prompt.geometry().right()
    assert chat.prompt.height() == Composer.MIN_HEIGHT
    assert send.height() == Composer.ACTION_HEIGHT
    assert "Ctrl+Enter" in chat.prompt.placeholderText()

    chat.prompt.setPlainText("hello from send")
    assert send.isEnabled() is True
    send.click()
    qtbot.waitUntil(lambda: session.prompts == ["hello from send"], timeout=10_000)
    wait_idle(qtbot, app)
    assert chat.prompt.text == ""
    assert send.isEnabled() is False


def test_expand_prompt_opens_pad_and_sends(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    find(chat.prompt, "expand-prompt", QPushButton).click()
    pad = find(chat, "composer-pad", QDialog)
    qtbot.waitUntil(pad.isVisible, timeout=5_000)
    editor = find(pad, "prompt-pad", QPlainTextEdit)
    long_text = "paste\n" * 40 + "done"
    editor.setPlainText(long_text)
    find(pad, "send-pad", QPushButton).click()
    qtbot.waitUntil(lambda: session.prompts == [long_text], timeout=10_000)
    wait_idle(qtbot, app)
    assert chat.prompt.text == ""
    assert pad.isVisible() is False


def test_expand_prompt_keeps_draft_on_close(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    find(chat.prompt, "expand-prompt", QPushButton).click()
    pad = find(chat, "composer-pad", QDialog)
    qtbot.waitUntil(pad.isVisible, timeout=5_000)
    find(pad, "prompt-pad", QPlainTextEdit).setPlainText("keep this draft")
    find(pad, "close-composer-pad", QPushButton).click()
    qtbot.waitUntil(lambda: not pad.isVisible(), timeout=5_000)
    assert chat.prompt.text == "keep this draft"
    assert session.prompts == []


def test_cancel_button_stops_generation(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    cancel = find(chat, "cancel-prompt", QPushButton)
    qtbot.waitUntil(lambda: cancel.isVisible() and cancel.isEnabled(), timeout=10_000)
    assert find(chat, "send-prompt", QPushButton).isVisible() is False
    cancel.click()
    wait_idle(qtbot, app)
    assert session.cancelled is True
    assert chat.busy is False
    assert chat.prompt_enabled is True


def test_chat_live_stream_keeps_timeline_and_appends_at_tail(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        turns = [
            [HistoryBlock("user", f"q{index}"), HistoryBlock("assistant", f"a{index}")]
            for index in range(30)
        ]
        turns[0].insert(1, HistoryBlock("tool", "Read file\nPath: a.py"))
        return Timeline.from_turn_blocks(turns)

    monkeypatch.setattr("kimix_tui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([TextPart(text="hel"), TextPart(text="lo"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    texts = [record.text for record in chat.transcript.records]
    assert "q0" in texts
    assert "q29" in texts
    assert any(record.kind == "tool" and "Read file" in record.text for record in chat.transcript.records)
    chat.transcript.jump_to_turn(0)
    assert chat.transcript.pinned_to_latest is False

    _submit(qtbot, chat, "hi")
    qtbot.waitUntil(lambda: session.prompts == ["hi"], timeout=10_000)
    wait_idle(qtbot, app)

    texts = [record.text for record in chat.transcript.records]
    assert "q0" in texts
    assert "q29" in texts
    assert chat.transcript.records[-1].text == "hello"
    assert chat.transcript.pinned_to_latest is False
    assert chat.transcript.viewport_turn() == 0


def test_chat_shows_streamed_tool_call_and_detailed_result(qtbot, tmp_path: Path) -> None:
    session = FakeSession(
        [
            ToolCall(
                id="call-1",
                function=ToolCall.FunctionBody(name="read", arguments=""),
            ),
            ToolCallPart(arguments_part='{"path":'),
            ToolCallPart(arguments_part='"a.py"}'),
            ToolResult(
                tool_call_id="call-1",
                return_value=ToolReturnValue(
                    is_error=False,
                    output="file contents",
                    message="success",
                    display=[BriefDisplayBlock(text="read a.py")],
                    extras=None,
                ),
            ),
            TurnEnd(),
        ]
    )

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: session.prompts == ["go"], timeout=10_000)
    wait_idle(qtbot, app)
    tool = next(record for record in chat.transcript.records if record.kind == "tool")
    result = next(record for record in chat.transcript.records if record.kind == "tool_result")
    assert "read" in tool.text
    assert "a.py" in tool.text
    assert "Call ID" not in tool.text
    assert "Arguments:" not in tool.text
    assert "a.py" in result.text
    assert "file contents" in result.text
    assert "Call ID" not in result.text
    assert "Message:" not in result.text


def test_chat_keeps_detailed_status_after_turn_finishes(qtbot, tmp_path: Path) -> None:
    session = FakeSession(
        [
            StatusUpdate(
                context_tokens=2_000,
                max_context_tokens=20_000,
                context_usage=0.1,
            ),
            StatusUpdate(
                token_usage=TokenUsage(
                    input_other=100,
                    input_cache_read=800,
                    input_cache_creation=50,
                    output=75,
                ),
                message_id="msg-1",
            ),
            TurnEnd(),
        ]
    )

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: session.prompts == ["go"], timeout=10_000)
    wait_idle(qtbot, app)
    session_line = widget_text(chat, "status")
    context = widget_text(chat, "context")
    assert "session fake-session" in session_line
    assert "context" not in session_line
    assert "context 2,000/20,000" in context
    assert "tokens in 950" in context
    assert "cache read 800" in context
    assert "out 75" in context
    assert "message" not in context


def test_resumed_session_shows_recent_history(qtbot, tmp_path: Path) -> None:
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    async def history_loader(_work_dir: Path, session_id: str) -> SessionHistory:
        assert session_id == "fake-session"
        return SessionHistory(
            blocks=[
                HistoryBlock("user", "fix login"),
                HistoryBlock("assistant", "Check the redirect."),
            ],
            omitted_turns=1,
        )

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        history_loader=history_loader,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    records = [(record.kind, record.text) for record in chat.transcript.records]
    assert ("system", "Session: fake-session") in records
    assert (
        "system",
        "Showing last 1 turns (1 earlier omitted)",
    ) in records
    assert ("user", "fix login") in records
    assert ("assistant", "Check the redirect.") in records


def test_chat_chrome_keeps_history_toolbar(qtbot, tmp_path: Path) -> None:
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    toolbar = find(chat, "history-toolbar")
    assert toolbar.isVisible()
    assert widget_text(chat, "history-info").startswith("History ·")
    assert find(chat, "open-settings", QPushButton).isVisible()
    assert find(chat, "leave-session", QPushButton).text() == "Home"
    assert find(chat, "load-older", QPushButton).text() == "←"
    status = find(chat, "status", QLabel)
    context = find(chat, "context", QLabel)
    assert status.parent().objectName() == "chat-toolbar"
    assert context.parent().objectName() == "composer-dock"
    assert "connecting" in status.text() or "session" in status.text().casefold()
    assert "context" in context.text() or "ready" in context.text().casefold()


def test_chat_history_toolbar_stays_visible_for_short_sessions(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(2)

    monkeypatch.setattr("kimix_tui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    assert find(chat, "history-toolbar").isVisible()
    assert widget_text(chat, "history-info").startswith("History · Turn 2 of 2")
    assert find(chat, "load-older", QPushButton).isEnabled()
    assert find(chat, "history-turn").isEnabled()


def test_chat_loads_older_history_pages_without_losing_latest_rows(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(130)

    monkeypatch.setattr("kimix_tui.qt.bridge.create_timeline", _timeline)
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app, size=(640, 700))
    chat = wait_chat_ready(qtbot, app)
    assert widget_text(chat, "history-info").startswith("History · Turn 130 of 130")
    turn_input = find(chat, "history-turn")
    assert turn_input.geometry().right() <= chat.width()
    assert turn_input.placeholderText() == "Turn 1-130"
    texts = [record.text for record in chat.transcript.records]
    assert "q0" in texts
    assert "q129" in texts

    chat.load_older_history()
    wait_idle(qtbot, app)
    qtbot.waitUntil(lambda: chat.transcript.pinned_to_latest is False, timeout=10_000)
    older_info = widget_text(chat, "history-info")
    assert older_info.startswith("History · Turn ")
    assert older_info.endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False
    assert any(record.text == "q128" for record in chat.transcript.records)
    assert any(record.text == "q129" for record in chat.transcript.records)

    qtbot.keyClick(window, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
    wait_idle(qtbot, app)
    assert widget_text(chat, "history-info").endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False

    qtbot.keyClick(window, Qt.Key.Key_F3)
    assert turn_input.hasFocus()
    turn_input.setText("5")
    turn_input.returnPressed.emit()
    wait_idle(qtbot, app)
    qtbot.waitUntil(
        lambda: any(record.text == "q4" for record in chat.transcript.records),
        timeout=10_000,
    )

    texts = [record.text for record in chat.transcript.records]
    assert "q4" in texts
    assert "q129" not in texts
    jumped_info = widget_text(chat, "history-info")
    assert jumped_info.startswith("History · Turn ")
    assert jumped_info.endswith(" of 130")
    assert "q4" in chat.transcript.visible_text()

    find(chat, "load-newer", QPushButton).click()
    wait_idle(qtbot, app)
    newer_info = widget_text(chat, "history-info")
    assert newer_info.startswith("History · Turn ")
    assert newer_info.endswith(" of 130")
    assert chat.transcript.pinned_to_latest is False

    find(chat, "jump-latest", QPushButton).click()
    wait_idle(qtbot, app)
    assert widget_text(chat, "history-info").startswith("History · Turn 130 of 130")
    assert chat.transcript.pinned_to_latest is True
    texts = [record.text for record in chat.transcript.records]
    assert "q129" in texts
    assert "q0" not in texts


def test_approval_is_resolved_from_keyboard(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TextPart(text="done"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    qtbot.keyClick(app.screen, Qt.Key.Key_A)
    wait_idle(qtbot, app)

    assert getattr(approval, "resolved", True)
    assert app.screen is chat
    assert any(record.text == "done" for record in chat.transcript.records)
    assert any("Approval decision: approve" in record.text for record in chat.transcript.records)


def test_approval_is_resolved_by_clicking_approve(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    find(app.screen, "approve").click()
    wait_idle(qtbot, app)

    assert any("Approval decision: approve" in record.text for record in chat.transcript.records)
    assert app.screen is chat


def test_escape_rejects_modal_without_leaving_chat(qtbot, tmp_path: Path) -> None:
    approval = ApprovalRequest(
        id="approval-1",
        tool_call_id="call-1",
        sender="write",
        action="write file",
        description="Write a.py",
    )
    session = FakeSession([approval, TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "go")
    qtbot.waitUntil(lambda: isinstance(app.screen, ApprovalDialog), timeout=10_000)
    qtbot.keyClick(app.screen, Qt.Key.Key_Escape)
    wait_idle(qtbot, app)

    assert any("Approval decision: reject" in record.text for record in chat.transcript.records)
    assert app.screen is chat
    assert chat.session_id == "fake-session"
    assert session.closed is False


def test_cancelling_prompt_unblocks_hung_generation(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    qtbot.keyClick(window, Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier)
    wait_idle(qtbot, app)

    assert session.prompts == ["cont"]
    assert session.cancelled is True
    assert chat.busy is False
    assert chat.prompt_enabled is True


def test_leave_during_running_prompt_returns_home_quietly(qtbot, tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    window = launch_app(qtbot, app)
    chat = wait_chat_ready(qtbot, app)
    _submit(qtbot, chat, "cont")
    qtbot.waitUntil(lambda: session.prompt_started.is_set(), timeout=10_000)
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    wait_home(qtbot, app)

    assert isinstance(app.screen, HomeView)
    assert window.chat is None
    assert chat.busy is False
    assert session.cancelled is True
    assert session.closed is True

from __future__ import annotations

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
from textual.widgets import Input, Static

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.history import HistoryBlock, SessionHistory
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.screens.chat import ChatScreen
from kimix_tui.screens.requests import ApprovalScreen


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
    def __init__(self, messages: list[object]) -> None:
        self.id = "fake-session"
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000,
            context_usage=0.1,
        )
        self._messages = messages
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
        for message in self._messages:
            yield message

    def cancel(self) -> None:
        self.cancelled = True

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_keyboard_submit_streams_into_transcript(tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="hello "), TextPart(text="world"), TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        prompt = chat.query_one("#prompt", Input)
        prompt.focus()
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()

        records = [(record.kind, record.text) for record in chat.transcript.records]
        assert session.prompts == ["hi"]
        assert ("user", "hi") in records
        assert ("assistant", "hello world") in records

    assert session.closed is True


@pytest.mark.asyncio
async def test_chat_shows_streamed_tool_call_and_detailed_result(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.query_one("#prompt", Input).focus()
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()

        tool = next(record for record in chat.transcript.records if record.kind == "tool")
        result = next(record for record in chat.transcript.records if record.kind == "tool_result")
        assert "read\nCall ID: call-1" in tool.text
        assert "Arguments:" in tool.text
        assert '"path": "a.py"' in tool.text
        assert "read · succeeded" in result.text
        assert "Message:\nsuccess" in result.text
        assert "Display:\nread a.py" in result.text
        assert "Output:\nfile contents" in result.text


@pytest.mark.asyncio
async def test_chat_keeps_detailed_status_after_turn_finishes(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.query_one("#prompt", Input).focus()
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()

        status = str(chat.query_one("#status", Static).content)
        assert "context 2,000/20,000" in status
        assert "tokens in 950" in status
        assert "cache read 800" in status
        assert "out 75" in status
        assert "message msg-1" in status


@pytest.mark.asyncio
async def test_resumed_session_shows_recent_history(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)):
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        records = [(record.kind, record.text) for record in chat.transcript.records]
        assert ("system", "Session: fake-session") in records
        assert (
            "system",
            "Showing last 1 turns (1 earlier omitted)",
        ) in records
        assert ("user", "fix login") in records
        assert ("assistant", "Check the redirect.") in records


@pytest.mark.asyncio
async def test_approval_is_resolved_from_keyboard(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.query_one("#prompt", Input).focus()
        await pilot.press("g", "o", "enter")
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("a")
        await app.workers.wait_for_complete()

        assert approval.resolved is True
        assert await approval.wait() == "approve"
        assert app.screen is chat
        assert any(record.text == "done" for record in chat.transcript.records)


@pytest.mark.asyncio
async def test_escape_rejects_modal_without_leaving_chat(tmp_path: Path) -> None:
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
    async with app.run_test(size=(100, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.query_one("#prompt", Input).focus()

        await pilot.press("g", "o", "enter")
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("escape")
        await app.workers.wait_for_complete()

        assert await approval.wait() == "reject"
        assert app.screen is chat
        assert chat.session is session
        assert session.closed is False

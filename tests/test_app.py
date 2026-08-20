from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
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
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header, Input, Static
from textual.worker import WorkerCancelled, WorkerFailed

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.history import HistoryBlock, SessionHistory, Timeline
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.screens.chat import ChatScreen
from kimix_tui.screens.home import HomeScreen
from kimix_tui.screens.requests import ApprovalScreen
from kimix_tui.widgets import PromptInput


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


async def _drain_workers(app: KimixTuiApp) -> None:
    for worker in list(app.workers):
        with suppress(WorkerCancelled, WorkerFailed):
            await worker.wait()


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
            await asyncio.Event().wait()

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
        prompt = chat.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()

        records = [(record.kind, record.text) for record in chat.transcript.records]
        assert session.prompts == ["hi"]
        assert ("user", "hi") in records
        assert ("assistant", "hello world") in records

    assert session.closed is True


@pytest.mark.asyncio
async def test_keyboard_submit_sends_multiline_prompt(tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="ok"), TurnEnd()])

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
        prompt = chat.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.press("h", "i", "ctrl+enter", "t", "h", "e", "r", "e", "enter")
        await app.workers.wait_for_complete()

        records = [(record.kind, record.text) for record in chat.transcript.records]
        assert session.prompts == ["hi\nthere"]
        assert ("user", "hi\nthere") in records
        assert prompt.text == ""

    assert session.closed is True


@pytest.mark.asyncio
async def test_chat_prompt_stays_within_screen(tmp_path: Path) -> None:
    session = FakeSession([TurnEnd()])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        prompt = chat.query_one("#prompt", PromptInput)
        assert prompt.region.x >= 0
        assert prompt.region.right <= chat.size.width
        prompt.focus()
        await pilot.press(*(["ctrl+enter"] * 5))
        await pilot.pause()
        assert prompt.region.x >= 0
        assert prompt.region.right <= chat.size.width


@pytest.mark.asyncio
async def test_chat_live_stream_keeps_timeline_and_appends_at_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimix_tui.screens.chat as chat_module

    async def _timeline(*_args, **_kwargs) -> Timeline:
        turns = [
            [HistoryBlock("user", f"q{index}"), HistoryBlock("assistant", f"a{index}")]
            for index in range(30)
        ]
        turns[0].insert(1, HistoryBlock("tool", "Read file\nPath: a.py"))
        return Timeline.from_turn_blocks(turns)

    monkeypatch.setattr(chat_module, "create_timeline", _timeline)

    session = FakeSession([TextPart(text="hel"), TextPart(text="lo"), TurnEnd()])

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
        texts = [record.text for record in chat.transcript.records]
        assert "q0" in texts
        assert "q29" in texts
        assert any(record.kind == "tool" and "Read file" in record.text for record in chat.transcript.records)
        chat.transcript.jump_to_turn(0)
        assert chat.transcript.pinned_to_latest is False

        prompt = chat.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.press("h", "i", "enter")
        await app.workers.wait_for_complete()

        texts = [record.text for record in chat.transcript.records]
        assert "q0" in texts
        assert "q29" in texts
        assert chat.transcript.records[-1].text == "hello"
        assert chat.transcript.pinned_to_latest is False
        assert chat.transcript.viewport_turn() == 0


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
        chat.query_one("#prompt", PromptInput).focus()
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()

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
        chat.query_one("#prompt", PromptInput).focus()
        await pilot.press("g", "o", "enter")
        await app.workers.wait_for_complete()

        session_line = str(chat.query_one("#status", Static).content)
        context = str(chat.query_one("#context", Static).content)
        assert "session fake-session" in session_line
        assert "context" not in session_line
        assert "context 2,000/20,000" in context
        assert "tokens in 950" in context
        assert "cache read 800" in context
        assert "out 75" in context
        assert "message" not in context


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
async def test_chat_chrome_keeps_history_and_dedupes_footer(tmp_path: Path) -> None:
    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)):
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        assert list(chat.query(Header)) == []
        toolbar = chat.query_one("#history-toolbar", Horizontal)
        assert toolbar.display is True
        assert str(chat.query_one("#history-info", Static).content).startswith("History ·")
        assert chat.query_one("#open-settings", Button).display is True
        assert chat.query_one("#leave-session", Button).display is True
        assert chat.query_one("#open-settings", Button).compact is True
        assert chat.query_one("#leave-session", Button).compact is True
        assert str(chat.query_one("#load-older", Button).label) == "←"
        footer_row = chat.query_one("#chat-footer", Horizontal)
        footer = footer_row.query_one(Footer)
        status = chat.query_one("#status", Static)
        context = chat.query_one("#context", Static)
        assert status.parent is chat.query_one("#chat-toolbar", Horizontal)
        assert context.parent is footer_row
        assert "connecting" in str(status.content) or "session" in str(
            status.content
        ).casefold()
        assert "context" in str(context.content) or "ready" in str(context.content).casefold()
        assert len(footer_row.query("#status")) == 0
        assert footer.show_command_palette is False
        shown = [binding.description for binding in ChatScreen.BINDINGS if binding.show]
        assert shown == ["Cancel"]
        palette_keys = [
            key
            for key in footer.query("*")
            if "-command-palette" in key.classes or key.__class__.__name__ == "FooterKey"
            and getattr(key, "action", "") == "app.command_palette"
        ]
        assert palette_keys == []
        footer_keys = [
            getattr(key, "description", "")
            for key in footer.query("*")
            if key.__class__.__name__ == "FooterKey"
        ]
        assert footer_keys == ["Cancel"]


@pytest.mark.asyncio
async def test_chat_history_toolbar_stays_visible_for_short_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimix_tui.screens.chat as chat_module

    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(2)

    monkeypatch.setattr(chat_module, "create_timeline", _timeline)

    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(100, 35)):
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        toolbar = chat.query_one("#history-toolbar", Horizontal)
        assert toolbar.display is True
        assert str(chat.query_one("#history-info", Static).content).startswith(
            "History · Turn 2 of 2"
        )
        assert chat.query_one("#load-older", Button).disabled is False
        assert chat.query_one("#history-turn", Input).disabled is False


@pytest.mark.asyncio
async def test_chat_loads_older_history_pages_without_losing_latest_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kimix_tui.screens.chat as chat_module

    async def _timeline(*_args, **_kwargs) -> Timeline:
        return _fake_timeline(130)

    monkeypatch.setattr(chat_module, "create_timeline", _timeline)

    session = FakeSession([])

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    async with app.run_test(size=(40, 35)) as pilot:
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        assert chat.has_class("-narrow")
        assert str(chat.query_one("#history-info", Static).content).startswith(
            "History · Turn 130 of 130"
        )
        assert all(
            button.region.right <= chat.size.width
            for button in chat.query("#history-actions Button")
        )
        turn_input = chat.query_one("#history-turn", Input)
        assert turn_input.region.right <= chat.size.width
        assert turn_input.placeholder == "Turn 1-130"
        texts = [record.text for record in chat.transcript.records]
        assert "q0" in texts
        assert "q129" in texts
        await pilot.click("#load-older")
        await app.workers.wait_for_complete()
        older_info = str(chat.query_one("#history-info", Static).content)
        assert older_info.startswith("History · Turn ")
        assert older_info.endswith(" of 130")
        assert chat.transcript.pinned_to_latest is False
        assert any(record.text == "q128" for record in chat.transcript.records)
        assert any(record.text == "q129" for record in chat.transcript.records)

        await pilot.press("ctrl+up")
        await app.workers.wait_for_complete()
        assert str(chat.query_one("#history-info", Static).content).endswith(" of 130")
        assert chat.transcript.pinned_to_latest is False

        await pilot.press("f3")
        assert turn_input.has_focus
        turn_input.value = "5"
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        texts = [record.text for record in chat.transcript.records]
        assert "q4" in texts
        assert "q129" not in texts
        jumped_info = str(chat.query_one("#history-info", Static).content)
        assert jumped_info.startswith("History · Turn ")
        assert jumped_info.endswith(" of 130")
        visible = "\n".join(
            chat.transcript.render_line(y).text for y in range(chat.transcript.size.height)
        )
        assert "q4" in visible

        await pilot.click("#load-newer")
        await app.workers.wait_for_complete()
        newer_info = str(chat.query_one("#history-info", Static).content)
        assert newer_info.startswith("History · Turn ")
        assert newer_info.endswith(" of 130")
        assert chat.transcript.pinned_to_latest is False

        await pilot.click("#jump-latest")
        await app.workers.wait_for_complete()
        assert str(chat.query_one("#history-info", Static).content).startswith(
            "History · Turn 130 of 130"
        )
        assert chat.transcript.pinned_to_latest is True
        texts = [record.text for record in chat.transcript.records]
        assert "q129" in texts
        assert "q0" not in texts


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
        chat.query_one("#prompt", PromptInput).focus()
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
async def test_approval_is_resolved_by_clicking_approve(tmp_path: Path) -> None:
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
        chat.query_one("#prompt", PromptInput).focus()

        await pilot.press("g", "o", "enter")
        await pilot.pause()
        approval_screen = app.screen
        assert isinstance(approval_screen, ApprovalScreen)
        await pilot.click("#approve")
        await app.workers.wait_for_complete()

        assert await approval.wait() == "approve"
        assert app.screen is chat


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
        chat.query_one("#prompt", PromptInput).focus()

        await pilot.press("g", "o", "enter")
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("escape")
        await app.workers.wait_for_complete()

        assert await approval.wait() == "reject"
        assert app.screen is chat
        assert chat.session is session
        assert session.closed is False


@pytest.mark.asyncio
async def test_cancelling_prompt_after_transcript_unmount_is_quiet(tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    failures: list[BaseException] = []

    async with app.run_test(size=(100, 35)) as pilot:
        app._handle_exception = failures.append  # type: ignore[method-assign]
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        prompt = chat.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.press("c", "o", "n", "t", "enter")
        await session.prompt_started.wait()
        await pilot.pause()
        await chat.query_one("#transcript").remove()
        chat.workers.cancel_group(chat, "prompt")
        await _drain_workers(app)
        await pilot.pause()

    assert failures == []
    assert session.prompts == ["cont"]
    assert chat.busy is False


@pytest.mark.asyncio
async def test_leave_during_running_prompt_returns_home_quietly(tmp_path: Path) -> None:
    session = FakeSession([TextPart(text="partial")], hang_prompt=True)

    async def factory(_options: SessionOptions) -> FakeSession:
        return session

    app = KimixTuiApp(
        SessionOptions(tmp_path, session_id="fake-session"),
        session_factory=factory,
        config_store=_config_store(tmp_path),
    )
    failures: list[BaseException] = []

    async with app.run_test(size=(100, 35)) as pilot:
        app._handle_exception = failures.append  # type: ignore[method-assign]
        await app.workers.wait_for_complete()
        chat = app.screen
        assert isinstance(chat, ChatScreen)
        chat.query_one("#prompt", PromptInput).focus()
        await pilot.press("c", "o", "n", "t", "enter")
        await session.prompt_started.wait()
        await pilot.pause()
        await pilot.press("escape")
        await _drain_workers(app)
        await pilot.pause()

        assert isinstance(app.screen, HomeScreen)
        assert chat.session is None
        assert chat.busy is False

    assert failures == []
    assert session.cancelled is True
    assert session.closed is True

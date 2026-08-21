"""Local memory-growth simulation for kimix-tui.

Simulates many real chat conversations end-to-end WITHOUT any network request:
it drives the PySide6 app (Home <-> Chat) with fake SDK sessions.  Every
conversation is opened (session factory), used (session.prompt -> render ->
Transcript), then left (Esc -> close_session), exactly like a user.

Reads process working-set size on Windows (workset) and also prints the capped
Transcript budget so we can see whether the design bounds resident memory.

Usage:
    uv run python scripts/simulate_many_chats_memory.py
    uv run python scripts/simulate_many_chats_memory.py --sessions 200 --turns 6
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
import tracemalloc
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from kimi_agent_sdk import TextPart, TurnBegin, TurnEnd
from PySide6.QtWidgets import QApplication

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions
from kimix_tui.llm_config import LLMConfigStore, inspect_llm_config
from kimix_tui.qt.chat_view import ChatView
from kimix_tui.qt.home_view import HomeView


def _rss_mib() -> float:
    """Return current working set (Windows) or RSS (POSIX) of this process."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_info = psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(get_process(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1024**2
        return 0.0

    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class FakeSession:
    """A session that streams a fixed, locally-generated conversation."""

    def __init__(self, session_id: str, turns: int, chars: int) -> None:
        self.id = session_id
        self.status = SimpleNamespace(
            context_tokens=100,
            max_context_tokens=1_000_000,
            context_usage=0.1,
        )
        self._turns = turns
        self._chars = chars
        self.closed = False
        self.cancelled = False
        self.prompts: list[str] = []
        self._line = "x" * chars

    async def prompt(
        self,
        user_input: str,
        *,
        merge_wire_messages: bool = False,
    ) -> AsyncIterator[object]:
        self.prompts.append(user_input)
        assert merge_wire_messages is False
        for _ in range(self._turns):
            yield TurnBegin(user_input=user_input)
            chunk = 64
            for start in range(0, self._chars, chunk):
                yield TextPart(text=self._line[start : start + chunk])
            yield TurnEnd()

    def cancel(self) -> None:
        self.cancelled = True

    async def clear(self, **custom_arguments: object) -> None:
        return None

    async def compact(self, *, custom_instruction: str = "") -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _make_config_store(tmp_path: Path) -> LLMConfigStore:
    config_file = tmp_path / "provider.json"
    config_file.write_text(
        json.dumps(
            {
                "model": "test-model",
                "max_context_size": 1_000_000,
                "url": "https://example.test/v1",
                "type": "openai_legacy",
                "api_key": "test-key",
            }
        ),
        encoding="utf-8",
    )
    store = LLMConfigStore(
        tmp_path / "kimix-tui.json",
        session_file_resolver=lambda _wd, sid: (tmp_path / "sessions" / sid / "kimix-tui.json"),
    )
    store.set_default(tmp_path, inspect_llm_config(config_file))
    return store


def _wait_idle(app: KimixTuiApp, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    qt_app = QApplication.instance()
    assert qt_app is not None
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if app.bridge.is_idle():
            return
        time.sleep(0.01)
    raise TimeoutError("bridge did not become idle")


def _sample(app: KimixTuiApp, label: str, i: int) -> None:
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    chat = app.screen
    records = 0
    chars = 0
    if isinstance(chat, ChatView):
        records = len(chat.transcript.records)
        chars = sum(len(record.text) for record in chat.transcript.records)
    print(
        f"{label} i={i} screen={type(app.screen).__name__} "
        f"records={records} chars={chars / 1024**2:.2f}MiB "
        f"heap={current / 1024**2:.1f}MiB rss={_rss_mib():.1f}MiB "
        f"heap_peak={peak / 1024**2:.1f}MiB"
    )


def main(args: argparse.Namespace) -> None:
    tracemalloc.start()
    tmp_path = Path(args.work_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_store = _make_config_store(tmp_path)
    sessions_created: list[FakeSession] = []

    def make_session(session_id: str) -> FakeSession:
        session = FakeSession(session_id, args.turns, args.chars)
        sessions_created.append(session)
        return session

    factory_calls: list[str] = []

    async def factory(options: SessionOptions) -> FakeSession:
        factory_calls.append(options.session_id or "?")
        return make_session(f"tui_{len(factory_calls)}")

    qt_app = QApplication.instance() or QApplication(sys.argv)
    app = KimixTuiApp(
        SessionOptions(tmp_path),
        session_factory=factory,
        config_store=config_store,
    )
    window = app.create_window()
    window.resize(1000, 700)
    window.show()
    _wait_idle(app)
    _sample(app, "start", 0)

    for i in range(1, args.sessions + 1):
        home = app.screen
        if not isinstance(home, HomeView):
            raise TypeError(f"unexpected screen: {type(app.screen).__name__}")
        home.request_new_session()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            qt_app.processEvents()
            if isinstance(app.screen, ChatView) and app.screen.prompt_enabled and app.bridge.is_idle():
                break
            time.sleep(0.01)
        chat = app.screen
        if not isinstance(chat, ChatView):
            raise TypeError(f"expected ChatView, got {type(app.screen).__name__}")
        prompt = chat.prompt
        prompt.setFocus()
        for _t in range(args.prompts):
            prompt.setPlainText("greeting")
            prompt.submitted.emit("greeting")
            _wait_idle(app)
        if i % args.sample_every == 0:
            _sample(app, "in_chat", i)
        app.leave_chat()
        _wait_idle(app)
        if not isinstance(app.screen, HomeView):
            raise TypeError(type(app.screen).__name__)
        if i % args.sample_every == 0:
            _sample(app, "after_leave", i)

    print("phase=one_long_conversation")
    home = app.screen
    assert isinstance(home, HomeView)
    home.request_new_session()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if isinstance(app.screen, ChatView) and app.screen.prompt_enabled and app.bridge.is_idle():
            break
        time.sleep(0.01)
    chat = app.screen
    assert isinstance(chat, ChatView)
    prompt = chat.prompt
    prompt.setFocus()
    for turn in range(args.long_turns):
        prompt.setPlainText("greeting")
        prompt.submitted.emit("greeting")
        _wait_idle(app)
        if (turn + 1) % args.sample_every == 0:
            _sample(app, "long", turn + 1)

    print(
        f"summary sessions_factory_called={len(factory_calls)} "
        f"created={len(sessions_created)} "
        f"closed={sum(1 for session in sessions_created if session.closed)} "
        f"rss_final={_rss_mib():.1f}MiB"
    )
    window.close()
    app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=60, help="conversations to open+leave")
    parser.add_argument("--prompts", type=int, default=3, help="messages per conversation")
    parser.add_argument("--turns", type=int, default=2, help="assistant replies per prompt")
    parser.add_argument("--chars", type=int, default=300, help="chars per assistant reply")
    parser.add_argument("--long-turns", type=int, default=4000, help="replies in the long convo")
    parser.add_argument("--long-chars", type=int, default=4000, help="chars per reply in long convo")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--work-dir", default=".kimix_cache/memory_sim")
    main(parser.parse_args())

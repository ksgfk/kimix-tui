"""Measure history page and transcript memory while revisiting a wire log."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import random
import tracemalloc
from pathlib import Path

from textual.app import App, ComposeResult

from kimix_tui.history import (
    MAX_HISTORY_BLOCKS,
    MAX_HISTORY_WINDOW_TURNS,
    SessionHistory,
    WireHistoryIndex,
    _scan_wire_history_index,
    load_wire_history_page,
)
from kimix_tui.widgets import Transcript


def rss_mib() -> float:
    """Return the current Windows working set, or zero on other platforms."""

    if not hasattr(ctypes, "WinDLL"):
        return 0.0
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
    if not get_info(get_process(), ctypes.byref(counters), counters.cb):
        return 0.0
    return counters.WorkingSetSize / 1024**2


class TranscriptApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")


async def load_page(index: WireHistoryIndex, start: int) -> SessionHistory:
    end = min(index.total_turns, start + MAX_HISTORY_WINDOW_TURNS)
    return await load_wire_history_page(
        index,
        end_turn=end,
        page_turns=end - start,
        max_blocks=MAX_HISTORY_BLOCKS,
    )


def print_stats(prefix: str, step: int, heap_peak: int) -> None:
    current, peak = tracemalloc.get_traced_memory()
    print(
        f"{prefix} step={step} heap={current / 1024**2:.1f}MiB "
        f"rss={rss_mib():.1f}MiB peak={max(peak, heap_peak) / 1024**2:.1f}MiB"
    )


async def main(path: Path, jumps: int, interval: int, paint: bool) -> None:
    randomizer = random.Random(246822)
    tracemalloc.start()
    index = await asyncio.to_thread(_scan_wire_history_index, path)
    print(f"index turns={index.total_turns} offsets={len(index.turn_offsets)}")

    print("phase=loader")
    before_loader = tracemalloc.take_snapshot()
    last_page_chars = 0
    for step in range(jumps):
        start = randomizer.randrange(max(1, index.total_turns - MAX_HISTORY_WINDOW_TURNS))
        page = await load_page(index, start)
        last_page_chars = sum(len(block.text) for block in page.blocks)
        del page
        if step % interval == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            print(
                f"loader step={step} page_chars={last_page_chars / 1024**2:.1f}MiB "
                f"heap={current / 1024**2:.1f}MiB rss={rss_mib():.1f}MiB "
                f"peak={peak / 1024**2:.1f}MiB"
            )

    await asyncio.sleep(0)
    await asyncio.sleep(0)
    gc.collect()
    after_loader = tracemalloc.take_snapshot()
    print("loader_top_allocations")
    for stat in after_loader.compare_to(before_loader, "lineno")[:8]:
        print(stat)

    print("phase=transcript")
    before_transcript = tracemalloc.take_snapshot()
    app = TranscriptApp()
    async with app.run_test(size=(100, 35)) as pilot:
        transcript = app.query_one(Transcript)
        await transcript.append_block("system", "Session: diagnostic")
        transcript.mark_history_window(1, 1)
        for step in range(jumps):
            start = randomizer.randrange(max(1, index.total_turns - MAX_HISTORY_WINDOW_TURNS))
            page = await load_page(index, start)
            await transcript.replace_history_blocks(
                [(block.kind, block.text) for block in page.blocks]
            )
            transcript.jump_to_history_start()
            transcript.scroll_end(animate=False, immediate=True, force=True)
            if paint:
                for row in range(transcript.size.height):
                    transcript.render_line(row)
            del page
            await pilot.pause()
            if step % interval == 0:
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                strip_count = sum(
                    len(strips) for strips in transcript._strip_cache.values()
                )
                record_chars = sum(len(record.text) for record in transcript.records)
                print(
                    f"transcript step={step} records={len(transcript.records)} "
                    f"strips={strip_count} chars={record_chars / 1024**2:.1f}MiB "
                    f"heap={current / 1024**2:.1f}MiB rss={rss_mib():.1f}MiB "
                    f"peak={peak / 1024**2:.1f}MiB"
                )

        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        after_transcript = tracemalloc.take_snapshot()
        print("transcript_top_allocations")
        for stat in after_transcript.compare_to(before_transcript, "lineno")[:12]:
            print(stat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--jumps", type=int, default=50)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--skip-paint", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.path, args.jumps, max(1, args.interval), not args.skip_paint))

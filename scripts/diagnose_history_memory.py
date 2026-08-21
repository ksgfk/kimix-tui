"""Measure timeline hydrate/unload and transcript memory while revisiting a wire log."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import os
import random
import sys
import tracemalloc
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from kimix_tui.history import Timeline, _scan_wire_history_index
from kimix_tui.qt.transcript import Transcript


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


async def seek_timeline(timeline: Timeline, turn: int) -> list[tuple[str, str, int]]:
    await timeline.ensure_turn(turn)
    return timeline.display_items()


async def main(path: Path, jumps: int, interval: int, paint: bool) -> None:
    qt_app = QApplication.instance() or QApplication(sys.argv)
    randomizer = random.Random(246822)
    tracemalloc.start()
    index = await asyncio.to_thread(_scan_wire_history_index, path)
    timeline = Timeline(index=index)
    await timeline.open()
    print(
        f"index turns={index.total_turns} offsets={len(index.turn_offsets)} "
        f"hydrated={timeline.hydrated_chars()}"
    )

    print("phase=loader")
    before_loader = tracemalloc.take_snapshot()
    last_chars = 0
    for step in range(jumps):
        turn = randomizer.randrange(max(1, index.total_turns))
        items = await seek_timeline(timeline, turn)
        last_chars = timeline.hydrated_chars()
        del items
        if step % interval == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            print(
                f"loader step={step} turn={turn} hydrated={last_chars / 1024**2:.1f}MiB "
                f"materialized={timeline.materialized_turn_count} "
                f"heap={current / 1024**2:.1f}MiB rss={rss_mib():.1f}MiB "
                f"peak={peak / 1024**2:.1f}MiB"
            )

    await asyncio.sleep(0)
    gc.collect()
    after_loader = tracemalloc.take_snapshot()
    print("loader_top_allocations")
    for stat in after_loader.compare_to(before_loader, "lineno")[:8]:
        print(stat)

    print("phase=transcript")
    before_transcript = tracemalloc.take_snapshot()
    transcript = Transcript()
    transcript.resize(800, 560)
    transcript.show()
    qt_app.processEvents()
    transcript.append_block("system", "Session: diagnostic")
    transcript.mark_history_window(1, 1)
    for step in range(jumps):
        turn = randomizer.randrange(max(1, index.total_turns))
        items = await seek_timeline(timeline, turn)
        transcript.replace_history_blocks(
            [(kind, text, item_turn) for kind, text, item_turn in items]
        )
        transcript.jump_to_turn(turn)
        if paint:
            transcript.visible_text()
        del items
        qt_app.processEvents()
        if step % interval == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            record_chars = sum(len(record.text) for record in transcript.records)
            print(
                f"transcript step={step} records={len(transcript.records)} "
                f"cache={len(transcript._strip_cache)} chars={record_chars / 1024**2:.1f}MiB "
                f"heap={current / 1024**2:.1f}MiB rss={rss_mib():.1f}MiB "
                f"peak={peak / 1024**2:.1f}MiB"
            )

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

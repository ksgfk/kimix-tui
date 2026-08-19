"""Measure bounded transcript memory and visible-row rendering latency.

Examples:
    uv run python scripts/benchmark_transcript.py --gigabytes 1
    uv run python scripts/benchmark_transcript.py --gigabytes 2 --max-chars 33554432
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import sys
import time
import tracemalloc

from textual.app import App

from kimix_tui.widgets import MAX_TRANSCRIPT_CHARS, Transcript


def _peak_rss_mib() -> float:
    """Return peak resident memory using only the standard library."""

    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        if get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / 1024**2
        return 0.0

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 if sys.platform != "darwin" else 1024**2)


class _BenchmarkApp(App[None]):
    def compose(self):
        yield Transcript(id="transcript", max_chars=self.max_chars)

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self.max_chars = max_chars


async def _run(args: argparse.Namespace) -> None:
    app = _BenchmarkApp(args.max_chars)
    tracemalloc.start()
    async with app.run_test(size=(args.width, args.height)) as pilot:
        transcript = app.query_one(Transcript)
        total_records = int(args.gigabytes * (1024**3) / args.record_bytes)
        started = time.perf_counter()
        for start in range(0, total_records, args.chunk):
            stop = min(total_records, start + args.chunk)
            await transcript.append_blocks(
                [
                    ("assistant", f"{index:08d} " + ("x" * (args.record_bytes - 9)))
                    for index in range(start, stop)
                ]
            )
        await pilot.pause()
        append_seconds = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        print(
            f"logical={total_records * args.record_bytes / 1024**3:.2f} GiB "
            f"records={len(transcript.records)} omitted={transcript.omitted_records} "
            f"append={append_seconds:.2f}s current={current / 1024**2:.1f} MiB "
            f"heap_peak={peak / 1024**2:.1f} MiB rss_peak={_peak_rss_mib():.1f} MiB"
        )

        samples: list[float] = []
        for position in (
            0,
            transcript.max_scroll_y // 4,
            transcript.max_scroll_y // 2,
            max(0, transcript.max_scroll_y - args.height),
            transcript.max_scroll_y,
        ):
            transcript.scroll_to(y=position, animate=False, immediate=True, force=True)
            await pilot.pause()
            started = time.perf_counter()
            for row in range(transcript.size.height):
                transcript.render_line(row)
            samples.append((time.perf_counter() - started) * 1000)
        _current, peak = tracemalloc.get_traced_memory()
        print(
            "scroll_render_ms="
            + ",".join(f"{sample:.2f}" for sample in samples)
            + f" cache={len(transcript._strip_cache)} heap_peak={peak / 1024**2:.1f} MiB "
            + f"rss_peak={_peak_rss_mib():.1f} MiB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gigabytes", type=float, default=1.0)
    parser.add_argument("--record-bytes", type=int, default=4_000)
    parser.add_argument("--chunk", type=int, default=1_000)
    parser.add_argument("--max-chars", type=int, default=MAX_TRANSCRIPT_CHARS)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=35)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()

"""Command-line entry point for ``kimix-tui``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kimix_tui.app import KimixTuiApp
from kimix_tui.backend import SessionOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard-first PySide6 client for Kimix")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Project working directory (default: current directory)",
    )
    parser.add_argument(
        "--session",
        help="Resume this session id, or create it when it does not exist",
    )
    parser.add_argument("--config", type=Path, help="Kimix provider JSON configuration file")
    parser.add_argument("--model", help="SDK model name")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Automatically approve SDK permission requests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = SessionOptions(
        work_dir=args.work_dir,
        session_id=args.session,
        config_file=args.config,
        model=args.model,
        thinking=args.thinking,
        yolo=args.yolo,
    )
    KimixTuiApp(options).run()


if __name__ == "__main__":
    main()

"""Progress and pytest subprocess helpers for long benchmark runs."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable


try:
    _PROGRESS_STREAM = open("/dev/tty", "w", buffering=1)
except OSError:
    _PROGRESS_STREAM = sys.stderr


def progress(message: str) -> None:
    """Write progress immediately, even when pytest captures stdout."""
    print(message, file=_PROGRESS_STREAM, flush=True)


def run_with_progress(label: str, runner, mode: str, interval: int = 30):
    """Run a blocking callable and emit heartbeat progress while it runs."""
    done = threading.Event()
    start = time.monotonic()
    progress(f"[{mode}] START {label}")

    def heartbeat():
        while not done.wait(interval):
            elapsed = time.monotonic() - start
            progress(f"[{mode}] STILL RUNNING {label} ({elapsed:.0f}s)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return runner()
    finally:
        done.set()
        elapsed = time.monotonic() - start
        progress(f"[{mode}] DONE {label} ({elapsed:.1f}s)")


def pytest_extra_args(argv: Iterable[str]) -> list[str]:
    """Forward pytest args, but leave benchmark JSON ownership to the launcher."""
    args = list(argv)
    extra: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--benchmark-json":
            skip_next = True
            continue
        if arg.startswith("--benchmark-json="):
            continue
        extra.append(arg)

    if "-s" not in extra and "--capture=no" not in extra and not any(
        arg.startswith("--capture=") for arg in extra
    ):
        extra.append("--capture=tee-sys")

    return extra


def pytest_command(test_file: str, benchmark_json: Path | str, extra_args: Iterable[str]) -> list[str]:
    """Build an unbuffered pytest command with a controlled benchmark JSON file."""
    return [
        sys.executable,
        "-u",
        "-m",
        "pytest",
        test_file,
        f"--benchmark-json={benchmark_json}",
        *extra_args,
    ]


def benchmark_out_dir(default: str) -> Path:
    """Resolve and create the benchmark output directory."""
    out_dir = Path(os.getenv("BENCHMARK_OUT_DIR", default))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

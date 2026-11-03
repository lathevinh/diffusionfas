"""Bounded subprocesses owned by a probe, never other users' processes."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def process_tree_rss(pid: int) -> int:
    pending, visited, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
        pending.extend(map(int, children))
    return total


def tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def signal_owned_group(process: subprocess.Popen, value: int) -> None:
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        pass


def run_bounded(
    command: list[str], *, cwd: Path, owned_root: Path, log: Path,
    max_seconds: float, max_rss_bytes: int, max_disk_bytes: int,
    additional_output_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if max_seconds <= 0 or max_rss_bytes <= 0 or max_disk_bytes <= 0:
        raise ValueError("positive probe process caps required")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise ValueError("new process log required")
    output_roots = (owned_root, *additional_output_roots)
    started, peak_rss, peak_disk = time.monotonic(), 0, sum(tree_bytes(p) for p in output_roots)
    failure = None
    with log.open("xb") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, process_tree_rss(process.pid))
                peak_disk = max(peak_disk, sum(tree_bytes(p) for p in output_roots))
                if time.monotonic() - started > max_seconds:
                    failure = "wall_clock_cap"
                elif peak_rss > max_rss_bytes:
                    failure = "process_tree_rss_cap"
                elif peak_disk > max_disk_bytes:
                    failure = "owned_disk_cap"
                if failure:
                    break
                time.sleep(0.2)
        finally:
            signal_owned_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_owned_group(process, signal.SIGKILL)
                process.wait(timeout=5)
            signal_owned_group(process, signal.SIGKILL)
    peak_disk = max(peak_disk, sum(tree_bytes(p) for p in output_roots))
    return {
        "returncode": process.returncode, "guard_failure": failure,
        "wall_seconds": time.monotonic() - started, "sampled_tree_peak_rss_bytes": peak_rss,
        "sampled_owned_peak_disk_bytes": peak_disk,
        "completed": process.returncode == 0 and failure is None and peak_disk <= max_disk_bytes,
        "sampling_interval_seconds": 0.2,
    }

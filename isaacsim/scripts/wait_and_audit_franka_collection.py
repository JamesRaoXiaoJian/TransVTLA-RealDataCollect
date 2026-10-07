#!/usr/bin/env python3
"""Wait for a Franka collection screen run to finish, then audit it."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ERROR_RE = re.compile(r"Traceback|Exception|Fatal|(^|[^A-Za-z])ERROR([^A-Za-z]|$)|\[Error\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/media/data"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-total-env-sessions", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--settle-seconds", type=int, default=60)
    parser.add_argument("--quick-images", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def screen_sessions(session_prefix: str) -> list[str]:
    proc = subprocess.run(["screen", "-ls"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    sessions: list[str] = []
    pattern = re.compile(rf"\b{re.escape(session_prefix)}_shard\d+_gpu\d+\b")
    for line in proc.stdout.splitlines():
        match = pattern.search(line)
        if match:
            sessions.append(match.group(0))
    return sessions


def log_paths(output_dir: Path, run_prefix: str) -> list[Path]:
    return sorted(output_dir.glob(f"{run_prefix}_shard*_gpu*.log"))


def run_dirs(output_dir: Path, run_prefix: str) -> list[Path]:
    return sorted(p for p in output_dir.glob(f"{run_prefix}_shard*_gpu*") if p.is_dir())


def completed_log_count(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "Collection completed" in text:
            count += 1
    return count


def has_errors(paths: list[Path]) -> bool:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if ERROR_RE.search(text):
            return True
    return False


def count_sessions(output_dir: Path, run_prefix: str) -> int:
    total = 0
    for run_dir in run_dirs(output_dir, run_prefix):
        total += sum(1 for p in run_dir.glob("session_ep*_env*") if (p / "frames.csv").exists())
    return total


def count_config_progress(output_dir: Path, run_prefix: str) -> dict[str, int]:
    progress: dict[str, int] = {}
    for run_dir in run_dirs(output_dir, run_prefix):
        progress[run_dir.name] = sum(1 for p in run_dir.glob("session_ep*_env*") if (p / "frames.csv").exists())
    return progress


def run_audit(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "audit_franka_collection.py"),
        "--output-dir",
        str(args.output_dir),
        "--run-prefix",
        args.run_prefix,
        "--expected-shards",
        str(args.expected_shards),
        "--expected-total-env-sessions",
        str(args.expected_total_env_sessions),
    ]
    if args.quick_images:
        cmd.append("--quick-images")
    print(f"[{timestamp()}] running audit: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True)
    print(f"[{timestamp()}] audit exit code: {proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            {
                "event": "watch_started",
                "time": timestamp(),
                "output_dir": str(args.output_dir),
                "run_prefix": args.run_prefix,
                "session_prefix": args.session_prefix,
                "expected_total_env_sessions": args.expected_total_env_sessions,
                "poll_seconds": args.poll_seconds,
                "settle_seconds": args.settle_seconds,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    while True:
        sessions = screen_sessions(args.session_prefix)
        logs = log_paths(args.output_dir, args.run_prefix)
        completed = completed_log_count(logs)
        session_count = count_sessions(args.output_dir, args.run_prefix)
        errors = has_errors(logs)
        print(
            json.dumps(
                {
                    "event": "poll",
                    "time": timestamp(),
                    "screen_sessions": sessions,
                    "logs": len(logs),
                    "collection_completed_logs": completed,
                    "session_dirs_with_frames": session_count,
                    "progress_by_shard": count_config_progress(args.output_dir, args.run_prefix),
                    "errors_seen": errors,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        all_logs_completed = completed >= args.expected_shards
        no_running_collection = not sessions and len(logs) >= args.expected_shards
        enough_sessions = session_count >= args.expected_total_env_sessions

        if all_logs_completed or (no_running_collection and enough_sessions) or (no_running_collection and errors):
            print(f"[{timestamp()}] collection appears finished; settling for {args.settle_seconds}s", flush=True)
            time.sleep(args.settle_seconds)
            return run_audit(args)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

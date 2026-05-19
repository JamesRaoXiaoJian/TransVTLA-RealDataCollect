#!/usr/bin/env python3
"""Collect gripper state through RM Plus and save CSV.

It reads the end-tool real-time information with rm_get_rm_plus_state_info();
pos[0] is recorded as the gripper opening value.
"""

from __future__ import annotations

import argparse
import csv
import signal
import time
from pathlib import Path
from typing import Optional

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


DEFAULT_HOST = "172.25.5.243"
DEFAULT_PORT = 8080
STOP = False


def handle_signal(signum: int, frame: object) -> None:
    del signum, frame
    global STOP
    STOP = True


def first(values: object) -> Optional[int]:
    if isinstance(values, list) and values:
        return values[0]
    return None


def connect(host: str, port: int) -> RoboticArm:
    print(f"connect robot: {host}:{port}")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(host, port)
    print(f"handle_id: {None if handle is None else handle.id}")
    if handle is None or getattr(handle, "id", -1) < 0:
        arm.rm_delete_robot_arm()
        raise RuntimeError(f"failed to connect robot at {host}:{port}")
    return arm


def cleanup(arm: RoboticArm, args: argparse.Namespace) -> None:
    print("cleanup")
    if not args.skip_delete:
        print(f"delete_robot_arm: {arm.rm_delete_robot_arm()}")


def collect(arm: RoboticArm, args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    period = 1.0 / args.hz
    duration_ns = None if args.duration <= 0 else int(args.duration * 1_000_000_000)
    start_ns = time.perf_counter_ns()
    next_tick = time.perf_counter()
    last_report = time.perf_counter()
    samples = 0
    misses = 0

    fieldnames = [
        "sample_index",
        "timestamp_ns",
        "elapsed_s",
        "target_hz",
        "rm_plus_read_code",
        "rm_plus_read_latency_ms",
        "sys_state",
        "gripper_pos",
        "gripper_speed",
        "gripper_current",
        "gripper_force",
        "gripper_dof_state",
        "gripper_dof_err",
        "deadline_late_ms",
    ]

    print(f"write csv: {output}")
    print(f"target_hz: {args.hz}")
    print("press Ctrl+C to stop")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not STOP:
            now_ns = time.perf_counter_ns()
            if duration_ns is not None and now_ns - start_ns >= duration_ns:
                break

            late_ms = max(0.0, (time.perf_counter() - next_tick) * 1000.0)
            if late_ms > 0.5:
                misses += 1

            read_start = time.perf_counter()
            try:
                code, data = arm.rm_get_rm_plus_state_info()
            except Exception:
                code, data = -1, {}
            read_latency_ms = (time.perf_counter() - read_start) * 1000.0

            payload = data if code == 0 and isinstance(data, dict) else {}
            gripper_pos = first(payload.get("pos"))
            gripper_speed = first(payload.get("speed"))
            gripper_current = first(payload.get("current"))
            gripper_force = first(payload.get("force"))
            gripper_dof_state = first(payload.get("dof_state"))
            gripper_dof_err = first(payload.get("dof_err"))
            sys_state = payload.get("sys_state", "")

            timestamp_ns = time.time_ns()
            elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000
            writer.writerow(
                {
                    "sample_index": samples,
                    "timestamp_ns": timestamp_ns,
                    "elapsed_s": f"{elapsed_s:.9f}",
                    "target_hz": args.hz,
                    "rm_plus_read_code": code,
                    "rm_plus_read_latency_ms": f"{read_latency_ms:.3f}",
                    "sys_state": sys_state,
                    "gripper_pos": "" if gripper_pos is None else gripper_pos,
                    "gripper_speed": "" if gripper_speed is None else gripper_speed,
                    "gripper_current": "" if gripper_current is None else gripper_current,
                    "gripper_force": "" if gripper_force is None else gripper_force,
                    "gripper_dof_state": "" if gripper_dof_state is None else gripper_dof_state,
                    "gripper_dof_err": "" if gripper_dof_err is None else gripper_dof_err,
                    "deadline_late_ms": f"{late_ms:.3f}",
                }
            )
            samples += 1

            if args.flush_every > 0 and samples % args.flush_every == 0:
                f.flush()

            now = time.perf_counter()
            if now - last_report >= args.report_interval:
                elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
                actual_hz = samples / elapsed if elapsed > 0 else 0.0
                print(
                    f"samples={samples} actual_hz={actual_hz:.1f} "
                    f"last_read_ms={read_latency_ms:.1f} misses={misses} "
                    f"last_code={code} last_pos={gripper_pos}"
                )
                last_report = now

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    elapsed = (time.perf_counter_ns() - start_ns) / 1_000_000_000
    actual_hz = samples / elapsed if elapsed > 0 else 0.0
    print(f"done: samples={samples}, elapsed_s={elapsed:.3f}, actual_hz={actual_hz:.2f}, misses={misses}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RM Plus gripper state to CSV.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Robot controller IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Robot API port.")
    parser.add_argument("--output", default="gripper_state.csv", help="Output CSV path.")
    parser.add_argument("--hz", type=float, default=200.0, help="Target polling rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration seconds. 0 means until Ctrl+C.")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Console report interval in seconds.")
    parser.add_argument("--flush-every", type=int, default=50, help="Flush CSV every N samples. 0 disables.")
    parser.add_argument("--skip-delete", action="store_true", help="Do not delete robot arm handle on exit.")
    return parser


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    args = build_parser().parse_args()
    arm = connect(args.host, args.port)
    try:
        collect(arm, args)
        return 0
    finally:
        cleanup(arm, args)


if __name__ == "__main__":
    raise SystemExit(main())

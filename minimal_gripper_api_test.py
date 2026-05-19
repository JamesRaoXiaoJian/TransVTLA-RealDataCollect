#!/usr/bin/env python3
"""Minimal RealMan gripper API test with detailed logging.

Uses official gripper control APIs and RM Plus state readback:
- rm_get_rm_plus_state_info
- rm_set_gripper_position
- rm_set_gripper_release
- rm_set_gripper_pick
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


DEFAULT_HOST = "172.25.5.243"
DEFAULT_PORT = 8080

ERROR_MESSAGES = {
    0: "success",
    1: "controller returned false: parameter error or robot state error",
    -1: "send failed: communication error",
    -2: "receive failed: communication error or controller timeout",
    -3: "response parse failed",
    -4: "timeout",
    -5: "arrival-device check failed: current arrival device is not gripper",
}


def code_text(code: int) -> str:
    return ERROR_MESSAGES.get(code, "unknown error")


def print_json(title: str, data: dict[str, Any]) -> None:
    print(f"{title}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def call_step(name: str, func: Callable[[], Any]) -> Any:
    print(f"\n===== {name} =====")
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    if isinstance(result, tuple) and result and isinstance(result[0], int):
        code = result[0]
        print(f"return_code: {code} ({code_text(code)})")
        if len(result) > 1 and isinstance(result[1], dict):
            print_json("data", result[1])
    elif isinstance(result, int):
        print(f"return_code: {result} ({code_text(result)})")
    else:
        print(f"result: {result!r}")

    print(f"elapsed_sec: {elapsed:.3f}")
    return result


def connect(host: str, port: int) -> RoboticArm:
    print("===== connect =====")
    print(f"target: {host}:{port}")
    start = time.perf_counter()

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(host, port)
    elapsed = time.perf_counter() - start

    print(f"handle: {handle!r}")
    print(f"handle_id: {None if handle is None else handle.id}")
    print(f"elapsed_sec: {elapsed:.3f}")

    if handle is None or getattr(handle, "id", -1) < 0:
        arm.rm_delete_robot_arm()
        raise RuntimeError(f"failed to connect robot at {host}:{port}")

    return arm


def run_test(args: argparse.Namespace) -> None:
    arm = connect(args.host, args.port)
    try:
        call_step("read initial RM Plus gripper state", arm.rm_get_rm_plus_state_info)

        call_step(
            f"set gripper position {args.position}",
            lambda: arm.rm_set_gripper_position(args.position, args.block, args.timeout),
        )
        time.sleep(args.pause)
        call_step("read RM Plus state after position", arm.rm_get_rm_plus_state_info)

        call_step(
            f"release gripper speed={args.speed}",
            lambda: arm.rm_set_gripper_release(args.speed, args.block, args.timeout),
        )
        time.sleep(args.pause)
        call_step("read RM Plus state after release", arm.rm_get_rm_plus_state_info)

        call_step(
            f"pick gripper speed={args.speed}, force={args.force}",
            lambda: arm.rm_set_gripper_pick(args.speed, args.force, args.block, args.timeout),
        )
        time.sleep(args.pause)
        call_step("read RM Plus state after pick", arm.rm_get_rm_plus_state_info)
    finally:
        print("\n===== disconnect =====")
        start = time.perf_counter()
        result = arm.rm_delete_robot_arm()
        elapsed = time.perf_counter() - start
        print(f"result: {result!r}")
        print(f"elapsed_sec: {elapsed:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal RealMan gripper API test with detailed print output."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Robot controller IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Robot API port.")
    parser.add_argument("--position", type=int, default=500, help="Gripper position, 1..1000.")
    parser.add_argument("--speed", type=int, default=300, help="Gripper speed, 1..1000.")
    parser.add_argument("--force", type=int, default=200, help="Gripper force, 50..1000.")
    parser.add_argument("--timeout", type=int, default=10, help="Blocking timeout in seconds.")
    parser.add_argument("--pause", type=float, default=0.5, help="Pause after each write.")
    parser.add_argument(
        "--no-block",
        action="store_false",
        dest="block",
        help="Use non-blocking gripper commands.",
    )
    parser.set_defaults(block=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

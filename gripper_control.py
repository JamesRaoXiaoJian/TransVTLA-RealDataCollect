#!/usr/bin/env python3
"""Official gripper control plus RM Plus state readback.

Control uses RealMan's built-in gripper APIs such as rm_set_gripper_position().
State readback uses rm_get_rm_plus_state_info(); dist["pos"][0] is treated as
the gripper opening value.
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
    1: "controller returned false / parameter or robot state error",
    -1: "send failed / communication error",
    -2: "receive failed / communication timeout",
    -3: "response parse failed",
    -4: "unsupported by this controller",
}


def text(code: int) -> str:
    return ERROR_MESSAGES.get(code, "unknown error")


def call(name: str, func: Callable[[], Any]) -> Any:
    print(f"\n===== {name} =====")
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    if isinstance(result, tuple) and result and isinstance(result[0], int):
        code = result[0]
        print(f"return_code: {code} ({text(code)})")
        if len(result) > 1:
            print(json.dumps(result[1], ensure_ascii=False, indent=2))
    elif isinstance(result, int):
        print(f"return_code: {result} ({text(result)})")
    else:
        print(f"result: {result!r}")

    print(f"elapsed_sec: {elapsed:.3f}")
    return result


def connect(host: str, port: int) -> RoboticArm:
    print("===== connect =====")
    print(f"target: {host}:{port}")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(host, port)
    print(f"handle_id: {None if handle is None else handle.id}")
    if handle is None or getattr(handle, "id", -1) < 0:
        arm.rm_delete_robot_arm()
        raise RuntimeError(f"failed to connect robot at {host}:{port}")
    return arm


def print_rm_plus_state(arm: RoboticArm) -> None:
    result = call("read RM Plus state info", arm.rm_get_rm_plus_state_info)
    if isinstance(result, tuple) and result[0] == 0 and isinstance(result[1], dict):
        pos = result[1].get("pos")
        if isinstance(pos, list) and pos:
            print(f"gripper_pos = dist['pos'][0] = {pos[0]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control gripper and read RM Plus state.")
    parser.add_argument("command", choices=("state", "position", "release", "pick"), help="Command to run.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Robot controller IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Robot API port.")
    parser.add_argument("--position", type=int, default=500, help="Target gripper position, 1..1000.")
    parser.add_argument("--speed", type=int, default=300, help="Gripper speed, 1..1000.")
    parser.add_argument("--force", type=int, default=200, help="Gripper force threshold, 50..1000.")
    parser.add_argument("--timeout", type=int, default=10, help="Blocking timeout in seconds.")
    parser.add_argument("--no-block", action="store_true", help="Use non-blocking gripper commands.")
    parser.add_argument("--pause", type=float, default=1.0, help="Pause after gripper actions before reading state.")
    parser.add_argument("--skip-delete", action="store_true", help="Do not delete robot arm handle on exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    arm = connect(args.host, args.port)
    try:
        block = not args.no_block

        if args.command == "state":
            print_rm_plus_state(arm)
        elif args.command == "position":
            call(
                f"set gripper position {args.position}",
                lambda: arm.rm_set_gripper_position(args.position, block, args.timeout),
            )
            time.sleep(args.pause)
            print_rm_plus_state(arm)
        elif args.command == "release":
            call(
                f"release gripper speed={args.speed}",
                lambda: arm.rm_set_gripper_release(args.speed, block, args.timeout),
            )
            time.sleep(args.pause)
            print_rm_plus_state(arm)
        elif args.command == "pick":
            call(
                f"pick gripper speed={args.speed}, force={args.force}",
                lambda: arm.rm_set_gripper_pick(args.speed, args.force, block, args.timeout),
            )
            time.sleep(args.pause)
            print_rm_plus_state(arm)
        return 0
    finally:
        if args.skip_delete:
            print("\nSkip delete robot arm handle by --skip-delete.")
        else:
            call("delete robot arm handle", arm.rm_delete_robot_arm)


if __name__ == "__main__":
    raise SystemExit(main())

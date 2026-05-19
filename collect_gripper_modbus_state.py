#!/usr/bin/env python3
"""Poll Zhixing CTAG2F120 gripper Modbus registers and save CSV.

The target rate defaults to 200 Hz for dataset collection. The RealMan API
Modbus read path may be slower than this on some controllers, so the script
records per-sample read latency and deadline misses instead of hiding them.

Known registers from the local operation manual:
- 258: target position, 2 registers / 4 bytes, 0..1000
- 264: run trigger register
- 284: moment register

The position register is a target/command register. It is useful for logging
the commanded gripper state, but it is not guaranteed to be real physical
position feedback unless the gripper firmware maps it that way.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_peripheral_read_write_params_t,
    rm_thread_mode_e,
)


DEFAULT_HOST = "172.25.5.243"
DEFAULT_PORT = 8080

MODBUS_PORT_TOOL = 1
BAUDRATE = 115200
MODBUS_TIMEOUT_100MS = 2
DEVICE_ADDR = 1
TOOL_VOLTAGE_24V = 3

REG_POSITION = 258
REG_RUN = 264
REG_MOMENT = 284

STOP = False


def handle_signal(signum: int, frame: object) -> None:
    del signum, frame
    global STOP
    STOP = True


def params(port: int, address: int, device: int, num: int) -> rm_peripheral_read_write_params_t:
    return rm_peripheral_read_write_params_t(port=port, address=address, device=device, num=num)


def decode_u32_be(bytes4: list[int]) -> Optional[int]:
    if len(bytes4) != 4:
        return None
    return (bytes4[0] << 24) | (bytes4[1] << 16) | (bytes4[2] << 8) | bytes4[3]


def connect(host: str, port: int) -> RoboticArm:
    print(f"connect robot: {host}:{port}")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(host, port)
    print(f"handle_id: {None if handle is None else handle.id}")
    if handle is None or getattr(handle, "id", -1) < 0:
        arm.rm_delete_robot_arm()
        raise RuntimeError(f"failed to connect robot at {host}:{port}")
    return arm


def setup_gripper_bus(arm: RoboticArm, args: argparse.Namespace) -> None:
    if args.no_setup:
        print("skip setup: --no-setup")
        return

    code = arm.rm_set_tool_voltage(TOOL_VOLTAGE_24V)
    print(f"set_tool_voltage(24V): {code}")
    if code != 0:
        raise RuntimeError(f"failed to set tool voltage, code={code}")
    time.sleep(args.setup_pause)

    code = arm.rm_set_modbus_mode(args.modbus_port, args.baudrate, args.modbus_timeout)
    print(
        "set_modbus_mode("
        f"port={args.modbus_port}, baudrate={args.baudrate}, timeout={args.modbus_timeout}"
        f"): {code}"
    )
    if code != 0:
        raise RuntimeError(f"failed to set modbus mode, code={code}")
    time.sleep(args.setup_pause)


def cleanup(arm: RoboticArm, args: argparse.Namespace) -> None:
    print("cleanup")
    if not args.no_setup and not args.keep_modbus:
        print(f"close_modbus_mode: {arm.rm_close_modbus_mode(args.modbus_port)}")
    if not args.no_setup and not args.keep_power:
        print(f"set_tool_voltage(0): {arm.rm_set_tool_voltage(0)}")
    print(f"delete_robot_arm: {arm.rm_delete_robot_arm()}")
    print(f"destroy: {arm.rm_destroy()}")


def read_position_register(arm: RoboticArm, args: argparse.Namespace) -> tuple[int, list[int]]:
    p = params(args.modbus_port, REG_POSITION, args.device, 2)
    return arm.rm_read_multiple_holding_registers(p)


def read_single_register(
    arm: RoboticArm,
    args: argparse.Namespace,
    address: int,
) -> tuple[int, Optional[int]]:
    p = params(args.modbus_port, address, args.device, 1)
    return arm.rm_read_holding_registers(p)


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
        "position_read_code",
        "position_read_latency_ms",
        "position_b0",
        "position_b1",
        "position_b2",
        "position_b3",
        "position_value",
        "run_read_code",
        "run_value",
        "moment_read_code",
        "moment_value",
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
            pos_code, pos_data = read_position_register(arm, args)
            read_latency_ms = (time.perf_counter() - read_start) * 1000.0

            pos_bytes = pos_data if pos_code == 0 else []
            pos_value = decode_u32_be(pos_bytes) if len(pos_bytes) == 4 else None

            run_code = ""
            run_value = ""
            moment_code = ""
            moment_value = ""
            if args.read_extra_every > 0 and samples % args.read_extra_every == 0:
                run_code, run_value = read_single_register(arm, args, REG_RUN)
                moment_code, moment_value = read_single_register(arm, args, REG_MOMENT)

            timestamp_ns = time.time_ns()
            elapsed_s = (time.perf_counter_ns() - start_ns) / 1_000_000_000
            writer.writerow(
                {
                    "sample_index": samples,
                    "timestamp_ns": timestamp_ns,
                    "elapsed_s": f"{elapsed_s:.9f}",
                    "target_hz": args.hz,
                    "position_read_code": pos_code,
                    "position_read_latency_ms": f"{read_latency_ms:.3f}",
                    "position_b0": pos_bytes[0] if len(pos_bytes) == 4 else "",
                    "position_b1": pos_bytes[1] if len(pos_bytes) == 4 else "",
                    "position_b2": pos_bytes[2] if len(pos_bytes) == 4 else "",
                    "position_b3": pos_bytes[3] if len(pos_bytes) == 4 else "",
                    "position_value": "" if pos_value is None else pos_value,
                    "run_read_code": run_code,
                    "run_value": run_value,
                    "moment_read_code": moment_code,
                    "moment_value": moment_value,
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
                    f"last_pos_code={pos_code} last_pos={pos_value}"
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
    parser = argparse.ArgumentParser(
        description="Collect Zhixing CTAG2F120 gripper Modbus register data to CSV."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Robot controller IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Robot API port.")
    parser.add_argument("--output", default="gripper_state.csv", help="Output CSV path.")
    parser.add_argument("--hz", type=float, default=200.0, help="Target polling rate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration seconds. 0 means until Ctrl+C.")
    parser.add_argument("--modbus-port", type=int, default=MODBUS_PORT_TOOL, help="1 means end-tool RS485.")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE, help="Modbus RTU baudrate.")
    parser.add_argument(
        "--modbus-timeout",
        type=int,
        default=MODBUS_TIMEOUT_100MS,
        help="Modbus timeout in 100 ms units. Lower values reduce blocking on failed reads.",
    )
    parser.add_argument("--device", type=int, default=DEVICE_ADDR, help="Modbus slave address.")
    parser.add_argument("--setup-pause", type=float, default=1.0, help="Pause after setup commands.")
    parser.add_argument("--report-interval", type=float, default=1.0, help="Console report interval in seconds.")
    parser.add_argument("--flush-every", type=int, default=50, help="Flush CSV every N samples. 0 disables.")
    parser.add_argument(
        "--read-extra-every",
        type=int,
        default=0,
        help="Read run and moment registers every N samples. 0 disables extra reads.",
    )
    parser.add_argument("--no-setup", action="store_true", help="Do not set 24V or Modbus mode.")
    parser.add_argument("--keep-modbus", action="store_true", help="Do not close Modbus mode on exit.")
    parser.add_argument("--keep-power", action="store_true", help="Do not turn off end-tool power on exit.")
    return parser


def main() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    args = build_parser().parse_args()
    arm = connect(args.host, args.port)
    try:
        setup_gripper_bus(arm, args)
        collect(arm, args)
        return 0
    finally:
        cleanup(arm, args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Minimal Modbus test for Zhixing CTAG2F120 gripper on a RealMan arm.

Reference:
- zhixing_90D_gripper.py
- 睿尔曼机械臂+知行CTAG2F120夹爪操作手册.docx

This gripper is controlled through the end-tool Modbus RTU path, not through
RealMan's built-in GripperControl state API.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_peripheral_read_write_params_t,
    rm_thread_mode_e,
)


DEFAULT_HOST = "172.25.5.243"
DEFAULT_PORT = 8080

MODBUS_PORT_TOOL = 1
BAUDRATE = 115200
MODBUS_TIMEOUT_100MS = 20
DEVICE_ADDR = 1
TOOL_VOLTAGE_24V = 3

REG_POSITION = 258
REG_RUN = 264
REG_MOMENT = 284


ERROR_MESSAGES = {
    0: "success",
    1: "controller returned false / parameter or robot state error",
    -1: "send failed / communication error",
    -2: "receive failed / communication timeout",
    -3: "response parse failed",
    -4: "unsupported by this controller",
}


def status_text(code: int) -> str:
    return ERROR_MESSAGES.get(code, "unknown error")


def position_to_bytes(position: int) -> list[int]:
    """Encode 0..1000 position as 4 big-endian bytes for two registers."""
    if not 0 <= position <= 1000:
        raise ValueError(f"position must be in 0..1000, got {position}")
    return [
        (position >> 24) & 0xFF,
        (position >> 16) & 0xFF,
        (position >> 8) & 0xFF,
        position & 0xFF,
    ]


def params(port: int, address: int, device: int, num: int) -> rm_peripheral_read_write_params_t:
    return rm_peripheral_read_write_params_t(
        port=port,
        address=address,
        device=device,
        num=num,
    )


def print_json(title: str, data: dict[str, Any]) -> None:
    print(f"{title}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def call_api(name: str, func: Callable[[], Any]) -> Any:
    print(f"\n===== {name} =====")
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start

    if isinstance(result, tuple) and result and isinstance(result[0], int):
        code = result[0]
        print(f"return_code: {code} ({status_text(code)})")
        if len(result) > 1:
            print(f"data: {result[1]!r}")
    elif isinstance(result, int):
        print(f"return_code: {result} ({status_text(result)})")
    else:
        print(f"result: {result!r}")

    print(f"elapsed_sec: {elapsed:.3f}")
    return result


class ZhixingGripperTester:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.connected = False
        self.modbus_open = False
        self.tool_power_on = False

    def connect(self) -> None:
        print("===== connect robot =====")
        print(f"target: {self.args.host}:{self.args.port}")
        start = time.perf_counter()
        handle = self.arm.rm_create_robot_arm(self.args.host, self.args.port)
        elapsed = time.perf_counter() - start
        print(f"handle: {handle!r}")
        print(f"handle_id: {None if handle is None else handle.id}")
        print(f"elapsed_sec: {elapsed:.3f}")
        if handle is None or getattr(handle, "id", -1) < 0:
            raise RuntimeError(f"failed to connect robot at {self.args.host}:{self.args.port}")
        self.connected = True

    def setup(self) -> None:
        print_json(
            "manual_step_set_tool_voltage_json",
            {"command": "set_tool_voltage", "voltage_type": TOOL_VOLTAGE_24V},
        )
        code = call_api(
            "set end-tool 24V power",
            lambda: self.arm.rm_set_tool_voltage(TOOL_VOLTAGE_24V),
        )
        if code != 0:
            raise RuntimeError("failed to set end-tool 24V power")
        self.tool_power_on = True
        time.sleep(self.args.setup_pause)

        print_json(
            "manual_step_set_modbus_mode_json",
            {
                "command": "set_modbus_mode",
                "port": self.args.modbus_port,
                "baudrate": self.args.baudrate,
                "timeout": self.args.modbus_timeout,
            },
        )
        code = call_api(
            "set end-tool Modbus RTU mode",
            lambda: self.arm.rm_set_modbus_mode(
                self.args.modbus_port,
                self.args.baudrate,
                self.args.modbus_timeout,
            ),
        )
        if code != 0:
            raise RuntimeError("failed to set end-tool Modbus RTU mode")
        self.modbus_open = True
        time.sleep(self.args.setup_pause)

    def write_single_register(self, address: int, data: int, label: str) -> int:
        p = params(self.args.modbus_port, address, self.args.device, 1)
        print_json(
            "equivalent_json",
            {
                "command": "write_single_register",
                "port": self.args.modbus_port,
                "address": address,
                "data": data,
                "device": self.args.device,
            },
        )
        print(f"sdk_params: port={p.port}, address={p.address}, device={p.device}, num={p.num}")
        return call_api(label, lambda: self.arm.rm_write_single_register(p, data))

    def write_registers(self, address: int, data: list[int], label: str) -> int:
        if len(data) % 2 != 0:
            raise ValueError("write_registers data length must be even")
        num = len(data) // 2
        p = params(self.args.modbus_port, address, self.args.device, num)
        print_json(
            "equivalent_json",
            {
                "command": "write_registers",
                "port": self.args.modbus_port,
                "address": address,
                "num": num,
                "data": data,
                "device": self.args.device,
            },
        )
        print(f"sdk_params: port={p.port}, address={p.address}, device={p.device}, num={p.num}")
        return call_api(label, lambda: self.arm.rm_write_registers(p, data))

    def read_holding_registers(self, address: int, num: int, label: str) -> None:
        p = params(self.args.modbus_port, address, self.args.device, num)
        print(f"sdk_params: port={p.port}, address={p.address}, device={p.device}, num={p.num}")
        if num == 1:
            call_api(label, lambda: self.arm.rm_read_holding_registers(p))
        else:
            call_api(label, lambda: self.arm.rm_read_multiple_holding_registers(p))

    def set_moment(self, moment: int) -> None:
        if not 0 <= moment <= 100:
            raise ValueError(f"moment must be in 0..100, got {moment}")
        code = self.write_single_register(REG_MOMENT, moment, f"set gripper moment {moment}%")
        if code != 0:
            raise RuntimeError("failed to set gripper moment")
        if self.args.read_back:
            self.read_holding_registers(REG_MOMENT, 1, "read moment register")

    def move_position(self, position: int, name: str) -> None:
        data = position_to_bytes(position)
        code = self.write_registers(REG_POSITION, data, f"write {name} position {position}")
        if code != 0:
            raise RuntimeError(f"failed to write {name} position")

        if self.args.read_back:
            self.read_holding_registers(REG_POSITION, 2, f"read position registers after {name}")

        code = self.write_single_register(REG_RUN, 1, f"trigger {name} run")
        if code != 0:
            raise RuntimeError(f"failed to trigger {name} run")

        time.sleep(self.args.action_pause)

    def run(self) -> None:
        self.connect()
        self.setup()
        self.set_moment(self.args.moment)

        print("\n===== start gripper actions =====")
        for index in range(1, self.args.cycles + 1):
            print(f"\n----- cycle {index}/{self.args.cycles} -----")
            if self.args.sequence in ("open-close", "open"):
                self.move_position(self.args.open_position, "open")
            if self.args.sequence in ("open-close", "close"):
                self.move_position(self.args.close_position, "close")
            if self.args.sequence == "position":
                self.move_position(self.args.position, "target")

        print("\n===== test finished =====")

    def cleanup(self) -> None:
        print("\n===== cleanup =====")
        if self.modbus_open:
            call_api(
                "close end-tool Modbus RTU mode",
                lambda: self.arm.rm_close_modbus_mode(self.args.modbus_port),
            )
        if self.tool_power_on and not self.args.keep_power:
            call_api("turn off end-tool power", lambda: self.arm.rm_set_tool_voltage(0))
        if self.connected:
            call_api("delete robot arm handle", self.arm.rm_delete_robot_arm)
        call_api("destroy SDK", self.arm.rm_destroy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detailed Modbus RTU test for Zhixing CTAG2F120 gripper."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Robot controller IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Robot API port.")
    parser.add_argument("--modbus-port", type=int, default=MODBUS_PORT_TOOL, help="1 means end-tool RS485.")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE, help="Modbus RTU baudrate.")
    parser.add_argument(
        "--modbus-timeout",
        type=int,
        default=MODBUS_TIMEOUT_100MS,
        help="Modbus timeout in 100 ms units.",
    )
    parser.add_argument("--device", type=int, default=DEVICE_ADDR, help="Modbus slave address.")
    parser.add_argument("--moment", type=int, default=50, help="Gripper moment percent, 0..100.")
    parser.add_argument("--open-position", type=int, default=0, help="Open position, 0..1000.")
    parser.add_argument("--close-position", type=int, default=1000, help="Close position, 0..1000.")
    parser.add_argument("--position", type=int, default=500, help="Target position for --sequence position.")
    parser.add_argument(
        "--sequence",
        choices=("open-close", "open", "close", "position"),
        default="open-close",
        help="Action sequence to run.",
    )
    parser.add_argument("--cycles", type=int, default=1, help="Number of action cycles.")
    parser.add_argument("--setup-pause", type=float, default=1.0, help="Pause after setup commands.")
    parser.add_argument("--action-pause", type=float, default=2.0, help="Pause after each move command.")
    parser.add_argument("--read-back", action="store_true", help="Try reading related holding registers.")
    parser.add_argument("--keep-power", action="store_true", help="Do not turn off end-tool power on exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tester = ZhixingGripperTester(args)
    try:
        tester.run()
        return 0
    finally:
        tester.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

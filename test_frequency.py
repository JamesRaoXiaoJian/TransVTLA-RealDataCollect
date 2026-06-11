"""数据采集频率实时测试脚本。

逐个测试各传感器的实际采集频率，实时显示：
  - 当前瞬时频率 (最近 1 秒)
  - 平均频率
  - 最小/最大时间间隔
  - 标准差 / 抖动 (jitter)
  - 间隔分布直方图
  - 丢帧检测

用法:
    python test_frequency.py                    # 交互式选择传感器
    python test_frequency.py --sensor pressure  # 直接测试压力传感器
    python test_frequency.py --sensor robot     # 直接测试机械臂
    python test_frequency.py --sensor gripper   # 直接测试夹爪
    python test_frequency.py --sensor dji       # 直接测试 DJI 相机
    python test_frequency.py --sensor realsense # 直接测试 RealSense
    python test_frequency.py --sensor all       # 依次测试全部

按 Ctrl+C 停止当前测试并显示报告。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# 强制 CPU，避免 GPU 初始化卡顿
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ============================================================
# 频率统计核心类
# ============================================================


@dataclass
class FreqStats:
    """实时频率统计数据结构。"""

    name: str
    target_hz: float
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100000))
    lock: threading.Lock = field(default_factory=threading.Lock)
    start_time: float = 0.0
    _last_print_len: int = 0

    def record(self, ts: float | None = None) -> None:
        """记录一个采样点。"""
        if ts is None:
            ts = time.monotonic()
        with self.lock:
            if not self.timestamps:
                self.start_time = ts
            self.timestamps.append(ts)

    def snapshot(self) -> dict:
        """获取当前统计快照（线程安全）。"""
        with self.lock:
            n = len(self.timestamps)
            if n < 2:
                return {
                    "n": n,
                    "elapsed": 0.0,
                    "avg_hz": 0.0,
                    "inst_hz": 0.0,
                    "interval_min": 0.0,
                    "interval_max": 0.0,
                    "interval_mean": 0.0,
                    "interval_std": 0.0,
                    "jitter_pct": 0.0,
                    "dropped_est": 0,
                    "hist": [],
                }
            ts_list = list(self.timestamps)

        elapsed = ts_list[-1] - ts_list[0]
        intervals = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
        n_intervals = len(intervals)

        avg_hz = n_intervals / elapsed if elapsed > 0 else 0.0

        # 瞬时频率：最近 1 秒内的采样数
        now = ts_list[-1]
        recent = sum(1 for t in ts_list if now - t <= 1.0)
        inst_hz = recent  # 最近 1 秒的采样数 ≈ Hz

        interval_min = min(intervals)
        interval_max = max(intervals)
        interval_mean = sum(intervals) / n_intervals
        interval_std = math.sqrt(
            sum((x - interval_mean) ** 2 for x in intervals) / n_intervals
        )
        jitter_pct = (interval_std / interval_mean * 100) if interval_mean > 0 else 0

        # 估算丢帧：基于目标频率和实际采样数
        expected_count = elapsed * self.target_hz
        dropped_est = max(0, int(expected_count - n))

        # 间隔分布直方图（10 个 bin）
        hist_bins = 10
        hist_min = interval_min
        hist_max = min(interval_max, interval_mean * 5)  # 截断异常值
        if hist_max <= hist_min:
            hist_max = hist_min + 1e-6
        bin_width = (hist_max - hist_min) / hist_bins
        hist = [0] * hist_bins
        for iv in intervals:
            idx = int((iv - hist_min) / bin_width)
            idx = max(0, min(hist_bins - 1, idx))
            hist[idx] += 1

        return {
            "n": n,
            "elapsed": elapsed,
            "avg_hz": avg_hz,
            "inst_hz": inst_hz,
            "interval_min": interval_min * 1000,
            "interval_max": interval_max * 1000,
            "interval_mean": interval_mean * 1000,
            "interval_std": interval_std * 1000,
            "jitter_pct": jitter_pct,
            "dropped_est": dropped_est,
            "hist": hist,
            "hist_min_ms": hist_min * 1000,
            "hist_max_ms": hist_max * 1000,
        }


def print_stats(stats: FreqStats, final: bool = False) -> None:
    """打印统计信息到终端。"""
    s = stats.snapshot()

    lines = []
    lines.append(f"\r\033[K{'─' * 70}")
    lines.append(f"  传感器: {stats.name}  |  目标频率: {stats.target_hz:.0f} Hz")
    lines.append(f"{'─' * 70}")

    # 频率
    hz_color = ""
    if s["avg_hz"] > 0:
        ratio = s["avg_hz"] / stats.target_hz
        if ratio >= 0.9:
            hz_color = "\033[32m"  # 绿色
        elif ratio >= 0.7:
            hz_color = "\033[33m"  # 黄色
        else:
            hz_color = "\033[31m"  # 红色
    reset = "\033[0m"

    lines.append(f"  采样数: {s['n']:>8}    耗时: {s['elapsed']:>8.2f} s")
    lines.append(
        f"  平均频率: {hz_color}{s['avg_hz']:>8.2f} Hz{reset}"
        f"    瞬时频率: {hz_color}{s['inst_hz']:>8.2f} Hz{reset}"
    )
    lines.append(
        f"  频率偏差: {abs(s['avg_hz'] - stats.target_hz) / stats.target_hz * 100:>8.2f} %"
        f"    估算丢帧: {s['dropped_est']:>8}"
    )

    # 间隔
    lines.append(f"{'─' * 70}")
    lines.append(f"  时间间隔 (ms):")
    lines.append(
        f"    最小: {s['interval_min']:>8.3f}"
        f"    最大: {s['interval_max']:>8.3f}"
        f"    均值: {s['interval_mean']:>8.3f}"
    )
    lines.append(
        f"    标准差: {s['interval_std']:>6.3f}"
        f"    抖动: {s['jitter_pct']:>8.2f} %"
    )

    # 直方图
    if s["hist"]:
        lines.append(f"{'─' * 70}")
        lines.append(f"  间隔分布 (ms):")
        hist = s["hist"]
        hist_min = s["hist_min_ms"]
        hist_max = s["hist_max_ms"]
        bin_width = (hist_max - hist_min) / len(hist)
        max_count = max(hist) if max(hist) > 0 else 1

        for i, count in enumerate(hist):
            lo = hist_min + i * bin_width
            hi = lo + bin_width
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            pct = count / sum(hist) * 100 if sum(hist) > 0 else 0
            if i == 0 or i == len(hist) - 1 or count > 0:
                lines.append(f"    {lo:7.2f}~{hi:7.2f}: {count:>6} ({pct:5.1f}%) {bar}")

    lines.append(f"{'─' * 70}")

    if final:
        lines.append(f"  ✅ 测试结束")
    else:
        lines.append(f"  按 Ctrl+C 停止测试...")

    output = "\n".join(lines)
    if not final:
        # 上移光标覆盖之前的输出
        n_lines = stats._last_print_len
        if n_lines > 0:
            sys.stdout.write(f"\033[{n_lines}A")
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    stats._last_print_len = len(lines)


# ============================================================
# 各传感器测试函数
# ============================================================


def test_pressure(stats: FreqStats) -> None:
    """测试压力传感器采集频率。"""
    from collectors.pressure import (
        DEFAULT_PRESSURE_LOCAL_PORT,
        DEFAULT_PRESSURE_REMOTE_IP,
        DEFAULT_PRESSURE_REMOTE_PORT,
        PressureCollector,
    )

    collector = PressureCollector(
        local_port=DEFAULT_PRESSURE_LOCAL_PORT,
        remote_ip=DEFAULT_PRESSURE_REMOTE_IP,
        remote_port=DEFAULT_PRESSURE_REMOTE_PORT,
    )
    collector.start()

    # hook: 在接收线程中记录时间戳
    original_recv = collector._recv_loop

    def patched_recv():
        import struct
        while collector.running:
            try:
                data, addr = collector.sock.recvfrom(4096)
                if len(data) >= 8:
                    stats.record()
                    # 同时更新 collector 的内部状态
                    with collector.lock:
                        collector.latest_timestamp_us = struct.unpack_from("<Q", data, 0)[0]
                        values = struct.unpack_from("<64h", data, 8)
                        collector.latest_values = list(values)
            except Exception:
                pass

    collector._recv_loop = patched_recv
    collector.thread = threading.Thread(target=patched_recv, daemon=True)
    collector.thread.start()

    return collector


def test_robot(stats: FreqStats) -> None:
    """测试机械臂状态采集频率。"""
    from collectors.robot_arm import DEFAULT_ARM_HOST, DEFAULT_ARM_PORT, RobotArmCollector

    collector = RobotArmCollector(host=DEFAULT_ARM_HOST, port=DEFAULT_ARM_PORT)
    collector.connect()

    # hook: 在轮询线程中记录时间戳
    original_poll = collector._poll_loop

    def patched_poll():
        while collector.running:
            try:
                code, data = collector.robot.rm_get_arm_current_state()
                stats.record()
                with collector.lock:
                    collector.latest_state = {"code": code, "data": data}
            except Exception:
                pass
            time.sleep(collector.interval_s)

    collector.running = True
    collector._poll_loop = patched_poll
    collector.thread = threading.Thread(target=patched_poll, daemon=True)
    collector.thread.start()

    return collector


def test_gripper(stats: FreqStats) -> None:
    """测试夹爪状态采集频率。"""
    from collectors.gripper_state import GripperStateCollector
    from collectors.robot_arm import DEFAULT_ARM_HOST, DEFAULT_ARM_PORT

    collector = GripperStateCollector(host=DEFAULT_ARM_HOST, port=DEFAULT_ARM_PORT)
    collector.connect()

    def patched_poll():
        while collector.running:
            try:
                code, dist = collector.robot.rm_get_rm_plus_state_info()
                stats.record()
                with collector.lock:
                    gripper_pos = dist["pos"][0] if code == 0 and dist else None
                    collector.latest_state = {
                        "code": code,
                        "gripper_pos": gripper_pos,
                        "latency_ms": None,
                    }
            except Exception:
                pass
            time.sleep(collector.interval_s)

    collector.running = True
    collector._poll_loop = patched_poll
    collector.thread = threading.Thread(target=patched_poll, daemon=True)
    collector.thread.start()

    return collector


def test_dji(stats: FreqStats) -> None:
    """测试 DJI 相机采集频率。"""
    from collectors.dji_camera import DJICamera

    collector = DJICamera()
    collector.start()

    def patched_loop():
        while collector.running:
            frame = collector.read()
            if frame is not None:
                stats.record()
            time.sleep(0.001)

    collector.running = True
    collector.thread = threading.Thread(target=patched_loop, daemon=True)
    collector.thread.start()

    return collector


def test_realsense(stats: FreqStats) -> None:
    """测试 RealSense 相机采集频率。"""
    from collectors.realsense_rgb import RealSenseRGB

    collector = RealSenseRGB()
    collector.start()

    def patched_loop():
        while collector.running:
            frame = collector.read()
            if frame is not None:
                stats.record()
            time.sleep(0.001)

    collector.running = True
    collector.thread = threading.Thread(target=patched_loop, daemon=True)
    collector.thread.start()

    return collector


# ============================================================
# 传感器注册表
# ============================================================

SENSOR_REGISTRY = {
    "pressure": {
        "name": "压力传感器 (UDP)",
        "target_hz": 200,
        "test_fn": test_pressure,
        "requires_network": True,
    },
    "robot": {
        "name": "机械臂状态",
        "target_hz": 100,
        "test_fn": test_robot,
        "requires_network": True,
    },
    "gripper": {
        "name": "夹爪状态 (RM Plus)",
        "target_hz": 200,
        "test_fn": test_gripper,
        "requires_network": True,
    },
    "dji": {
        "name": "DJI 相机",
        "target_hz": 20,
        "test_fn": test_dji,
        "requires_network": False,
    },
    "realsense": {
        "name": "RealSense 相机",
        "target_hz": 20,
        "test_fn": test_realsense,
        "requires_network": False,
    },
}


# ============================================================
# 主流程
# ============================================================


def run_single(sensor_key: str) -> None:
    """测试单个传感器。"""
    cfg = SENSOR_REGISTRY[sensor_key]
    stats = FreqStats(name=cfg["name"], target_hz=cfg["target_hz"])

    print(f"\n🚀 开始测试: {cfg['name']} (目标 {cfg['target_hz']} Hz)")
    print(f"   按 Ctrl+C 停止\n")

    collector = None
    try:
        collector = cfg["test_fn"](stats)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return

    try:
        while True:
            time.sleep(1.0)
            print_stats(stats)
    except KeyboardInterrupt:
        pass
    finally:
        if collector and hasattr(collector, "stop"):
            collector.stop()
        print_stats(stats, final=True)


def run_all() -> None:
    """依次测试全部传感器。"""
    for key in SENSOR_REGISTRY:
        print(f"\n{'=' * 70}")
        print(f"  即将测试: {SENSOR_REGISTRY[key]['name']}")
        print(f"  按 Enter 开始，输入 s 跳过，q 退出")
        print(f"{'=' * 70}")

        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break
        if choice == "s":
            continue

        run_single(key)


def interactive_menu() -> None:
    """交互式菜单。"""
    print("\n" + "=" * 70)
    print("  数据采集频率实时测试工具")
    print("=" * 70)

    keys = list(SENSOR_REGISTRY.keys())
    for i, key in enumerate(keys):
        cfg = SENSOR_REGISTRY[key]
        print(f"  [{i + 1}] {cfg['name']:<25} 目标 {cfg['target_hz']:>3} Hz")

    print(f"  [A] 依次测试全部")
    print(f"  [Q] 退出")
    print("=" * 70)

    try:
        choice = input("\n  选择传感器编号: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "q":
        return
    if choice == "a":
        run_all()
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            run_single(keys[idx])
        else:
            print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入数字")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="数据采集频率实时测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sensor",
        "-s",
        choices=list(SENSOR_REGISTRY.keys()) + ["all"],
        default=None,
        help="直接指定要测试的传感器",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="测试时长（秒），不指定则持续到 Ctrl+C",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sensor:
        if args.sensor == "all":
            run_all()
        else:
            if args.duration:
                # 定时模式
                cfg = SENSOR_REGISTRY[args.sensor]
                stats = FreqStats(name=cfg["name"], target_hz=cfg["target_hz"])
                print(f"\n🚀 测试 {cfg['name']}，时长 {args.duration} 秒\n")
                collector = None
                try:
                    collector = cfg["test_fn"](stats)
                except Exception as e:
                    print(f"❌ 启动失败: {e}")
                    return

                try:
                    end_time = time.monotonic() + args.duration
                    while time.monotonic() < end_time:
                        time.sleep(1.0)
                        print_stats(stats)
                except KeyboardInterrupt:
                    pass
                finally:
                    if collector and hasattr(collector, "stop"):
                        collector.stop()
                    print_stats(stats, final=True)
            else:
                run_single(args.sensor)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()

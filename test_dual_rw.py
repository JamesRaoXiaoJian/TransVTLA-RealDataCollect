"""双 RealSense 实际读写频率测试。

同时启动两台 RealSense，实际读取 RGB + Depth 帧，
测量：
  - world_camera 单独读取频率
  - wrist_camera 单独读取频率
  - 双相机同步帧对频率
  - 每帧实际耗时（含 depth 读取）

用法:
    python test_dual_rw.py              # 默认运行 30 秒
    python test_dual_rw.py -d 60        # 运行 60 秒
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque

import pyrealsense2 as rs
import numpy as np

from realsense_standard import STANDARD_RS_FPS, STANDARD_RS_HEIGHT, STANDARD_RS_WIDTH


# ============================================================
# 统计类
# ============================================================

class Stats:
    def __init__(self, name: str, target_hz: float):
        self.name = name
        self.target_hz = target_hz
        self.timestamps: deque = deque(maxlen=100000)
        self.lock = threading.Lock()
        self.start_time = 0.0
        self._last_lines = 0

    def record(self):
        ts = time.monotonic()
        with self.lock:
            if not self.timestamps:
                self.start_time = ts
            self.timestamps.append(ts)

    def snapshot(self) -> dict:
        with self.lock:
            n = len(self.timestamps)
            if n < 2:
                return {"n": n, "elapsed": 0, "avg_hz": 0, "inst_hz": 0,
                        "min_ms": 0, "max_ms": 0, "mean_ms": 0, "std_ms": 0, "jitter": 0}
            ts = list(self.timestamps)
        elapsed = ts[-1] - ts[0]
        intervals = [(ts[i+1] - ts[i]) * 1000 for i in range(len(ts)-1)]
        avg_hz = (n - 1) / elapsed if elapsed > 0 else 0
        recent = sum(1 for t in ts if ts[-1] - t <= 1.0)
        mean_iv = sum(intervals) / len(intervals)
        std_iv = math.sqrt(sum((x - mean_iv)**2 for x in intervals) / len(intervals))
        return {
            "n": n, "elapsed": elapsed, "avg_hz": avg_hz, "inst_hz": recent,
            "min_ms": min(intervals), "max_ms": max(intervals),
            "mean_ms": mean_iv, "std_ms": std_iv,
            "jitter": std_iv / mean_iv * 100 if mean_iv > 0 else 0,
        }

    def print_stats(self, final=False):
        s = self.snapshot()
        lines = [
            f"{'─' * 60}",
            f"  {self.name}  |  目标: {self.target_hz:.0f} Hz",
            f"{'─' * 60}",
        ]
        color = ""
        if s["avg_hz"] > 0:
            ratio = s["avg_hz"] / self.target_hz
            color = "\033[32m" if ratio >= 0.9 else "\033[33m" if ratio >= 0.7 else "\033[31m"
        rst = "\033[0m"
        lines.append(f"  采样: {s['n']:>6}  耗时: {s['elapsed']:>7.1f}s")
        lines.append(f"  平均: {color}{s['avg_hz']:>7.2f} Hz{rst}  瞬时: {color}{s['inst_hz']:>5} Hz{rst}")
        lines.append(f"  间隔: min={s['min_ms']:.2f}  max={s['max_ms']:.2f}  "
                     f"mean={s['mean_ms']:.2f}  std={s['std_ms']:.2f}  抖动={s['jitter']:.1f}%")
        lines.append(f"{'─' * 60}")
        if final:
            lines.append(f"  ✅ 测试结束")
        # 打印
        if self._last_lines > 0 and not final:
            sys.stdout.write(f"\033[{self._last_lines}A")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._last_lines = len(lines)


# ============================================================
# 采集线程
# ============================================================

def camera_loop(serial: str, name: str, stats: Stats, stop_event: threading.Event,
                width: int, height: int, fps: int):
    """单台 RealSense 采集线程，实际读取 RGB + Depth。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"  ❌ {name} ({serial}) 启动失败: {e}")
        return

    # 丢弃前几帧（自动曝光收敛）
    for _ in range(10):
        pipeline.wait_for_frames()

    print(f"  ✅ {name} ({serial}) 已启动 {width}x{height}@{fps}fps")

    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(5000)  # 5s 超时
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame and depth_frame:
                # 实际读取数据到 numpy（模拟真实写入/处理）
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                stats.record()
        except Exception as e:
            if not stop_event.is_set():
                print(f"  ⚠️ {name} 帧获取异常: {e}")
            break

    pipeline.stop()
    print(f"  {name} 已停止")


# ============================================================
# 同步监控
# ============================================================

def sync_monitor(stats_world: Stats, stats_wrist: Stats, stats_sync: Stats,
                 stop_event: threading.Event):
    """监控同步帧对：取两边最小帧数。"""
    last_pairs = 0
    while not stop_event.is_set():
        pairs = min(stats_world.n(), stats_wrist.n())
        if pairs > last_pairs:
            for _ in range(pairs - last_pairs):
                stats_sync.record()
            last_pairs = pairs
        time.sleep(0.005)


# 给 Stats 加个 n() 方法
def _stats_n(self):
    with self.lock:
        return len(self.timestamps)
Stats.n = _stats_n


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="双 RealSense 实际读写频率测试")
    parser.add_argument("-d", "--duration", type=float, default=30, help="测试时长（秒）")
    parser.add_argument("--width", type=int, default=STANDARD_RS_WIDTH)
    parser.add_argument("--height", type=int, default=STANDARD_RS_HEIGHT)
    parser.add_argument("--fps", type=int, default=STANDARD_RS_FPS)
    args = parser.parse_args()

    # 枚举设备
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        print(f"❌ 需要 2 台 RealSense，当前检测到 {len(devices)} 台")
        for i, d in enumerate(devices):
            print(f"  [{i}] {d.get_info(rs.camera_info.name)} - {d.get_info(rs.camera_info.serial_number)}")
        return

    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    world_serial, wrist_serial = serials[0], serials[1]

    print(f"\n{'=' * 60}")
    print(f"  双 RealSense 实际读写频率测试")
    print(f"{'=' * 60}")
    print(f"  world_camera: {world_serial}")
    print(f"  wrist_camera: {wrist_serial}")
    print(f"  分辨率: {args.width}x{args.height} @ {args.fps}fps")
    print(f"  测试时长: {args.duration}s")
    print(f"{'=' * 60}\n")

    stats_world = Stats("world_camera (单独)", args.fps)
    stats_wrist = Stats("wrist_camera (单独)", args.fps)
    stats_sync = Stats("双相机同步帧对", args.fps)

    stop_event = threading.Event()

    # 启动采集线程
    t_world = threading.Thread(target=camera_loop,
                               args=(world_serial, "world_camera", stats_world, stop_event,
                                     args.width, args.height, args.fps), daemon=True)
    t_wrist = threading.Thread(target=camera_loop,
                               args=(wrist_serial, "wrist_camera", stats_wrist, stop_event,
                                     args.width, args.height, args.fps), daemon=True)
    t_sync = threading.Thread(target=sync_monitor,
                              args=(stats_world, stats_wrist, stats_sync, stop_event), daemon=True)

    t_world.start()
    t_wrist.start()
    t_sync.start()

    # 实时显示
    end_time = time.monotonic() + args.duration
    try:
        while time.monotonic() < end_time:
            time.sleep(1.0)
            stats_world.print_stats()
            stats_wrist.print_stats()
            stats_sync.print_stats()
            remaining = end_time - time.monotonic()
            print(f"  ⏱️  剩余 {remaining:.0f}s ...")
    except KeyboardInterrupt:
        print("\n  手动中断")

    stop_event.set()
    time.sleep(1.0)

    # 最终结果
    print(f"\n{'=' * 60}")
    print(f"  最终结果")
    print(f"{'=' * 60}")
    stats_world.print_stats(final=True)
    stats_wrist.print_stats(final=True)
    stats_sync.print_stats(final=True)
    print()


if __name__ == "__main__":
    main()

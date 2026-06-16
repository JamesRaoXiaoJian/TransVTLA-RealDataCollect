"""自动化全流程频率测试 — 每个传感器测 40 秒，失败立即停止。

目标频率:
  - 压力传感器: 200 Hz
  - 机械臂: 120 Hz (4×30Hz)
  - 夹爪: 120 Hz (4×30Hz)
  - RealSense: 30 Hz
"""
from __future__ import annotations

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from test_frequency import FreqStats, SENSOR_REGISTRY, print_stats

DURATION = 40  # 每个传感器测试秒数


def run_test(sensor_key: str) -> bool:
    """测试单个传感器，返回 True=成功，False=失败。"""
    cfg = SENSOR_REGISTRY[sensor_key]
    stats = FreqStats(name=cfg["name"], target_hz=cfg["target_hz"])

    print(f"\n{'='*70}")
    print(f"  测试: {cfg['name']}  |  目标 {cfg['target_hz']} Hz  |  时长 {DURATION}s")
    print(f"{'='*70}")

    collector = None
    try:
        collector = cfg["test_fn"](stats)
    except Exception as e:
        print(f"\n  [FAIL] 启动异常: {e}")
        return False

    # 检查是否真的启动了（realsense 的 available 属性）
    if hasattr(collector, "available") and not collector.available:
        print(f"\n  [FAIL] {cfg['name']} 未能启动（设备不可用）")
        if hasattr(collector, "stop"):
            collector.stop()
        return False

    # 等待 2 秒看看是否有数据
    time.sleep(2.0)
    s = stats.snapshot()
    if s["n"] == 0 and sensor_key in ("world_camera", "wrist_camera", "dual_realsense"):
        # RealSense 类传感器如果 2 秒内 0 帧，可能是 pipeline 没真正启动
        if hasattr(collector, "available") and not collector.available:
            print(f"\n  [FAIL] {cfg['name']} 2 秒内 0 帧，设备未就绪")
            if hasattr(collector, "stop"):
                collector.stop()
            return False

    # 正式采集
    try:
        end_time = time.monotonic() + DURATION
        while time.monotonic() < end_time:
            time.sleep(1.0)
            print_stats(stats)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        if collector and hasattr(collector, "stop"):
            collector.stop()
        print_stats(stats, final=True)

    # 最终检查
    final = stats.snapshot()
    if final["n"] < 10:
        print(f"\n  [WARN] 采样数过少 ({final['n']}), 传感器可能未正常工作")
        return False

    return True


def main():
    sensors = ["pressure", "robot", "gripper", "world_camera", "wrist_camera", "dual_realsense"]
    results = {}

    print("\n" + "=" * 70)
    print("  全流程频率自动化测试")
    print(f"  每个传感器 {DURATION} 秒，共 {len(sensors)} 个")
    print("=" * 70)

    for key in sensors:
        ok = run_test(key)
        results[key] = ok
        if not ok:
            print(f"\n{'!'*70}")
            print(f"  传感器 {SENSOR_REGISTRY[key]['name']} 测试失败，停止后续测试")
            print(f"{'!'*70}")
            break

    # 汇总
    print("\n" + "=" * 70)
    print("  测试汇总")
    print("=" * 70)
    for key in sensors:
        if key in results:
            status = "PASS" if results[key] else "FAIL"
            print(f"  [{status}] {SENSOR_REGISTRY[key]['name']}")
        else:
            print(f"  [SKIP] {SENSOR_REGISTRY[key]['name']}")
    print("=" * 70)

    if all(results.get(k) for k in sensors):
        print("\n  All sensors passed!")
    else:
        print("\n  Some sensors failed — see above for details.")


if __name__ == "__main__":
    main()

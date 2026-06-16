"""全面数据质量检查工具。

对所有 session 进行详细检查，包括：
  - 文件结构完整性
  - 各传感器采样率
  - 时间戳单调性与异常
  - 图像完整性
  - aligned_timesteps 对齐质量
  - 生成汇总报告

用法:
    python check_all_sessions.py                    # 检查所有 session
    python check_all_sessions.py --latest 2         # 只检查最新 2 个
    python check_all_sessions.py --session SESSION  # 检查指定 session
    python check_all_sessions.py --export report.csv # 导出 CSV 报告
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SensorStats:
    """单个传感器的统计结果。"""
    name: str
    row_count: int = 0
    duration_s: float = 0.0
    mean_hz: float = 0.0
    median_hz: float = 0.0
    min_interval_ms: float = 0.0
    max_interval_ms: float = 0.0
    mean_interval_ms: float = 0.0
    std_interval_ms: float = 0.0
    gap_count: int = 0          # >2x 平均间隔
    max_gap_ms: float = 0.0
    near_dup_count: int = 0     # <5μs 的间隔
    nan_count: int = 0
    timestamp_monotonic: bool = True
    anomalies: list[str] = field(default_factory=list)


@dataclass
class ImageStats:
    """图像目录统计。"""
    name: str
    count: int = 0
    expected: int = 0
    missing: list[int] = field(default_factory=list)
    min_size_kb: float = 0.0
    max_size_kb: float = 0.0
    mean_size_kb: float = 0.0


@dataclass
class SessionReport:
    """单个 session 的完整报告。"""
    session_name: str
    session_path: str
    total_size_mb: float = 0.0

    # 文件结构
    has_frames_csv: bool = False
    has_aligned_csv: bool = False
    has_camera_meta: bool = False
    has_session_meta: bool = False

    # 各传感器
    robot: Optional[SensorStats] = None
    gripper: Optional[SensorStats] = None
    pressure: Optional[SensorStats] = None
    frames: Optional[SensorStats] = None

    # 图像
    world_rgb: Optional[ImageStats] = None
    world_depth: Optional[ImageStats] = None
    wrist_rgb: Optional[ImageStats] = None
    wrist_depth: Optional[ImageStats] = None

    # 对齐质量
    robot_offset_mean_ms: float = 0.0
    robot_offset_p95_ms: float = 0.0
    gripper_offset_mean_ms: float = 0.0
    gripper_offset_p95_ms: float = 0.0
    pressure_offset_mean_ms: float = 0.0
    pressure_offset_p95_ms: float = 0.0

    # 问题汇总
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================
# 分析函数
# ============================================================

def analyze_csv_timestamps(path: Path, ts_col_name: str, name: str) -> SensorStats:
    """分析 CSV 文件的时间戳统计。"""
    stats = SensorStats(name=name)

    if not path.exists():
        stats.anomalies.append(f"文件不存在: {path}")
        return stats

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    stats.row_count = len(rows)
    if stats.row_count < 2:
        stats.anomalies.append("数据行不足 2 行")
        return stats

    # 找时间戳列
    if ts_col_name not in header:
        # 尝试其他名称
        for alt in ["timestamp_us", "host_monotonic_us", "capture_monotonic_us"]:
            if alt in header:
                ts_col_name = alt
                break
        else:
            stats.anomalies.append(f"找不到时间戳列 {ts_col_name}")
            return stats

    ts_idx = header.index(ts_col_name)
    timestamps = []
    for i, row in enumerate(rows):
        try:
            timestamps.append(int(row[ts_idx]))
        except (ValueError, IndexError):
            stats.nan_count += 1

    if len(timestamps) < 2:
        return stats

    # 基本统计
    stats.duration_s = (timestamps[-1] - timestamps[0]) / 1e6
    intervals_us = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    # 单调性检查
    for i, iv in enumerate(intervals_us):
        if iv < 0:
            stats.timestamp_monotonic = False
            stats.anomalies.append(f"时间戳回退: row {i}, delta={iv}μs")

    # 过滤正值间隔
    pos_intervals = [iv for iv in intervals_us if iv > 0]
    if not pos_intervals:
        return stats

    intervals_ms = [iv / 1000.0 for iv in pos_intervals]
    mean_iv = statistics.mean(intervals_ms)
    stats.mean_interval_ms = mean_iv
    stats.median_interval_ms = statistics.median(intervals_ms) if hasattr(stats, 'median_interval_ms') else mean_iv
    stats.std_interval_ms = statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0
    stats.min_interval_ms = min(intervals_ms)
    stats.max_interval_ms = max(intervals_ms)
    stats.mean_hz = 1000.0 / mean_iv if mean_iv > 0 else 0

    # median Hz
    med_iv = statistics.median(intervals_ms)
    stats.median_hz = 1000.0 / med_iv if med_iv > 0 else 0

    # gap 检测 (>2x 平均间隔)
    gap_threshold = mean_iv * 2
    for i, iv in enumerate(intervals_ms):
        if iv > gap_threshold:
            stats.gap_count += 1
            if iv > stats.max_gap_ms:
                stats.max_gap_ms = iv

    # 近重复时间戳 (<5μs)
    stats.near_dup_count = sum(1 for iv in pos_intervals if iv < 5)

    # NaN 检测
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val.strip() == "" or val.strip().lower() in ("nan", "null", "none"):
                stats.nan_count += 1

    return stats


def analyze_images(directory: Path, name: str, expected: int = 0) -> ImageStats:
    """分析图像目录。"""
    stats = ImageStats(name=name)

    if not directory.exists():
        return stats

    files = sorted(directory.glob("*"))
    files = [f for f in files if f.is_file()]
    stats.count = len(files)
    stats.expected = expected

    if not files:
        return stats

    sizes = [f.stat().st_size for f in files]
    stats.min_size_kb = min(sizes) / 1024
    stats.max_size_kb = max(sizes) / 1024
    stats.mean_size_kb = statistics.mean(sizes) / 1024

    # 检查连续性
    if files[0].suffix in (".jpg", ".png"):
        ids = set()
        for f in files:
            try:
                ids.add(int(f.stem))
            except ValueError:
                pass
        if ids and expected > 0:
            expected_set = set(range(1, expected + 1))
            stats.missing = sorted(expected_set - ids)

    return stats


def analyze_aligned(path: Path) -> dict:
    """分析 aligned_timesteps.csv 的对齐质量。"""
    result = {}

    if not path.exists():
        return result

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    result["row_count"] = len(rows)

    # 分析 offset 列
    for col_name in ["robot_offset_ms", "gripper_offset_ms", "pressure_offset_ms"]:
        if col_name not in header:
            continue
        col_idx = header.index(col_name)
        vals = []
        for row in rows:
            try:
                vals.append(abs(float(row[col_idx])))
            except (ValueError, IndexError):
                pass
        if vals:
            s = sorted(vals)
            n = len(s)
            result[f"{col_name}_mean"] = statistics.mean(vals)
            result[f"{col_name}_p95"] = s[int(n * 0.95)]
            result[f"{col_name}_max"] = s[-1]
            result[f"{col_name}_over10ms"] = sum(1 for v in vals if v > 10)

    return result


def get_dir_size(path: Path) -> float:
    """获取目录大小 (MB)。"""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 * 1024)


# ============================================================
# 主检查函数
# ============================================================

def check_session(session_path: Path) -> SessionReport:
    """检查单个 session。"""
    report = SessionReport(
        session_name=session_path.name,
        session_path=str(session_path),
        total_size_mb=get_dir_size(session_path),
    )

    # 1. 文件结构检查
    report.has_frames_csv = (session_path / "frames.csv").exists()
    report.has_aligned_csv = (session_path / "aligned_timesteps.csv").exists()
    report.has_camera_meta = (session_path / "camera_metadata.json").exists()
    report.has_session_meta = (session_path / "session_meta.json").exists()

    if not report.has_frames_csv:
        report.issues.append("缺少 frames.csv")
    if not report.has_aligned_csv:
        report.issues.append("缺少 aligned_timesteps.csv")
    if not report.has_camera_meta:
        report.warnings.append("缺少 camera_metadata.json")

    # 2. 传感器分析
    robot_path = session_path / "robot_state" / "robot_state.csv"
    gripper_path = session_path / "robot_state" / "gripper_state.csv"
    pressure_path = session_path / "pressure" / "pressure.csv"
    frames_path = session_path / "frames.csv"

    report.robot = analyze_csv_timestamps(robot_path, "timestamp_us", "robot_state")
    report.gripper = analyze_csv_timestamps(gripper_path, "timestamp_us", "gripper_state")
    report.pressure = analyze_csv_timestamps(pressure_path, "host_monotonic_us", "pressure")
    report.frames = analyze_csv_timestamps(frames_path, "capture_monotonic_us", "frames")

    # 3. 图像分析
    frame_count = report.frames.row_count if report.frames else 0

    report.world_rgb = analyze_images(
        session_path / "world_camera" / "rgb", "world_rgb", frame_count)
    report.world_depth = analyze_images(
        session_path / "world_camera" / "depth", "world_depth", frame_count)
    report.wrist_rgb = analyze_images(
        session_path / "wrist_camera" / "rgb", "wrist_rgb", frame_count)
    report.wrist_depth = analyze_images(
        session_path / "wrist_camera" / "depth", "wrist_depth", frame_count)

    # 4. 对齐质量
    aligned_info = analyze_aligned(session_path / "aligned_timesteps.csv")
    report.robot_offset_mean_ms = aligned_info.get("robot_offset_ms_mean", 0)
    report.robot_offset_p95_ms = aligned_info.get("robot_offset_ms_p95", 0)
    report.gripper_offset_mean_ms = aligned_info.get("gripper_offset_ms_mean", 0)
    report.gripper_offset_p95_ms = aligned_info.get("gripper_offset_ms_p95", 0)
    report.pressure_offset_mean_ms = aligned_info.get("pressure_offset_ms_mean", 0)
    report.pressure_offset_p95_ms = aligned_info.get("pressure_offset_ms_p95", 0)

    # 5. 问题检测
    # 相机 FPS 检查
    if report.frames and report.frames.mean_hz > 0:
        cam_hz = report.frames.mean_hz
        if cam_hz < 25:
            report.issues.append(f"相机实际帧率 {cam_hz:.1f} Hz，低于目标 30 Hz")

    # 传感器采样率检查
    if report.robot and report.robot.mean_hz > 0:
        if report.robot.mean_hz < 100:
            report.warnings.append(
                f"robot_state 采样率 {report.robot.mean_hz:.1f} Hz，低于 120 Hz 目标")
        if report.robot.gap_count > 0:
            report.warnings.append(
                f"robot_state 有 {report.robot.gap_count} 个大间隔，最大 {report.robot.max_gap_ms:.1f} ms")
        if report.robot.near_dup_count > 10:
            report.warnings.append(
                f"robot_state 有 {report.robot.near_dup_count} 个近重复时间戳 (<5μs)")

    if report.gripper and report.gripper.mean_hz > 0:
        if report.gripper.mean_hz < 100:
            report.warnings.append(
                f"gripper_state 采样率 {report.gripper.mean_hz:.1f} Hz，低于 120 Hz 目标")

    if report.pressure and report.pressure.mean_hz > 0:
        if report.pressure.mean_hz < 120:
            report.warnings.append(
                f"pressure 采样率 {report.pressure.mean_hz:.1f} Hz，低于 200 Hz 目标")

    # 图像完整性
    for img_stats in [report.world_rgb, report.world_depth, report.wrist_rgb, report.wrist_depth]:
        if img_stats and img_stats.missing:
            report.issues.append(f"{img_stats.name} 缺失 {len(img_stats.missing)} 帧: {img_stats.missing[:10]}")

    # 对齐质量
    if report.robot_offset_p95_ms > 20:
        report.warnings.append(f"robot 对齐 p95 offset {report.robot_offset_p95_ms:.1f} ms > 20ms")

    return report


# ============================================================
# 输出函数
# ============================================================

def print_report(report: SessionReport, verbose: bool = False) -> None:
    """打印单个 session 的报告。"""
    W = 70

    print(f"\n{'═' * W}")
    print(f"  Session: {report.session_name}")
    print(f"  路径: {report.session_path}")
    print(f"  大小: {report.total_size_mb:.1f} MB")
    print(f"{'═' * W}")

    # 文件结构
    print(f"\n  📁 文件结构")
    status = lambda ok: "✅" if ok else "❌"
    print(f"    {status(report.has_frames_csv)} frames.csv")
    print(f"    {status(report.has_aligned_csv)} aligned_timesteps.csv")
    print(f"    {status(report.has_camera_meta)} camera_metadata.json")
    print(f"    {status(report.has_session_meta)} session_meta.json")

    # 各传感器
    def print_sensor(s: Optional[SensorStats], target_hz: float = 0):
        if not s:
            print(f"    ❌ 数据不存在")
            return
        hz_ok = s.mean_hz >= target_hz * 0.9 if target_hz > 0 else True
        hz_icon = "✅" if hz_ok else "⚠️"
        print(f"    行数: {s.row_count}    时长: {s.duration_s:.2f}s")
        print(f"    {hz_icon} 采样率: {s.mean_hz:.1f} Hz (median {s.median_hz:.1f})")
        if target_hz > 0:
            print(f"       目标: {target_hz} Hz  偏差: {abs(s.mean_hz - target_hz) / target_hz * 100:.1f}%")
        print(f"    间隔: mean={s.mean_interval_ms:.2f}  std={s.std_interval_ms:.2f}  "
              f"min={s.min_interval_ms:.2f}  max={s.max_interval_ms:.2f} ms")
        if s.gap_count > 0:
            print(f"    ⚠️ 大间隔: {s.gap_count} 个 (max {s.max_gap_ms:.1f} ms)")
        if s.near_dup_count > 0:
            print(f"    ⚠️ 近重复时间戳: {s.near_dup_count} 个 (<5μs)")
        if s.nan_count > 0:
            print(f"    ❌ NaN/空值: {s.nan_count}")
        if not s.timestamp_monotonic:
            print(f"    ❌ 时间戳非单调!")
        for a in s.anomalies:
            print(f"    ❌ {a}")

    print(f"\n  🤖 robot_state (目标 120 Hz)")
    print_sensor(report.robot, 120)

    print(f"\n  🦾 gripper_state (目标 120 Hz)")
    print_sensor(report.gripper, 120)

    print(f"\n  📊 pressure (目标 200 Hz)")
    print_sensor(report.pressure, 200)

    print(f"\n  📷 frames.csv (目标 30 Hz)")
    print_sensor(report.frames, 30)

    # 图像
    def print_img(s: Optional[ImageStats]):
        if not s:
            print(f"    ❌ 目录不存在")
            return
        ok = s.count == s.expected if s.expected > 0 else s.count > 0
        icon = "✅" if ok and not s.missing else "❌"
        print(f"    {icon} 数量: {s.count}  期望: {s.expected}")
        if s.missing:
            print(f"    ❌ 缺失帧: {s.missing[:20]}")
        if s.count > 0:
            print(f"    大小: {s.min_size_kb:.1f} - {s.max_size_kb:.1f} KB (mean {s.mean_size_kb:.1f})")

    print(f"\n  🖼️ world_camera/rgb")
    print_img(report.world_rgb)
    print(f"\n  🖼️ world_camera/depth")
    print_img(report.world_depth)
    print(f"\n  🖼️ wrist_camera/rgb")
    print_img(report.wrist_rgb)
    print(f"\n  🖼️ wrist_camera/depth")
    print_img(report.wrist_depth)

    # 对齐质量
    print(f"\n  🔗 对齐质量 (offset, ms)")
    print(f"    robot:    mean={report.robot_offset_mean_ms:.2f}  p95={report.robot_offset_p95_ms:.2f}")
    print(f"    gripper:  mean={report.gripper_offset_mean_ms:.2f}  p95={report.gripper_offset_p95_ms:.2f}")
    print(f"    pressure: mean={report.pressure_offset_mean_ms:.2f}  p95={report.pressure_offset_p95_ms:.2f}")

    # 问题汇总
    if report.issues:
        print(f"\n  ❌ 问题 ({len(report.issues)})")
        for issue in report.issues:
            print(f"    • {issue}")

    if report.warnings:
        print(f"\n  ⚠️ 警告 ({len(report.warnings)})")
        for w in report.warnings:
            print(f"    • {w}")

    if not report.issues and not report.warnings:
        print(f"\n  ✅ 无问题")

    print(f"{'─' * W}")


def print_summary(reports: list[SessionReport]) -> None:
    """打印汇总表格。"""
    W = 100
    print(f"\n{'═' * W}")
    print(f"  汇总 ({len(reports)} 个 session)")
    print(f"{'═' * W}")

    # 表头
    print(f"  {'Session':<28} {'Frames':>6} {'CamHz':>7} {'RobotHz':>8} {'GripHz':>8} {'PressHz':>8} {'Issues':>7} {'Warn':>5}")
    print(f"  {'─' * 28} {'─' * 6} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 5}")

    for r in reports:
        frames = r.frames.row_count if r.frames else 0
        cam_hz = r.frames.mean_hz if r.frames else 0
        rob_hz = r.robot.mean_hz if r.robot else 0
        grip_hz = r.gripper.mean_hz if r.gripper else 0
        press_hz = r.pressure.mean_hz if r.pressure else 0
        issues = len(r.issues)
        warns = len(r.warnings)

        # 颜色标记
        cam_ok = "✅" if cam_hz >= 27 else "⚠️" if cam_hz >= 20 else "❌"
        rob_ok = "✅" if rob_hz >= 108 else "⚠️"
        grip_ok = "✅" if grip_hz >= 108 else "⚠️"
        press_ok = "✅" if press_hz >= 180 else "⚠️" if press_hz >= 120 else "❌"

        name = r.session_name[-28:] if len(r.session_name) > 28 else r.session_name
        print(f"  {name:<28} {frames:>6} {cam_hz:>5.1f}{cam_ok} {rob_hz:>6.1f}{rob_ok} "
              f"{grip_hz:>6.1f}{grip_ok} {press_hz:>6.1f}{press_ok} {issues:>5}❌ {warns:>3}⚠️")

    # 统计
    all_cam = [r.frames.mean_hz for r in reports if r.frames and r.frames.mean_hz > 0]
    all_rob = [r.robot.mean_hz for r in reports if r.robot and r.robot.mean_hz > 0]
    all_grip = [r.gripper.mean_hz for r in reports if r.gripper and r.gripper.mean_hz > 0]
    all_press = [r.pressure.mean_hz for r in reports if r.pressure and r.pressure.mean_hz > 0]

    print(f"\n  📊 统计")
    if all_cam:
        print(f"    相机:    mean={statistics.mean(all_cam):.1f}  min={min(all_cam):.1f}  max={max(all_cam):.1f} Hz")
    if all_rob:
        print(f"    robot:   mean={statistics.mean(all_rob):.1f}  min={min(all_rob):.1f}  max={max(all_rob):.1f} Hz")
    if all_grip:
        print(f"    gripper: mean={statistics.mean(all_grip):.1f}  min={min(all_grip):.1f}  max={max(all_grip):.1f} Hz")
    if all_press:
        print(f"    pressure:mean={statistics.mean(all_press):.1f}  min={min(all_press):.1f}  max={max(all_press):.1f} Hz")

    total_issues = sum(len(r.issues) for r in reports)
    total_warns = sum(len(r.warnings) for r in reports)
    print(f"\n  总计: {total_issues} 个问题, {total_warns} 个警告")
    print(f"{'═' * W}")


def export_csv(reports: list[SessionReport], path: Path) -> None:
    """导出汇总报告为 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session", "duration_s", "frames", "cam_hz",
            "robot_rows", "robot_hz", "robot_gap_count", "robot_max_gap_ms", "robot_near_dup",
            "gripper_rows", "gripper_hz",
            "pressure_rows", "pressure_hz",
            "world_rgb_count", "world_depth_count", "wrist_rgb_count", "wrist_depth_count",
            "robot_offset_p95_ms", "gripper_offset_p95_ms", "pressure_offset_p95_ms",
            "issues", "warnings",
        ])
        for r in reports:
            writer.writerow([
                r.session_name,
                r.frames.duration_s if r.frames else 0,
                r.frames.row_count if r.frames else 0,
                f"{r.frames.mean_hz:.1f}" if r.frames else 0,
                r.robot.row_count if r.robot else 0,
                f"{r.robot.mean_hz:.1f}" if r.robot else 0,
                r.robot.gap_count if r.robot else 0,
                f"{r.robot.max_gap_ms:.1f}" if r.robot else 0,
                r.robot.near_dup_count if r.robot else 0,
                r.gripper.row_count if r.gripper else 0,
                f"{r.gripper.mean_hz:.1f}" if r.gripper else 0,
                r.pressure.row_count if r.pressure else 0,
                f"{r.pressure.mean_hz:.1f}" if r.pressure else 0,
                r.world_rgb.count if r.world_rgb else 0,
                r.world_depth.count if r.world_depth else 0,
                r.wrist_rgb.count if r.wrist_rgb else 0,
                r.wrist_depth.count if r.wrist_depth else 0,
                f"{r.robot_offset_p95_ms:.2f}",
                f"{r.gripper_offset_p95_ms:.2f}",
                f"{r.pressure_offset_p95_ms:.2f}",
                "; ".join(r.issues),
                "; ".join(r.warnings),
            ])
    print(f"\n  ✅ 已导出: {path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="全面数据质量检查工具")
    parser.add_argument("--session", "-s", type=str, default=None, help="指定 session 路径")
    parser.add_argument("--latest", "-n", type=int, default=None, help="只检查最新 N 个 session")
    parser.add_argument("--export", "-e", type=str, default=None, help="导出 CSV 报告路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        print("❌ 未找到 sessions 目录")
        return

    # 确定要检查的 session
    if args.session:
        session_paths = [Path(args.session)]
    else:
        session_paths = sorted(
            [d for d in sessions_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if args.latest:
            session_paths = session_paths[:args.latest]

    if not session_paths:
        print("❌ 未找到任何 session")
        return

    print(f"\n🔍 开始检查 {len(session_paths)} 个 session...")

    reports = []
    for i, sp in enumerate(session_paths, 1):
        print(f"\n  [{i}/{len(session_paths)}] {sp.name}...", end="", flush=True)
        report = check_session(sp)
        reports.append(report)
        issue_count = len(report.issues)
        warn_count = len(report.warnings)
        if issue_count > 0:
            print(f" ❌ {issue_count} 问题, {warn_count} 警告")
        elif warn_count > 0:
            print(f" ⚠️ {warn_count} 警告")
        else:
            print(f" ✅")

    # 打印详细报告
    for report in reports:
        print_report(report, args.verbose)

    # 打印汇总
    print_summary(reports)

    # 导出
    if args.export:
        export_csv(reports, Path(args.export))


if __name__ == "__main__":
    main()

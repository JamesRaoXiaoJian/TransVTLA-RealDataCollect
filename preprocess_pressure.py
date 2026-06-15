"""柔性触觉传感器数据预处理脚本。

将原始 ADC 读数转化为 VLA 模型可直接使用的归一化张量数据。

处理流程：
    1. 读取所有 pressure.csv，提取 20 个有效通道
    2. 动态基线消除（取前 50 行均值作为基线）
    3. 全局 Min-Max 归一化到 [0, 1]
    4. 滑动窗口切片为 [Samples, window_size, 20] 张量
    5. 保存为 .npz 文件，并保留窗口时间戳用于同步

用法：
    python preprocess_pressure.py --data-root sessions --output processed_data.npz
    python preprocess_pressure.py --test  # 运行内置测试
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 通道定义（从 Channel Mapping.txt 统一读取）
# 最终输出维度顺序：左总压(1) + 右总压(1) + 左矩阵(9) + 右矩阵(9) = 20
# ============================================================
from channel_config import (
    LEFT_CHANNEL, RIGHT_CHANNEL,
    LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
    VALID_CHANNELS, INTERPOLATE_CHANNELS,
)

LEFT_TOTAL_CH = LEFT_CHANNEL
RIGHT_TOTAL_CH = RIGHT_CHANNEL

VALID_COL_NAMES: list[str] = [f"CH{ch}" for ch in VALID_CHANNELS]

# ============================================================
# 处理参数
# ============================================================
BASELINE_ROWS = 50           # 基线计算行数（前 0.25 秒 @ 200Hz）
MAX_PRESSURE_DROP = 3500.0   # 全局最大压力差（ADC 单位）
WINDOW_SIZE = 16             # 滑动窗口大小
STRIDE = 1                   # 滑动窗口步长


def load_pressure_csv(csv_path: Path) -> pd.DataFrame:
    """读取单个 pressure.csv 文件，返回标准 20 通道 DataFrame。

    兼容旧格式（timestamp_us + CH1..CH64）和新格式
    （sensor_timestamp_us + host_monotonic_us + 标准 20 通道）。
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in VALID_COL_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，文件: {csv_path}")
    return df[VALID_COL_NAMES]


def load_pressure_csv_with_timestamps(csv_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """读取压力 CSV，返回标准 20 通道和可用时间戳数组。"""
    raw = pd.read_csv(csv_path)
    missing = [c for c in VALID_COL_NAMES if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，文件: {csv_path}")

    timestamps: dict[str, np.ndarray] = {}
    for col in ("sensor_timestamp_us", "host_monotonic_us", "timestamp_us"):
        if col in raw.columns:
            timestamps[col] = raw[col].to_numpy(dtype=np.int64, copy=True)
    if "sensor_timestamp_us" not in timestamps and "timestamp_us" in timestamps:
        timestamps["sensor_timestamp_us"] = timestamps["timestamp_us"].copy()
    return raw[VALID_COL_NAMES], timestamps


def extract_valid_channels(df: pd.DataFrame) -> np.ndarray:
    """从 64 通道 DataFrame 中提取 20 个有效通道，返回 (N, 20) 数组。

    输入可以是完整 CH1..CH64，也可以已经是标准 20 通道。

    对于 INTERPOLATE_CHANNELS 中标记的异常通道，使用相邻通道均值替代。
    """
    missing = [c for c in VALID_COL_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少有效通道列: {missing}")
    data = df[VALID_COL_NAMES].values.astype(np.float64)

    # 处理需要插值的异常通道
    if INTERPOLATE_CHANNELS:
        for bad_ch, adjacent_chs in INTERPOLATE_CHANNELS.items():
            if bad_ch not in VALID_CHANNELS:
                continue
            bad_idx = VALID_CHANNELS.index(bad_ch)
            # 找到相邻通道在 VALID_CHANNELS 中的索引
            adj_indices = []
            for adj_ch in adjacent_chs:
                if adj_ch in VALID_CHANNELS:
                    adj_indices.append(VALID_CHANNELS.index(adj_ch))
            if adj_indices:
                # 用相邻通道均值替代
                data[:, bad_idx] = data[:, adj_indices].mean(axis=1)

    return data


def baseline_subtract(data: np.ndarray, baseline_rows: int = BASELINE_ROWS) -> np.ndarray:
    """动态基线消除。

    对每个通道取前 baseline_rows 行的均值作为基线 B_0，
    计算 Delta_P = B_0 - P_raw，并 clip 到 [0, +inf)。

    Args:
        data: (N, 20) 原始 ADC 读数
        baseline_rows: 用于计算基线的行数

    Returns:
        (N, 20) 去基线后的压力差值，所有值 >= 0
    """
    n = data.shape[0]
    if n < baseline_rows:
        # 数据不足 baseline_rows 行时，用全部数据计算基线
        baseline_rows = n

    # B_0: (20,) 每个通道的基线均值
    baseline = data[:baseline_rows].mean(axis=0)

    # Delta_P = B_0 - P_raw（ADC 值越大代表压力越小，所以取反）
    delta = baseline - data

    # clip 负值为 0（消除底噪波动导致的微小负数）
    return np.maximum(0.0, delta)


def normalize(delta: np.ndarray, max_drop: float = MAX_PRESSURE_DROP) -> np.ndarray:
    """全局 Min-Max 归一化到 [0, 1]。

    P_norm = Delta_P / MAX_PRESSURE_DROP，再 clip 到 [0, 1]。

    Args:
        delta: (N, 20) 去基线后的压力差值
        max_drop: 最大压力差常量

    Returns:
        (N, 20) 归一化后的压力值，范围 [0.0, 1.0]
    """
    normalized = delta / max_drop
    return np.clip(normalized, 0.0, 1.0)


def sliding_window(data: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    """滑动窗口切片，将 (N, 20) 切为 (Samples, window_size, 20)。

    Args:
        data: (N, 20) 归一化后的时序数据
        window_size: 窗口大小
        stride: 步长

    Returns:
        (Samples, window_size, 20) 的三维数组
    """
    n = data.shape[0]
    if n < window_size:
        return np.empty((0, window_size, data.shape[1]))

    # 计算切片数量
    n_samples = (n - window_size) // stride + 1

    # 利用 stride_tricks 零拷贝创建滑动窗口视图
    shape = (n_samples, window_size, data.shape[1])
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()


def sliding_window_timestamps(
    timestamps: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return start/end/center timestamps for each sliding window."""
    n = len(timestamps)
    if n < window_size:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    starts = np.arange(0, n - window_size + 1, stride, dtype=np.int64)
    start_ts = timestamps[starts]
    end_ts = timestamps[starts + window_size - 1]
    center_ts = ((start_ts.astype(np.int64) + end_ts.astype(np.int64)) // 2).astype(np.int64)
    return start_ts, end_ts, center_ts


def process_single_csv(csv_path: Path) -> np.ndarray:
    """处理单个 CSV 文件，返回滑动窗口切片后的张量。

    Pipeline: 读取 -> 提取通道 -> 基线消除 -> 归一化 -> 滑动窗口

    Returns:
        (Samples, WINDOW_SIZE, 20) 的 numpy 数组
    """
    windows, _timestamps = process_single_csv_with_timestamps(csv_path)
    return windows


def process_single_csv_with_timestamps(csv_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """处理单个 CSV，并返回窗口张量与同步时间戳。"""
    df, timestamps = load_pressure_csv_with_timestamps(csv_path)
    data = extract_valid_channels(df)
    delta = baseline_subtract(data)
    normed = normalize(delta)
    windows = sliding_window(normed)

    output_ts: dict[str, np.ndarray] = {}
    for key, values in timestamps.items():
        output_ts[key] = values
    host_ts = timestamps.get("host_monotonic_us")
    if host_ts is not None:
        start_ts, end_ts, center_ts = sliding_window_timestamps(host_ts)
        output_ts["window_start_host_us"] = start_ts
        output_ts["window_end_host_us"] = end_ts
        output_ts["window_center_host_us"] = center_ts
    sensor_ts = timestamps.get("sensor_timestamp_us")
    if sensor_ts is not None:
        _start_ts, _end_ts, center_ts = sliding_window_timestamps(sensor_ts)
        output_ts["window_center_sensor_us"] = center_ts
    return windows, output_ts


def process_all_sessions(data_root: Path, output_path: Path | None = None) -> None:
    """遍历 data_root 下所有 pressure.csv，按 session 保存，并可选合并保存。

    同时将每个 session 的结果单独保存到 session 目录下的
    preprocessed_pressure/{session_name}.npz，供 RLDS 构建脚本使用。

    目录结构约定：
        data_root/**/pressure/pressure.csv
    输出到：
        data_root/**/preprocessed_pressure/session_name.npz
        output_path（可选）合并所有 session 后的 .npz
    """
    csv_files = sorted(data_root.rglob("pressure/pressure.csv"))

    if not csv_files:
        print(f"未找到 pressure.csv 文件，请检查目录: {data_root}")
        return

    print(f"找到 {len(csv_files)} 个 pressure.csv 文件")

    all_windows: list[np.ndarray] = []
    session_info: list[dict] = []
    total_samples = 0

    for csv_path in csv_files:
        session_dir = csv_path.parent.parent
        session_name = session_dir.name
        try:
            windows, timestamps = process_single_csv_with_timestamps(csv_path)
        except Exception as e:
            print(f"  跳过 {session_name}: {e}")
            continue

        if windows.shape[0] == 0:
            print(f"  跳过 {session_name}: 数据不足 {WINDOW_SIZE} 行")
            continue

        all_windows.append(windows)
        session_info.append({
            "name": session_name,
            "csv": str(csv_path),
            "samples": windows.shape[0],
        })

        # 保存到 session 同级的 preprocessed_pressure 目录
        out_dir = csv_path.parent.parent / "preprocessed_pressure"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session_name}.npz"
        np.savez_compressed(
            out_path,
            data=windows,
            channels=np.array(VALID_CHANNELS),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            max_pressure_drop=MAX_PRESSURE_DROP,
            baseline_rows=BASELINE_ROWS,
            **timestamps,
        )
        total_samples += windows.shape[0]
        print(f"  {session_name}: {windows.shape[0]} 个样本 -> {out_path}")

    print(f"\n共处理 {total_samples} 个样本")

    if output_path is None:
        return

    if not all_windows:
        print("无有效数据可合并")
        return

    merged = np.concatenate(all_windows, axis=0)
    print(f"合并后总样本数: {merged.shape[0]}")
    print(f"张量形状: {merged.shape}  (Samples, {WINDOW_SIZE}, 20)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=merged,
        channels=np.array(VALID_CHANNELS),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_pressure_drop=MAX_PRESSURE_DROP,
        baseline_rows=BASELINE_ROWS,
        session_info=np.array(session_info, dtype=object),
    )
    print(f"已保存合并数据: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


# ============================================================
# 内置测试：生成 dummy data 验证处理逻辑
# ============================================================
def run_test() -> None:
    """生成模拟数据，验证完整 pipeline 的正确性。"""
    print("=" * 60)
    print("运行内置测试")
    print("=" * 60)

    np.random.seed(42)
    n_rows = 500
    n_channels = 64

    # 1. 生成模拟 ADC 数据
    #    - 基线值在 4700~5007 之间（模拟未接触状态）
    #    - 在第 100~200 行模拟按压（ADC 值下降到 2000~3000）
    raw = np.random.randint(4700, 5007, size=(n_rows, n_channels)).astype(np.float64)

    # 在有效通道上模拟按压
    for ch in VALID_CHANNELS:
        col_idx = ch - 1
        raw[100:200, col_idx] = np.random.randint(1500, 3000, size=100).astype(np.float64)
        raw[300:350, col_idx] = np.random.randint(2000, 3500, size=50).astype(np.float64)

    # 转为 DataFrame 模拟 CSV 格式
    ch_cols = [f"CH{i}" for i in range(1, 65)]
    df = pd.DataFrame(raw, columns=ch_cols)

    print(f"\n模拟数据: {n_rows} 行 x {n_channels} 通道")
    print(f"有效通道数: {len(VALID_CHANNELS)}")

    # 2. 提取有效通道
    data = extract_valid_channels(df)
    assert data.shape == (n_rows, 20), f"形状错误: {data.shape}"
    print(f"提取通道后: {data.shape}")

    # 3. 基线消除
    delta = baseline_subtract(data)
    assert delta.shape == (n_rows, 20)
    assert np.all(delta >= 0), "基线消除后存在负值！"
    print(f"基线消除后: min={delta.min():.1f}, max={delta.max():.1f}, mean={delta.mean():.1f}")

    # 验证：前 50 行（基线段）delta 应接近 0
    baseline_delta_mean = delta[:BASELINE_ROWS].mean()
    assert baseline_delta_mean < 50, f"基线段均值过大: {baseline_delta_mean}"
    print(f"基线段均值: {baseline_delta_mean:.2f} (应接近 0)")

    # 验证：按压段 delta 应显著大于 0
    press_delta_mean = delta[100:200].mean()
    assert press_delta_mean > 500, f"按压段均值过小: {press_delta_mean}"
    print(f"按压段均值: {press_delta_mean:.1f} (应明显 > 0)")

    # 4. 归一化
    normed = normalize(delta)
    assert normed.shape == (n_rows, 20)
    assert np.all(normed >= 0.0) and np.all(normed <= 1.0), "归一化后超出 [0, 1]！"
    print(f"归一化后: min={normed.min():.4f}, max={normed.max():.4f}")

    # 5. 滑动窗口
    windows = sliding_window(normed)
    expected_samples = (n_rows - WINDOW_SIZE) // STRIDE + 1
    assert windows.shape == (expected_samples, WINDOW_SIZE, 20), f"窗口形状错误: {windows.shape}"
    print(f"滑动窗口后: {windows.shape}  (Samples, {WINDOW_SIZE}, 20)")

    # 验证窗口连续性
    np.testing.assert_array_equal(windows[0], normed[0:WINDOW_SIZE])
    np.testing.assert_array_equal(windows[1], normed[1:WINDOW_SIZE + 1])
    print("窗口连续性验证通过")

    # 6. 测试数据不足的情况
    short_data = np.random.rand(5, 20)
    short_windows = sliding_window(short_data)
    assert short_windows.shape[0] == 0, "短数据应返回空数组"
    print("短数据处理验证通过")

    # 7. 端到端：process_single_csv 需要文件，这里用临时方式验证
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="柔性触觉传感器数据预处理")
    parser.add_argument("--data-root", type=Path, default=Path("sessions"),
                        help="数据根目录（包含 pressure/pressure.csv 的父目录）")
    parser.add_argument("--output", type=Path, default=None,
                        help="可选：合并所有 session 后的输出 .npz 文件路径")
    parser.add_argument("--test", action="store_true",
                        help="运行内置测试验证处理逻辑")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        process_all_sessions(args.data_root, args.output)


if __name__ == "__main__":
    main()

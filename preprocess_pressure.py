"""柔性触觉传感器数据预处理脚本。

将原始 ADC 读数转化为 VLA 模型可直接使用的归一化张量数据。

处理流程：
    1. 读取所有 pressure.csv，提取 20 个有效通道
    2. 动态基线消除（取前 50 行均值作为基线）
    3. 全局 Min-Max 归一化到 [0, 1]
    4. 滑动窗口切片为 [Samples, window_size, 20] 张量
    5. 保存为 .npy 文件

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
# 通道定义（1-indexed，与硬件协议一致）
# 最终输出维度顺序：左总压(1) + 右总压(1) + 左矩阵(9) + 右矩阵(9) = 20
# ============================================================
LEFT_TOTAL_CH = 19
RIGHT_TOTAL_CH = 18

LEFT_MATRIX_CHANNELS: list[list[int]] = [
    [1, 16, 15],
    [14, 13, 12],
    [11, 10, 9],
]
RIGHT_MATRIX_CHANNELS: list[list[int]] = [
    [17, 32, 31],
    [30, 29, 28],
    [27, 26, 25],
]

# 按固定顺序排列的 20 个有效通道（1-indexed）
VALID_CHANNELS: list[int] = (
    [LEFT_TOTAL_CH, RIGHT_TOTAL_CH]
    + [ch for row in LEFT_MATRIX_CHANNELS for ch in row]
    + [ch for row in RIGHT_MATRIX_CHANNELS for ch in row]
)

# CSV 中的列名格式为 CH1, CH2, ..., CH64
# pandas 读入后列索引从 0 开始，通道号需减 1 映射到列索引
VALID_COL_INDICES: list[int] = [ch - 1 for ch in VALID_CHANNELS]

# ============================================================
# 处理参数
# ============================================================
BASELINE_ROWS = 50           # 基线计算行数（前 0.25 秒 @ 200Hz）
MAX_PRESSURE_DROP = 3500.0   # 全局最大压力差（ADC 单位）
WINDOW_SIZE = 16             # 滑动窗口大小
STRIDE = 1                   # 滑动窗口步长


def load_pressure_csv(csv_path: Path) -> pd.DataFrame:
    """读取单个 pressure.csv 文件，返回 DataFrame。

    CSV 格式：timestamp_us, CH1, CH2, ..., CH64
    第一列为时间戳，后续 64 列为 ADC 通道值。
    """
    df = pd.read_csv(csv_path)
    # 丢弃 timestamp_us 列，只保留通道数据
    ch_cols = [f"CH{i}" for i in range(1, 65)]
    missing = [c for c in ch_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，文件: {csv_path}")
    return df[ch_cols]


def extract_valid_channels(df: pd.DataFrame) -> np.ndarray:
    """从 64 通道 DataFrame 中提取 20 个有效通道，返回 (N, 20) 数组。

    列索引对应关系：
        CH1 -> col 0, CH2 -> col 1, ..., CH64 -> col 63
    有效通道按 VALID_COL_INDICES 选取。
    """
    return df.values[:, VALID_COL_INDICES].astype(np.float64)


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


def process_single_csv(csv_path: Path) -> np.ndarray:
    """处理单个 CSV 文件，返回滑动窗口切片后的张量。

    Pipeline: 读取 -> 提取通道 -> 基线消除 -> 归一化 -> 滑动窗口

    Returns:
        (Samples, WINDOW_SIZE, 20) 的 numpy 数组
    """
    df = load_pressure_csv(csv_path)
    data = extract_valid_channels(df)
    delta = baseline_subtract(data)
    normed = normalize(delta)
    windows = sliding_window(normed)
    return windows


def process_all_sessions(data_root: Path, output_path: Path) -> None:
    """遍历 data_root 下所有 pressure.csv，合并处理后保存。

    同时将每个 session 的结果单独保存到 session 目录下的
    preprocessed_pressure/{session_name}.npz，供 RLDS 构建脚本使用。

    目录结构约定：
        data_root/**/pressure/pressure.csv
    """
    csv_files = sorted(data_root.rglob("pressure/pressure.csv"))

    if not csv_files:
        print(f"未找到 pressure.csv 文件，请检查目录: {data_root}")
        return

    print(f"找到 {len(csv_files)} 个 pressure.csv 文件")

    all_windows: list[np.ndarray] = []
    session_info: list[dict] = []

    for csv_path in csv_files:
        session_dir = csv_path.parent.parent
        session_name = session_dir.name
        try:
            windows = process_single_csv(csv_path)
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
        print(f"  {session_name}: {windows.shape[0]} 个样本")

        # 按 session 单独保存到 preprocessed_pressure/{session_name}.npz
        per_session_dir = session_dir / "preprocessed_pressure"
        per_session_dir.mkdir(parents=True, exist_ok=True)
        per_session_path = per_session_dir / f"{session_name}.npz"
        np.savez_compressed(
            per_session_path,
            data=windows,
            channels=np.array(VALID_CHANNELS),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            max_pressure_drop=MAX_PRESSURE_DROP,
            baseline_rows=BASELINE_ROWS,
        )
        print(f"    -> 已保存: {per_session_path}")

    if not all_windows:
        print("无有效数据可处理")
        return

    # 合并所有 session 的样本
    merged = np.concatenate(all_windows, axis=0)
    print(f"\n合并后总样本数: {merged.shape[0]}")
    print(f"张量形状: {merged.shape}  (Samples, {WINDOW_SIZE}, 20)")

    # 保存为 .npz（包含数据和元信息）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=merged,
        channels=np.array(VALID_CHANNELS),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        max_pressure_drop=MAX_PRESSURE_DROP,
        baseline_rows=BASELINE_ROWS,
    )
    print(f"已保存: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


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
    for col_idx in VALID_COL_INDICES:
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
    parser.add_argument("--output", type=Path, default=Path("processed_data.npz"),
                        help="输出文件路径（.npz 格式）")
    parser.add_argument("--test", action="store_true",
                        help="运行内置测试验证处理逻辑")
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        process_all_sessions(args.data_root, args.output)


if __name__ == "__main__":
    main()

"""触觉传感器通道映射配置模块。

从 channel_mapping.json 读取通道映射，供所有相关脚本统一使用。

使用方式:
    from channel_config import (
        LEFT_CHANNEL, RIGHT_CHANNEL,
        LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
        VALID_CHANNELS, VALID_CHANNEL_INDEX,
        PRESSURE_VALUE_COLUMNS, STANDARD_PRESSURE_COLUMNS,
        INTERPOLATE_CHANNELS,
        channel_value, interpolate_pressure_values,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# 默认映射文件路径（与本文件同目录）
_MAPPING_FILE = Path(__file__).parent / "channel_mapping.json"


def load_channel_mapping(
    mapping_file: Path | str | None = None,
) -> Tuple[int, int, List[List[int]], List[List[int]], Dict[int, List[int]]]:
    """从 JSON 映射文件读取通道配置。

    Args:
        mapping_file: 映射文件路径，为 None 时使用默认的 channel_mapping.json

    Returns:
        (left_channel, right_channel, left_matrix, right_matrix, interpolate_channels)
    """
    if mapping_file is None:
        mapping_file = _MAPPING_FILE
    mapping_file = Path(mapping_file)

    if not mapping_file.exists():
        raise FileNotFoundError(f"通道映射文件不存在: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 插值通道配置: {异常通道: [相邻通道列表]}
    interp_raw = cfg.get("INTERPOLATE_CHANNELS", {})
    interp = {int(k): list(map(int, v)) for k, v in interp_raw.items()}

    return (
        int(cfg["LEFT_CHANNEL"]),
        int(cfg["RIGHT_CHANNEL"]),
        [list(map(int, row)) for row in cfg["LEFT_MATRIX_CHANNELS"]],
        [list(map(int, row)) for row in cfg["RIGHT_MATRIX_CHANNELS"]],
        interp,
    )


# ---------------------------------------------------------------------------
# 模块级常量：导入时自动加载
# ---------------------------------------------------------------------------
(
    LEFT_CHANNEL,
    RIGHT_CHANNEL,
    LEFT_MATRIX_CHANNELS,
    RIGHT_MATRIX_CHANNELS,
    INTERPOLATE_CHANNELS,
) = load_channel_mapping()

# 展平为一维列表（20 个有效通道，CSV 与模型均使用此顺序）
VALID_CHANNELS: List[int] = (
    [LEFT_CHANNEL, RIGHT_CHANNEL]
    + [ch for row in LEFT_MATRIX_CHANNELS for ch in row]
    + [ch for row in RIGHT_MATRIX_CHANNELS for ch in row]
)

# 标准压力 CSV：两个时间戳列 + 20 个有效通道列。
PRESSURE_TIMESTAMP_COLUMNS: List[str] = ["sensor_timestamp_us", "host_monotonic_us"]
PRESSURE_VALUE_COLUMNS: List[str] = [f"CH{ch}" for ch in VALID_CHANNELS]
STANDARD_PRESSURE_COLUMNS: List[str] = PRESSURE_TIMESTAMP_COLUMNS + PRESSURE_VALUE_COLUMNS

# 20 通道数组中的索引映射。
VALID_CHANNEL_INDEX: Dict[int, int] = {ch: idx for idx, ch in enumerate(VALID_CHANNELS)}

# 兼容旧代码命名：在标准 20 通道数组中，索引不再等于物理通道号。
VALID_COL_INDICES: List[int] = list(VALID_CHANNELS)


def channel_value(values: List[int], channel: int, default: int = 0) -> int:
    """按物理通道号读取标准 20 通道数组中的值。"""
    idx = VALID_CHANNEL_INDEX.get(channel)
    if idx is None or idx >= len(values):
        return default
    return values[idx]


def interpolate_pressure_values(values: List[int]) -> List[int]:
    """对标准 20 通道数组应用配置的异常通道插值。"""
    out = list(values)
    for bad_ch, adjacent_chs in INTERPOLATE_CHANNELS.items():
        bad_idx = VALID_CHANNEL_INDEX.get(bad_ch)
        if bad_idx is None or bad_idx >= len(out):
            continue
        adjacent_values = [
            out[idx]
            for ch in adjacent_chs
            if (idx := VALID_CHANNEL_INDEX.get(ch)) is not None and idx < len(out)
        ]
        if adjacent_values:
            out[bad_idx] = int(sum(adjacent_values) / len(adjacent_values))
    return out

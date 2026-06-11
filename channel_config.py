"""触觉传感器通道映射配置模块。

从 channel_mapping.json 读取通道映射，供所有相关脚本统一使用。

使用方式:
    from channel_config import (
        LEFT_CHANNEL, RIGHT_CHANNEL,
        LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS,
        VALID_CHANNELS, VALID_COL_INDICES,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

# 默认映射文件路径（与本文件同目录）
_MAPPING_FILE = Path(__file__).parent / "channel_mapping.json"


def load_channel_mapping(
    mapping_file: Path | str | None = None,
) -> Tuple[int, int, List[List[int]], List[List[int]]]:
    """从 JSON 映射文件读取通道配置。

    Args:
        mapping_file: 映射文件路径，为 None 时使用默认的 channel_mapping.json

    Returns:
        (left_channel, right_channel, left_matrix, right_matrix)
    """
    if mapping_file is None:
        mapping_file = _MAPPING_FILE
    mapping_file = Path(mapping_file)

    if not mapping_file.exists():
        raise FileNotFoundError(f"通道映射文件不存在: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return (
        int(cfg["LEFT_CHANNEL"]),
        int(cfg["RIGHT_CHANNEL"]),
        [list(map(int, row)) for row in cfg["LEFT_MATRIX_CHANNELS"]],
        [list(map(int, row)) for row in cfg["RIGHT_MATRIX_CHANNELS"]],
    )


# ---------------------------------------------------------------------------
# 模块级常量：导入时自动加载
# ---------------------------------------------------------------------------
LEFT_CHANNEL, RIGHT_CHANNEL, LEFT_MATRIX_CHANNELS, RIGHT_MATRIX_CHANNELS = (
    load_channel_mapping()
)

# 展平为一维列表（20 个有效通道）
VALID_CHANNELS: List[int] = (
    [LEFT_CHANNEL, RIGHT_CHANNEL]
    + [ch for row in LEFT_MATRIX_CHANNELS for ch in row]
    + [ch for row in RIGHT_MATRIX_CHANNELS for ch in row]
)

# CSV 列索引（CSV 第一列是 timestamp，通道数据从第 2 列开始，所以索引 = 通道号）
VALID_COL_INDICES: List[int] = list(VALID_CHANNELS)

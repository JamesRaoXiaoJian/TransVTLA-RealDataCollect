"""高精度单调时间戳工具模块。

保证单调递增的时间戳生成器，用于所有采集器统一时间基准。

使用方式:
    from timestamp_utils import MonotonicTimestamp, get_timestamp_us

    ts = MonotonicTimestamp()
    us = ts.get_us()  # 微秒精度，单调递增

    # 或直接使用便捷函数
    us = get_timestamp_us()
"""

from __future__ import annotations

import time


class MonotonicTimestamp:
    """保证单调递增的时间戳生成器。

    基于 time.monotonic_ns()，不受系统时钟调整（NTP 等）影响。
    输出微秒精度的 Unix 时间戳。
    """

    def __init__(self):
        self._last: int = 0
        # 计算 monotonic 和 wall clock 之间的偏移
        self._offset: int = time.time_ns() - time.monotonic_ns()

    def get_us(self) -> int:
        """获取当前时间戳（微秒），保证单调递增。"""
        ts = int((time.monotonic_ns() + self._offset) / 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts


# 全局实例
_global_ts = MonotonicTimestamp()


def get_timestamp_us() -> int:
    """便捷函数：获取全局单调递增时间戳（微秒）。"""
    return _global_ts.get_us()

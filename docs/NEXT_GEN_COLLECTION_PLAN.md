# 下一代采集系统优化方案

> 基于调研和现有代码分析，针对网线直连机械臂、视觉帧率提升、编码优化、时间戳同步的综合方案。

---

## 1. 现状问题总结

| 模态 | 目标 | 实际 | 瓶颈 |
|------|------|------|------|
| DJI 相机 | 20 fps | **~8 fps** | YUYV 格式饱和 USB 带宽 |
| RealSense | 20 fps | **~8 fps** | 同步阻塞 + 深度滤波 + 1280x720 |
| 机械臂 | 200 Hz | **~16 Hz** | TCP 网络延迟 ~60ms |
| 夹爪 | 200 Hz | **~19 Hz** | TCP 网络延迟 + 时间戳重复 |
| 压力 | 200 Hz | **~139 Hz** | 硬件发送频率限制 |
| 图像时间戳 | 有 | **无** | 只有序列号，无采集时间 |

---

## 2. DJI 相机优化

### 2.1 核心问题

当前 `collectors/dji_camera.py` 第 30 行：
```python
cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)  # ❌ Windows 专用后端
```

- `CAP_DSHOW` (DirectShow) 是 Windows 专用，Linux 下回退到默认后端
- 未设置像素格式，默认请求 YUYV（未压缩），1280x720@30fps 需要 ~180 MB/s，超过 USB 2.0 带宽上限

### 2.2 优化方案

```python
# collectors/dji_camera.py - start() 方法

def start(self) -> None:
    # 1. 使用 V4L2 后端 (Linux)
    cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

    if not cap.isOpened():
        # 回退到默认后端
        cap = cv2.VideoCapture(self.index)

    # 2. 设置 MJPEG 格式 (关键！减少 USB 带宽 ~10x)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 3. 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 4. 最小化缓冲区
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ... 后续不变
```

### 2.3 带宽对比

| 格式 | 1280x720@30fps 带宽 | USB 2.0 (480Mbps) 可行 |
|------|---------------------|------------------------|
| YUYV | ~180 MB/s = 1440 Mbps | ❌ 超限 |
| MJPEG | ~15-25 MB/s = 120-200 Mbps | ✅ 可行 |
| NV12 | ~60 MB/s = 480 Mbps | ⚠️ 临界 |

### 2.4 V4L2 控制项（采集一致性）

```bash
# 锁定曝光（避免亮度波动影响训练）
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1  # 1=手动
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=100

# 锁定白平衡
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4600

# 查看可用控制项
v4l2-ctl -d /dev/video0 --list-ctrls
```

### 2.5 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 帧率 | ~8 fps | **25-30 fps** |
| USB 带宽 | ~180 MB/s | ~20 MB/s |
| 帧间延迟 | ~125 ms | ~33 ms |

---

## 3. RealSense 相机优化

### 3.1 核心问题

当前 `collectors/realsense_rgb.py` 存在 5 个问题：

1. **同步阻塞**: `read()` 调用 `_grab_frames()` 阻塞 Qt 事件循环
2. **深度滤波**: 每帧运行 3 个滤波器（spatial + temporal + hole_filling）
3. **分辨率过高**: 1280x720 双流（color + depth）接近 USB 带宽上限
4. **无后台线程**: 不像 DJI 有独立采集线程
5. **自动曝光**: 导致帧间亮度不一致

### 3.2 优化方案

```python
# collectors/realsense_rgb.py - 完整重构

import threading
import time
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealSenseRGB:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_depth: bool = True, enable_filters: bool = False):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.enable_filters = enable_filters

        self.pipeline = None
        self.align = None
        self.available = False

        self._last_color: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if rs is None:
            self.available = False
            print("Warning: pyrealsense2 not installed.")
            return

        pipeline = rs.pipeline()
        config = rs.config()

        # 配置 color 流
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)

        # 配置 depth 流（可选）
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height,
                                 rs.format.z16, self.fps)

        try:
            pipeline_profile = pipeline.start(config)
        except RuntimeError as exc:
            self.available = False
            print(f"Warning: RealSense pipeline failed: {exc}")
            return

        # 调整传感器设置
        device = pipeline_profile.get_device()

        # 关闭自动曝光（训练数据一致性）
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        color_sensor = device.first_color_sensor()
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        # 手动曝光（根据环境光调整）
        if color_sensor.supports(rs.option.exposure):
            color_sensor.set_option(rs.option.exposure, 8.0)  # 8ms

        self.pipeline = pipeline
        if self.enable_depth:
            self.align = rs.align(rs.stream.color)

        # 启动后台采集线程
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.available = True

    def _capture_loop(self) -> None:
        """后台线程：持续采集帧，缓存最新帧。"""
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                if self.align:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_data = np.asanyarray(color_frame.get_data())

                depth_data = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        # 深度滤波可选（默认关闭，离线处理）
                        if self.enable_filters:
                            depth_frame = rs.spatial_filter().process(depth_frame)
                            depth_frame = rs.temporal_filter().process(depth_frame)
                            depth_frame = rs.hole_filling_filter().process(depth_frame)
                        depth_data = np.asanyarray(depth_frame.get_data())

                with self._lock:
                    self._last_color = color_data
                    self._last_depth = depth_data

            except Exception as e:
                if self._running:
                    time.sleep(0.01)

    def read(self) -> np.ndarray:
        """非阻塞读取最新帧。"""
        with self._lock:
            if self._last_color is not None:
                return self._last_color.copy()
        return self._zero_frame()

    def read_depth(self) -> np.ndarray:
        """非阻塞读取最新深度帧。"""
        with self._lock:
            if self._last_depth is not None:
                return self._last_depth.copy()
        return self._zero_depth()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self.available = False

    def _zero_frame(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _zero_depth(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint16)
```

### 3.3 分辨率选择

| 分辨率 | Color Max FPS | Depth Max FPS | USB 带宽 | 推荐 |
|--------|---------------|---------------|----------|------|
| 1920x1080 | 30 | 30 | 高 | ❌ 过高 |
| 1280x720 | 30 | 60 | 中高 | ⚠️ 当前 |
| **848x480** | **60** | **60** | **中** | ✅ 推荐 |
| 640x480 | 60 | 60 | 低 | ✅ 备选 |

**推荐 848x480@30fps**：
- 比 1280x720 节省 ~45% 带宽
- 比 640x480 保留更多细节
- 30fps 稳定可靠
- 下采样到 224x224 质量足够

### 3.4 预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 帧率 | ~8 fps | **25-30 fps** |
| 深度滤波 | 每帧 3 个 | 离线处理 |
| 自动曝光 | 开启 | 关闭（一致性） |
| 采集方式 | 同步阻塞 | 后台线程 |

---

## 4. 机械臂优化（网线直连）

### 4.1 网络延迟分析

| 连接方式 | 典型延迟 | 可达频率 |
|----------|----------|----------|
| WiFi (当前) | 50-70 ms | ~16 Hz |
| **网线直连** | **1-5 ms** | **~100-200 Hz** |

### 4.2 优化方案

```python
# collectors/robot_arm.py - _poll_loop() 优化

def _poll_loop(self) -> None:
    """网线直连后可达 ~100-200 Hz。"""
    while self.running:
        if self.handle is None:
            time.sleep(0.1)
            continue

        loop_start = time.perf_counter()

        try:
            status = self.robot.rm_get_current_arm_state()
            if not isinstance(status, tuple) or len(status) != 2:
                state = {"code": -1, "data": None}
            else:
                code, payload = status
                state = {"code": code, "data": payload}
        except Exception:
            state = {"code": -1, "data": None}

        # 使用高精度时间戳
        timestamp_us = int(time.perf_counter() * 1e6)

        payload = state.get("data") if isinstance(state, dict) else None
        joints = payload.get("joint") if isinstance(payload, dict) else None
        pose = payload.get("pose") if isinstance(payload, dict) else None

        with self.lock:
            self.latest_state = state
            self.latest_joints = list(joints) if joints else None
            self.latest_pose = list(pose) if pose else None

            if self.recording and self.csv_writer is not None:
                row = [timestamp_us]
                row += list(joints) if joints else [0] * 7
                row += list(pose) if pose else [0] * 6
                self.row_buffer.append(row)
                self._flush_locked(force=False)

        # 不 sleep，让 SDK 调用耗时决定节奏
        # 网线直连时单次调用 ~1-5ms，可达 ~200Hz
```

### 4.3 预期效果

| 指标 | WiFi (当前) | 网线直连 |
|------|-------------|----------|
| 延迟 | 50-70 ms | 1-5 ms |
| 频率 | ~16 Hz | **100-200 Hz** |
| 抖动 | ±20 ms | ±2 ms |

---

## 5. 夹爪优化

### 5.1 时间戳去重

当前 110/306 session 存在重复时间戳（占 ~60% 数据量）。

```python
# collectors/gripper_state.py - _poll_loop() 添加去重

def _poll_loop(self) -> None:
    last_timestamp_us = 0  # 新增：去重

    while self.running:
        # ... 读取数据 ...

        timestamp_us = int(time.perf_counter() * 1e6)

        # 去重：跳过相同时间戳
        if timestamp_us == last_timestamp_us:
            continue
        last_timestamp_us = timestamp_us

        # ... 写入 buffer ...
```

---

## 6. 图像编码优化

### 6.1 当前问题

`collect_data.py` 中使用 QImage 保存图像：
```python
_bgr_to_qimage(dji_frame).save(str(path))  # ~8-12ms，经过 Qt 转换
```

### 6.2 优化方案

```python
# 方案 A：cv2.imwrite（简单直接）
cv2.imwrite(str(path), dji_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # ~3-5ms

# 方案 B：cv2.imencode + 异步写入（最高性能）
_, buf = cv2.imencode('.jpg', dji_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
# 将 buf 放入异步写入队列
```

### 6.3 JPEG 质量选择

| 质量 | 文件大小 (1280x720) | 训练影响 | 推荐 |
|------|---------------------|----------|------|
| 75 | ~50-70 KB | 轻微块状 | - |
| **85** | **~60-90 KB** | **可忽略** | ✅ 推荐 |
| 90 | ~90-130 KB | 无 | - |
| 95 | ~120-180 KB | 无 | 存储浪费 |

**推荐 JPEG Q85**：节省 40-50% 存储，训练效果无影响。主流数据集（Open X-Embodiment, RT-X）均使用 80-95。

### 6.4 预期效果

| 指标 | 优化前 (QImage) | 优化后 (cv2 Q85) |
|------|-----------------|------------------|
| 编码耗时 | 8-12 ms | 3-5 ms |
| 文件大小 | ~150 KB | ~75 KB |
| 存储节省 | - | **~50%** |

---

## 7. 时间戳同步方案

### 7.1 当前问题

| 模态 | 时间戳来源 | 问题 |
|------|-----------|------|
| 图像 | 无 | ❌ 只有序列号 |
| 压力 | 传感器时钟 | ⚠️ 与主机时钟不同域 |
| 机械臂 | `time.time()` | ⚠️ 受 NTP 调整影响 |
| 夹爪 | `time.time()` | ⚠️ 同上 |

### 7.2 优化方案：双时间戳 + 单调时钟

```python
import time

class MonotonicTimestamp:
    """保证单调递增的时间戳生成器。"""
    def __init__(self):
        self._last = 0
        self._offset = time.time_ns() - time.monotonic_ns()

    def get_us(self) -> int:
        ts = int((time.monotonic_ns() + self._offset) / 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts

# 全局实例
_ts_gen = MonotonicTimestamp()

# 在所有采集器中使用
timestamp_us = _ts_gen.get_us()  # 单调递增，微秒精度
```

### 7.3 双时间戳记录

每个 CSV 行记录两个时间戳：

```csv
sensor_timestamp_us,host_monotonic_us,CH1,CH2,...,CH64
```

- `sensor_timestamp_us`：传感器自身时钟（保留原始数据）
- `host_monotonic_us`：主机单调时钟（用于对齐）

### 7.4 图像时间戳

新增 `frames.csv`：

```csv
frame_id,capture_monotonic_us,dji_save_us,realsense_save_us
0001,1718000001234,1718000001250,1718000001260
0002,1718000051234,1718000051250,1718000051260
```

### 7.5 时钟域对齐

```python
# 在 session 开始时记录参考点
session_start_wall = time.time()           # 墙钟时间
session_start_mono = time.monotonic()      # 单调时间
session_start_perf = time.perf_counter()   # 高精度单调

# 压力传感器时钟偏移估算
# 取前 100 个样本，计算 host_ts - sensor_ts 的中位数
pressure_clock_offset = median(host_ts[:100] - sensor_ts[:100])

# 后续对齐：sensor_ts + pressure_clock_offset ≈ host_ts
```

---

## 8. 数据记录增强

### 8.1 Session 元数据

```json
{
    "episode_id": 1,
    "session_start_time": "2026-06-11T15:30:00+08:00",
    "task_description": "pick up the cube",
    "object_type": "cube",
    "success": true,
    "robot_model": "RM75",
    "operator_id": "james",
    "environment_id": "lab_desk_1",
    "calibration_hash": "abc123",
    "collection_software_version": "acc6fb3",
    "sensor_config": {
        "dji_resolution": "1280x720",
        "dji_fps": 30,
        "dji_format": "MJPG",
        "realsense_resolution": "848x480",
        "realsense_fps": 30,
        "pressure_hz": 140
    }
}
```

### 8.2 每帧元数据

新增 `frames.csv`：

| 列名 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧序号 |
| capture_monotonic_us | int | 采集时间戳（单调时钟） |
| dji_captured | bool | DJI 是否采集成功 |
| realsense_captured | bool | RealSense 是否采集成功 |
| depth_captured | bool | 深度帧是否采集成功 |

### 8.3 采集后自动验证

```python
def verify_session(session_path: Path) -> list[str]:
    """采集完成后自动检查数据质量。"""
    issues = []

    # 1. 检查帧数匹配
    dji_count = len(list((session_path / "dji").glob("*.jpg")))
    rs_count = len(list((session_path / "realsense_rgb").glob("*.jpg")))
    if dji_count != rs_count:
        issues.append(f"帧数不匹配: DJI={dji_count}, RS={rs_count}")

    # 2. 检查频率
    pressure_csv = session_path / "pressure" / "pressure.csv"
    with open(pressure_csv) as f:
        reader = csv.reader(f)
        next(reader)
        ts = [int(row[0]) for row in reader]
    intervals = np.diff(ts)
    freq = 1e6 / np.mean(intervals)
    if freq < 100:
        issues.append(f"压力频率过低: {freq:.1f} Hz")

    # 3. 检查时间戳连续性
    gaps = np.where(intervals > 100000)[0]  # >100ms 的间隔
    if len(gaps) > 0:
        issues.append(f"压力时间戳有 {len(gaps)} 个大间隔 (>100ms)")

    # 4. 检查文件完整性
    robot_csv = session_path / "robot_state" / "robot_state.csv"
    if not robot_csv.exists():
        issues.append("缺少 robot_state.csv")

    return issues
```

---

## 9. 实施优先级

| 优先级 | 优化项 | 预期收益 | 改动量 |
|--------|--------|----------|--------|
| **P0** | DJI 切换 MJPEG + V4L2 | 8→30 fps | 小 |
| **P0** | RealSense 后台线程 + 640x480 | 8→30 fps | 中 |
| **P1** | 机械臂网线直连 + 移除 sleep | 16→100+ Hz | 小 |
| **P1** | 夹爪时间戳去重 | 减少 60% 冗余 | 小 |
| **P1** | 图像编码 cv2.imwrite Q85 | 存储 -50% | 小 |
| **P2** | 时间戳统一为单调时钟 | 对齐精度提升 | 中 |
| **P2** | 新增 frames.csv | 可追溯性 | 小 |
| **P2** | session 元数据增强 | 数据管理 | 小 |
| **P3** | 采集后自动验证 | 质量保证 | 中 |

---

## 10. 测试验证

### 10.1 频率测试

```bash
# 使用已有的 test_frequency.py
python test_frequency.py -s dji -d 30
python test_frequency.py -s realsense -d 30
python test_frequency.py -s robot -d 30
python test_frequency.py -s pressure -d 30
```

### 10.2 目标指标

| 模态 | 目标频率 | 允许偏差 |
|------|----------|----------|
| DJI | 20 fps (采集 30) | ±2 fps |
| RealSense | 20 fps (采集 30) | ±2 fps |
| 机械臂 | 100 Hz | ±10 Hz |
| 夹爪 | 100 Hz | ±10 Hz |
| 压力 | 139 Hz | ±10 Hz |

---

## 11. 修订记录

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-06-11 | v1.0 | 初始版本，基于调研和代码分析 |

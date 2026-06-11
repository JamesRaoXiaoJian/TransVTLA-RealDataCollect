# 采集脚本优化与后续注意事项

> 针对当前采集系统频率不达标、时间戳抖动等问题，提出代码优化方案和后续采集注意事项。

---

## 0. 最严重问题：Gripper 数据大量重复

**现状**：110/306 个 session 的 `gripper_state.csv` 存在重复时间戳，去重后数据量减少约 60%。

| 统计项 | 数值 |
|--------|------|
| 受影响 session 数 | 110 / 306 (36%) |
| 总重复行数 | 82,842 行 |
| 典型减少比例 | 60-70% |
| 最严重 session | `session_20260603_202235`: 2004→686 行 (-66%) |

**根因**：`GripperStateCollector._poll_loop()` 中，`rm_get_rm_plus_state_info()` 调用耗时波动较大（0-60ms）。当调用很快返回时，`time.sleep()` 等待到下一个 tick，但 `int(time.time() * 1e6)` 生成的时间戳与上一次相同（微秒精度下同一毫秒内多次调用产生相同值）。结果是同一时间戳的数据被多次写入 buffer。

**修复优先级**：P0 — 这是数据质量问题，比频率不达标更严重。

---

## 1. 当前问题根因分析

### 1.1 Robot state 采集频率：目标 200Hz → 实际 16Hz

**根因**：`collectors/robot_arm.py` 中的 `_poll_loop()` 使用同步阻塞调用：

```python
# 当前代码（line 118-151）
def _poll_loop(self) -> None:
    while self.running:
        status = self.robot.rm_get_current_arm_state()  # ← 网络调用，耗时 ~60ms
        timestamp_us = int(time.time() * 1e6)
        # ... 写入 buffer
        time.sleep(self.interval_s)  # ← 设置 10ms，但上面调用已耗时 60ms
```

**问题**：
- `rm_get_current_arm_state()` 是 TCP 网络调用，单次往返延迟 ~50-70ms
- `time.sleep(10ms)` 在 SDK 调用之后，实际循环周期 = 60ms + 10ms ≈ 70ms → ~14Hz
- `ROBOT_ARM_FPS = 100` 的设定完全无法达到

### 1.2 Gripper state 采集频率：目标 200Hz → 实际 19Hz

**根因**：与 robot state 相同，`rm_get_rm_plus_state_info()` 网络调用延迟 ~50ms。

### 1.3 Pressure 采集频率：目标 200Hz → 实际 140Hz

**根因**：硬件端 UDP 发送频率为 ~140Hz（7ms 间隔），非软件问题。

### 1.4 图像帧率：目标 20fps → 实际 8fps

**根因**：
- DJI camera 使用 `cv2.CAP_DSHOW` 后端（DirectShow，仅适用于 Windows）
- Linux 环境下应使用 V4L2 后端
- USB 带宽限制或相机硬件输出帧率不足

---

## 2. 代码优化方案

### 2.1 Robot Arm Collector 优化

#### 方案 A：降低目标频率至实际可达值（推荐，改动最小）

```python
# collectors/robot_arm.py

# 修改前
ROBOT_ARM_FPS = 100
ROBOT_ARM_INTERVAL_S = 1.0 / ROBOT_ARM_FPS  # 10ms

# 修改后
ROBOT_ARM_FPS = 15          # 匹配实际可达频率
ROBOT_ARM_INTERVAL_S = 0.0  # 不额外 sleep，让 SDK 调用耗时决定节奏
```

同时修改 `_poll_loop()`，移除无效的 sleep：

```python
def _poll_loop(self) -> None:
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

        timestamp_us = int(time.time() * 1e6)
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

        # 可选：记录实际循环耗时，用于调试
        elapsed = time.perf_counter() - loop_start
        # 不 sleep，直接进入下一次轮询
```

#### 方案 B：异步批量查询（改动较大，适合需要高频率的场景）

如果确实需要接近 200Hz 的频率，需要：
1. 使用机械臂 SDK 的异步回调模式（如果支持）
2. 或在独立线程中持续发送请求，用另一个线程收集结果
3. 需要确认 SDK 是否支持 pipeline 式请求

```python
# 伪代码示意
class RobotArmCollectorAsync:
    def __init__(self):
        self.request_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)

    def _sender_loop(self):
        """持续发送请求，不等待响应"""
        while self.running:
            try:
                status = self.robot.rm_get_current_arm_state()
                self.result_queue.put((time.time(), status))
            except Exception:
                pass

    def _recorder_loop(self):
        """从结果队列中取出并记录"""
        while self.running:
            try:
                ts, status = self.result_queue.get(timeout=0.1)
                # 写入 CSV
            except queue.Empty:
                pass
```

**注意**：需确认 `RoboticArm` 对象是否线程安全，是否支持并发调用。

---

### 2.2 Gripper State Collector 优化

与 robot arm 同理，推荐方案 A：

```python
# collectors/gripper_state.py

# 修改前
GRIPPER_FPS = 200
GRIPPER_INTERVAL_S = 1.0 / GRIPPER_FPS  # 5ms

# 修改后
GRIPPER_FPS = 15            # 匹配实际可达频率
GRIPPER_INTERVAL_S = 0.0    # 不额外 sleep
```

同时在 `_poll_loop()` 中添加去重逻辑（**关键优化**：当前 110/306 个 session 有重复时间戳，去重后数据量减少 ~60%）：

```python
def _poll_loop(self) -> None:
    next_tick = time.perf_counter()
    last_timestamp_us = 0  # 新增：用于去重

    while self.running:
        if self.handle is None:
            time.sleep(0.1)
            continue

        late_ms = max(0.0, (time.perf_counter() - next_tick) * 1000.0)
        read_start = time.perf_counter()
        try:
            code, data = self.robot.rm_get_rm_plus_state_info()
        except Exception:
            code, data = -1, {}
        read_latency_ms = (time.perf_counter() - read_start) * 1000.0

        payload = data if code == 0 and isinstance(data, dict) else {}
        # ... 解析 payload ...

        timestamp_us = int(time.time() * 1e6)

        # 去重：跳过相同时间戳（当前 110/306 session 有重复，占总行数 ~60%）
        if timestamp_us == last_timestamp_us:
            continue
        last_timestamp_us = timestamp_us

        with self.lock:
            # ... 写入 buffer ...
```

---

### 2.3 DJI Camera 优化

```python
# collectors/dji_camera.py

def start(self) -> None:
    # 修改前：使用 DirectShow（仅 Windows）
    # cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

    # 修改后：使用默认后端（Linux 自动选择 V4L2）
    cap = cv2.VideoCapture(self.index)

    # 或者显式指定 V4L2
    # cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)

    if not cap.isOpened():
        # ...
```

**进一步优化**：
```python
# 设置更高的帧率（如果相机支持）
cap.set(cv2.CAP_PROP_FPS, 30)

# 设置像素格式为 MJPEG（减少 USB 传输量）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# 设置分辨率（降低分辨率可提升帧率）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**调试命令**（查看相机支持的格式和帧率）：
```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

---

### 2.4 Pressure Collector 优化

Pressure 传感器的问题在硬件端，软件层面可做的优化有限：

```python
# collectors/pressure.py
# 建议：在数据中记录硬件实际发送频率，而非标注 200Hz

TACTILE_FPS = 140  # 修正：实际硬件发送频率
```

如果需要更高频率，需要：
1. 联系压力传感器供应商确认固件是否支持配置发送频率
2. 或在固件端进行插值上采样（不推荐，引入虚假数据）

---

### 2.5 时间戳精度优化

当前所有模态使用 `int(time.time() * 1e6)` 生成时间戳。优化建议：

```python
# 使用 time.perf_counter_ns() 获得更高精度的相对时间戳
# 或使用 time.time_ns()（Python 3.7+）

# 修改前
timestamp_us = int(time.time() * 1e6)

# 修改后（更高精度，单调递增）
timestamp_us = int(time.time_ns() / 1000)  # 纳秒转微秒
```

**注意**：`time.time()` 受系统时钟调整影响（NTP 同步等），可能出现时间戳回退。`time.time_ns()` 同样受此影响。如果需要单调递增的时间戳，可以使用：

```python
import time

class MonotonicTimestamp:
    """保证单调递增的时间戳生成器"""
    def __init__(self):
        self._last = 0
        self._offset = time.time_ns() - time.monotonic_ns()

    def get_us(self) -> int:
        ts = int((time.monotonic_ns() + self._offset) / 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts
```

---

## 3. 后续采集注意事项

### 3.1 采集前检查清单

```markdown
□ 确认机械臂网络连接正常（ping 192.168.31.92）
□ 确认压力传感器 UDP 连接正常（检查 4321 端口）
□ 确认 DJI 相机已连接（ls /dev/video*）
□ 确认 RealSense 相机已连接（rs-enumerate-devices）
□ 确认磁盘空间充足（每 session 约 25MB，100 sessions ≈ 2.5GB）
□ 运行采集脚本后先做 5 秒测试录制，检查各模态频率
□ 确认无百度网盘等同步工具正在运行（避免生成 .cfg 残留文件）
```

### 3.2 采集中的监控

建议在 `collect_data.py` 的 GUI 中添加实时频率显示：

```python
# 在采集循环中计算并显示各模态实际频率
def update_frequency_display(self):
    """每秒更新一次各模态的实际采集频率"""
    now = time.time()
    elapsed = now - self.last_freq_update
    if elapsed < 1.0:
        return

    pressure_hz = self.pressure_count / elapsed
    robot_hz = self.robot_count / elapsed
    gripper_hz = self.gripper_count / elapsed
    image_hz = self.image_count / elapsed

    self.status_label.setText(
        f"Pressure: {pressure_hz:.0f}Hz | "
        f"Robot: {robot_hz:.0f}Hz | "
        f"Gripper: {gripper_hz:.0f}Hz | "
        f"Image: {image_hz:.0f}fps"
    )

    self.pressure_count = 0
    self.robot_count = 0
    self.gripper_count = 0
    self.image_count = 0
    self.last_freq_update = now
```

### 3.3 采集后验证

每次采集完成后，立即运行快速验证：

```bash
# 快速检查最新 session 的频率
python3 -c "
import csv, os, glob
import numpy as np

# 找到最新 session
sessions = sorted(glob.glob('dataset/phase2_realdata_sessions/sessions/session_*'))
latest = sessions[-1]
print(f'Checking: {os.path.basename(latest)}')

# 检查 pressure 频率
p_csv = os.path.join(latest, 'pressure', 'pressure.csv')
with open(p_csv) as f:
    r = csv.reader(f); next(r)
    ts = [int(row[0]) for row in r]
intervals = np.diff(ts)
print(f'Pressure: {len(ts)} rows, {1e6/np.mean(intervals):.1f} Hz')

# 检查 robot 频率
r_csv = os.path.join(latest, 'robot_state', 'robot_state.csv')
with open(r_csv) as f:
    r = csv.reader(f); next(r)
    ts = [int(row[0]) for row in r]
intervals = np.diff(ts)
print(f'Robot: {len(ts)} rows, {1e6/np.mean(intervals):.1f} Hz')

# 检查图像帧数
dji_count = len(glob.glob(os.path.join(latest, 'dji', '*.jpg')))
rs_count = len(glob.glob(os.path.join(latest, 'realsense_rgb', '*.jpg')))
print(f'DJI: {dji_count} frames, RealSense: {rs_count} frames')
print(f'Match: {\"YES\" if dji_count == rs_count else \"NO\"}')" 
```

### 3.4 数据格式注意事项

| 事项 | 说明 |
|------|------|
| **时间戳单位** | 微秒（μs），`int(time.time() * 1e6)` |
| **时间戳基准** | Unix epoch（1970-01-01 00:00:00 UTC） |
| **图像编号** | 从 0001 开始，4 位零填充 |
| **CSV 编码** | UTF-8，无 BOM |
| **CSV 行尾** | LF（`\n`），不要 CRLF |
| **图像格式** | DJI/RS_RGB: JPEG, RS_Depth: PNG (16-bit) |
| **坐标系** | 机械臂为度（°），位姿为米（m）+ 弧度（rad） |

### 3.5 环境与依赖

```bash
# 固定依赖版本（避免不同版本行为差异）
pip install opencv-python==4.9.0.80
pip install pyrealsense2==2.55.1.6485
pip install PySide6==6.6.1
pip install numpy==1.26.4

# 系统依赖
sudo apt install v4l-utils  # 用于调试摄像头
```

### 3.6 存储与备份

```bash
# 采集完成后立即备份
rsync -avP --progress \
  dataset/phase2_realdata_sessions/sessions/ \
  /backup/transvtla/sessions_$(date +%Y%m%d)/

# 禁止同步工具监控数据目录
# 在百度网盘/OneDrive 等设置中排除 dataset/ 目录
```

---

## 4. 长期改进方向

### 4.1 采集系统架构优化

```
当前架构（单线程轮询，阻塞式）：
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Robot Arm│────▶│  Sleep   │────▶│  Write   │  ← 每次循环 ~70ms
└──────────┘     └──────────┘     └──────────┘

建议架构（多线程非阻塞）：
┌──────────┐     ┌──────────┐
│ Sender   │────▶│  Robot   │
│ Thread   │     │  Arm SDK │
└──────────┘     └────┬─────┘
                      │ callback / queue
┌──────────┐     ┌────▼─────┐
│ Recorder │◀────│  Result  │
│ Thread   │     │  Queue   │
└──────────┘     └──────────┘
```

### 4.2 时间同步方案

当前各模态独立记录时间戳，对齐依赖系统时钟。改进方案：

1. **硬件同步**：使用 PTP（Precision Time Protocol）同步所有设备时钟
2. **软件同步**：在采集开始时发送同步信号（如 LED 闪烁），作为对齐基准
3. **序列号方案**：为每帧数据添加全局递增序列号，不依赖时间戳

### 4.3 数据质量自动检查

在采集脚本中集成实时质量检查：

```python
def check_session_quality(session_path):
    """采集完成后自动检查数据质量"""
    issues = []

    # 检查频率
    # 检查帧数匹配
    # 检查时间戳连续性
    # 检查文件完整性

    if issues:
        show_warning_dialog(f"Session {session_path} has issues:\n" + "\n".join(issues))
```

---

## 5. 修订记录

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-06-10 | v1.0 | 初始版本，基于 306 session 全量审计结果 |

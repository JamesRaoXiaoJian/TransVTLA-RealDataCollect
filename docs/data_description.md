# TransVTLA 真实数据集说明

## 数据集概览

| 物体类别 | 采样次数 | Session 数 |
|----------|---------|-----------|
| cube（方块） | 10 | 10 |
| cup（杯子） | 10 | 11 |
| softbottle（软瓶） | 5 | 5 |
| colacup（可乐杯） | 3 | 3 |
| **合计** | — | **29** |

## 目录结构

```
sessions/
├── {object}_sample{N}/                    # 物体类别 + 采样编号
│   └── session_{YYYYMMDD}_{HHMMSS}/       # 采集时间戳
│       ├── pressure/                      # 触觉传感器原始数据
│       │   └── pressure.csv
│       ├── preprocessed_pressure/         # 预处理后的张量数据
│       │   └── session_{...}.npz
│       ├── world_camera/                  # world RealSense RGB-D
│       │   ├── rgb/
│       │   │   ├── 0001.jpg
│       │   │   └── ...
│       │   └── depth/
│       │       ├── 0001.png               # uint16 毫米深度图，对齐 rgb
│       │       └── ...
│       ├── wrist_camera/                  # wrist RealSense RGB-D
│       │   ├── rgb/
│       │   │   ├── 0001.jpg
│       │   │   └── ...
│       │   └── depth/
│       │       ├── 0001.png               # uint16 毫米深度图，对齐 rgb
│       │       └── ...
│       ├── camera_metadata.json           # 两台 RealSense 的 profile / 内参 / depth scale
│       └── robot_state/                   # 机械臂与夹爪状态
│           ├── robot_state.csv            # 机械臂关节与末端位姿
│           └── gripper_state.csv          # 可选，夹爪 RM Plus 实时状态
```

不同采集脚本生成的数据范围不同：

| 脚本 | 图像 | 机械臂 | 压力 | 夹爪 |
|------|------|--------|------|------|
| `collect_vision.py` | 是 | 否 | 否 | 否 |
| `collect_robot.py` | 是 | 是 | 否 | 否 |
| `collect_pressure.py` | 是 | 是 | 是 | 否 |
| `collect_data.py` | 是 | 是 | 是 | 是 |

## 原始数据格式

### 1. 触觉传感器数据 — `pressure/pressure.csv`

新采集 CSV 只保存 20 个建模有效通道，格式为 22 列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `sensor_timestamp_us` | int | 压力包内传感器时间戳 |
| `host_monotonic_us` | int | 采集主机单调时钟时间戳，用于跨模态同步 |
| 标准 20 个 `CH*` | int | 有效触觉通道 ADC 原始读数 |

- 采样率：~200 Hz
- 传感器为柔性触觉传感器，ADC 值越大代表压力越小（无接触时约 4900~5007）
- 旧版 64 通道 CSV 可用 `scripts/trim_pressure_channels.py` 先备份再裁剪为标准 20 通道。

### 2. 机械臂状态 — `robot_state/robot_state.csv`

CSV 格式，14 列：

| 列名 | 说明 |
|------|------|
| `timestamp_us` | 时间戳（微秒） |
| `joint_1` ~ `joint_7` | 7 个关节角度（度） |
| `pose_1` ~ `pose_6` | 末端执行器位姿（位置 + 姿态） |

- 目标采样率：100 Hz，实际采样率受机械臂 SDK 查询耗时影响。
- 实际采样率受机械臂 SDK 查询耗时影响。

### 3. 夹爪状态 — `robot_state/gripper_state.csv`

该文件仅由 `collect_data.py` 生成，用于记录官方状态读取 API 返回的夹爪实时状态。

CSV 格式，12 列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp_us` | int | 采样时间戳（微秒，采集端系统时间） |
| `target_hz` | int | 目标采样率，当前为 100 |
| `rm_plus_read_code` | int | `rm_get_rm_plus_state_info()` 返回码，0 表示成功 |
| `rm_plus_read_latency_ms` | float | 本次状态读取耗时，单位毫秒 |
| `sys_state` | int/空 | 末端设备系统状态 |
| `gripper_pos` | int/空 | 夹爪开合值，来源于 `dist["pos"][0]` |
| `gripper_speed` | int/空 | 夹爪速度，来源于 `dist["speed"][0]` |
| `gripper_current` | int/空 | 夹爪电流，来源于 `dist["current"][0]` |
| `gripper_force` | int/空 | 夹爪力，来源于 `dist["force"][0]` |
| `gripper_dof_state` | int/空 | 第 0 自由度状态，来源于 `dist["dof_state"][0]` |
| `gripper_dof_err` | int/空 | 第 0 自由度错误码，来源于 `dist["dof_err"][0]` |
| `deadline_late_ms` | float | 兼容字段，当前不再依赖固定 sleep 调度 |

采集调用：

```python
code, dist = arm.rm_get_rm_plus_state_info()
gripper_pos = dist["pos"][0]
```

注意：

- 采集代码只读取状态，不下发夹爪动作。
- 夹爪动作可由控制程序使用 `arm.rm_set_gripper_position()` 等官方接口完成。
- 后处理时应使用 `timestamp_us` 和 `rm_plus_read_latency_ms` 判断真实同步质量。

### 4. 相机图像 — `world_camera/` & `wrist_camera/`

- 标准采集 profile：`848x480@30`
- RGB 格式：JPEG（`rgb/XXXX.jpg`，四位序号）
- 深度格式：uint16 PNG（`depth/XXXX.png`，四位序号），单位为毫米，已对齐到 RGB/color 像素
- `world_camera`：固定外部 RealSense 主视角
- `wrist_camera`：腕部 RealSense 视角
- 目标保存频率：30 Hz
- `camera_metadata.json`：保存每台相机的 RealSense SDK 内参、distortion、depth scale、序列号和深度保存单位
- `frames.csv`：保存每个视觉帧的 `capture_monotonic_us` 和各图像文件落盘时间戳。

旧版数据使用 `dji/` 作为 world RGB、`realsense_rgb/` 作为 wrist RGB、`realsense_depth/` 作为 wrist depth。当前回放、标注和转换脚本兼容旧格式。

RLDS/TFDS 转换输出中，图像字段保持 `primary_image` / `wrist_image`，深度字段为：

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `primary_depth` | `(224, 224, 1)` | `uint16` | world depth，单位毫米，对齐 `primary_image` |
| `wrist_depth` | `(224, 224, 1)` | `uint16` | wrist depth，单位毫米，对齐 `wrist_image` |

转换时深度图使用最近邻缩放，避免插值改变毫米深度值。旧版 DJI world 数据没有 world depth 时，`primary_depth` 填零。

## 触觉数据预处理

### 通道映射

标准压力 CSV 直接保存 20 个有效通道；旧 64 通道 CSV 在预处理时会按以下顺序提取：

```
[左总压(1), 右总压(1), 左矩阵(3×3=9), 右矩阵(3×3=9)] = 20 通道
```

| 位置 | 通道编号 |
|------|---------|
| 左总压 | CH51 |
| 右总压 | CH50 |
| 左矩阵第1行 | CH63, CH60, CH57 |
| 左矩阵第2行 | CH64, CH61, CH58 |
| 左矩阵第3行 | CH49, CH62, CH59 |
| 右矩阵第1行 | CH47, CH44, CH41 |
| 右矩阵第2行 | CH48, CH45, CH42 |
| 右矩阵第3行 | CH33, CH46, CH43 |

当前映射来自 `channel_mapping.json`。其中 CH58 会在预处理阶段按配置用相邻通道插值替代。

### 处理流程

```
标准 CSV (N, 20) 或旧 CSV (N, 64)
    ↓ 统一为有效通道顺序 (N, 20)
    ↓ 动态基线消除
(N, 20)   ΔP = B₀ - P_raw,  clip ≥ 0
    ↓ Min-Max 归一化
(N, 20)   P_norm = ΔP / 3500,  clip [0, 1]
    ↓ 滑动窗口切片
(Samples, 16, 20)   window=16, stride=1
```

#### Step 1: 提取有效通道

新数据已是标准 20 通道；旧数据从 64 通道中按固定顺序取出 20 个有效通道。

#### Step 2: 动态基线消除

取每个 session 前 50 行（约 0.25 秒 @ 200Hz）的均值作为基线 $B_0$：

$$\Delta P = \max(0,\ B_0 - P_{raw})$$

ADC 值越大代表压力越小，因此取反。负值截断为 0 以消除底噪。

#### Step 3: 归一化

除以全局最大压力差常量（3500 ADC 单位），截断到 [0, 1]：

$$P_{norm} = \text{clip}\left(\frac{\Delta P}{3500},\ 0,\ 1\right)$$

#### Step 4: 滑动窗口切片

以 window=16、stride=1 切片，生成三维张量：

- 输入：`(N, 20)`
- 输出：`(Samples, 16, 20)`，其中 `Samples = N - 16 + 1`

### 预处理参数

| 参数 | 值 | 说明 |
|------|---|------|
| `BASELINE_ROWS` | 50 | 基线计算行数 |
| `MAX_PRESSURE_DROP` | 3500.0 | 归一化分母（ADC 单位） |
| `WINDOW_SIZE` | 16 | 滑动窗口大小 |
| `STRIDE` | 1 | 滑动窗口步长 |

### 输出文件 — `preprocessed_pressure/session_{...}.npz`

每个 session 单独保存为一个 `.npz` 文件，包含：

| Key | 形状/值 | 说明 |
|-----|--------|------|
| `data` | `(Samples, 16, 20)` | 预处理后的压力张量 |
| `channels` | `(20,)` | 有效通道编号列表 |
| `window_size` | 16 | 窗口大小 |
| `stride` | 1 | 步长 |
| `max_pressure_drop` | 3500.0 | 归一化常量 |
| `baseline_rows` | 50 | 基线行数 |
| `host_monotonic_us` | `(N,)` | 压力原始行主机单调时间戳 |
| `sensor_timestamp_us` | `(N,)` | 压力原始行传感器时间戳 |
| `window_start_host_us` | `(Samples,)` | 每个触觉窗口起点主机时间 |
| `window_end_host_us` | `(Samples,)` | 每个触觉窗口终点主机时间 |
| `window_center_host_us` | `(Samples,)` | 每个触觉窗口中心主机时间，RLDS 对齐优先使用 |

## 时间同步说明

各模态数据由不同采集线程写入，统一依赖时间戳做后处理对齐：

| 模态 | 文件 | 时间戳字段 | 目标采样率 |
|------|------|------------|------------|
| World RGB | `world_camera/rgb/XXXX.jpg` | `frames.csv:capture_monotonic_us` | 30 Hz |
| World Depth | `world_camera/depth/XXXX.png` | `frames.csv:capture_monotonic_us` | 30 Hz |
| Wrist RGB | `wrist_camera/rgb/XXXX.jpg` | `frames.csv:capture_monotonic_us` | 30 Hz |
| Wrist Depth | `wrist_camera/depth/XXXX.png` | `frames.csv:capture_monotonic_us` | 30 Hz |
| 压力 | `pressure/pressure.csv` | `host_monotonic_us` | 200 Hz |
| 机械臂 | `robot_state/robot_state.csv` | `timestamp_us`，采集端系统时间 | 100 Hz 目标 |
| 夹爪 | `robot_state/gripper_state.csv` | `timestamp_us`，采集端系统时间 | 100 Hz 目标 |

对齐建议：

- 先运行 `scripts/build_sync_index.py --data-root sessions` 生成 `sync/sync_index.csv`。
- 图像以 `frames.csv:capture_monotonic_us` 为标准时间轴。
- 压力使用 `host_monotonic_us` 做最近邻或线性插值对齐。
- 触觉窗口使用 `preprocessed_pressure/*.npz` 中的 `window_center_host_us` 对齐。
- 机械臂、夹爪使用 `timestamp_us` 做最近邻或线性插值对齐。
- 夹爪数据需要同时检查 `rm_plus_read_code == 0`。
- 如果 `rm_plus_read_latency_ms` 较大，说明该行夹爪数据的实时性较差。

## 使用方法

```bash
# 激活环境
source .venv/bin/activate

# 运行预处理
python preprocess_pressure.py --data-root sessions

# 运行内置测试
python preprocess_pressure.py --test
```

```python
# 加载预处理数据
import numpy as np

data = np.load("sessions/cube_sample10/session_20260507_191639/"
               "preprocessed_pressure/session_20260507_191639.npz")
tensors = data["data"]  # shape: (4534, 16, 20)
```

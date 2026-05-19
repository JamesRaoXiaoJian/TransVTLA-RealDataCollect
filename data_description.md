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
│       ├── realsense_rgb/                 # RealSense 相机 RGB 图像
│       │   ├── 0001.jpg
│       │   ├── 0002.jpg
│       │   └── ...
│       ├── dji/                           # DJI 相机图像
│       │   ├── 0001.jpg
│       │   ├── 0002.jpg
│       │   └── ...
│       └── robot_state/                   # 机械臂与夹爪状态
│           ├── robot_state.csv            # 机械臂关节与末端位姿
│           └── gripper_state.csv          # 可选，知行夹爪 Modbus 状态
```

不同采集脚本生成的数据范围不同：

| 脚本 | 图像 | 机械臂 | 压力 | 夹爪 |
|------|------|--------|------|------|
| `collect_vision.py` | 是 | 否 | 否 | 否 |
| `collect_robot.py` | 是 | 是 | 否 | 否 |
| `collect_pressure.py` | 是 | 是 | 是 | 否 |
| `collect_gripper.py` | 是 | 是 | 是 | 是 |

## 原始数据格式

### 1. 触觉传感器数据 — `pressure/pressure.csv`

CSV 格式，65 列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp_us` | int | 时间戳（微秒） |
| `CH1` ~ `CH64` | int | 64 通道 ADC 原始读数（0~5007） |

- 采样率：~200 Hz
- 传感器为柔性触觉传感器，ADC 值越大代表压力越小（无接触时约 4900~5007）

### 2. 机械臂状态 — `robot_state/robot_state.csv`

CSV 格式，14 列：

| 列名 | 说明 |
|------|------|
| `timestamp_us` | 时间戳（微秒） |
| `joint_1` ~ `joint_7` | 7 个关节角度（度） |
| `pose_1` ~ `pose_6` | 末端执行器位姿（位置 + 姿态） |

- 目标采样率：200 Hz
- 实际采样率受机械臂 SDK 查询耗时影响。

### 3. 夹爪状态 — `robot_state/gripper_state.csv`

该文件仅由 `collect_gripper.py` 生成，用于记录知行 CTAG2F120 夹爪的 Modbus 寄存器读数。

CSV 格式，10 列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp_us` | int | 采样时间戳（微秒，采集端系统时间） |
| `target_hz` | int | 目标采样率，当前为 200 |
| `position_read_code` | int | 读取位置寄存器返回码，0 表示成功 |
| `position_read_latency_ms` | float | 本次 Modbus 读取耗时，单位毫秒 |
| `position_b0` | int/空 | 位置寄存器第 1 字节 |
| `position_b1` | int/空 | 位置寄存器第 2 字节 |
| `position_b2` | int/空 | 位置寄存器第 3 字节 |
| `position_b3` | int/空 | 位置寄存器第 4 字节 |
| `position_value` | int/空 | 由 4 字节大端解码得到的位置值，范围通常为 0~1000 |
| `deadline_late_ms` | float | 相对 200 Hz 调度周期的延迟，单位毫秒 |

位置寄存器地址为 `258`，读取 2 个 Modbus 寄存器，即 4 个字节。解码方式：

```python
position_value = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
```

常见位置值：

| 夹爪位置 | 字节值 | 十进制 |
|----------|--------|--------|
| 打开 | `[0, 0, 0, 0]` | 0 |
| 中间 | `[0, 0, 1, 244]` | 500 |
| 闭合 | `[0, 0, 3, 232]` | 1000 |

注意：

- `position_value` 是 Modbus 位置寄存器读回值，主要表示最近写入/保持的目标位置。
- 它不一定等同于实时物理开口位置。
- 当前通过睿尔曼 API 轮询 Modbus 的实测速度低于 200 Hz，通常受 `position_read_latency_ms` 限制。
- 后处理时应使用 `timestamp_us` 和 `position_read_latency_ms` 判断真实同步质量。

### 4. 相机图像 — `realsense_rgb/` & `dji/`

- 格式：JPEG（`XXXX.jpg`，四位序号）
- `realsense_rgb`：RealSense 相机视角
- `dji`：DJI 相机视角
- 目标采样率：20 Hz

## 触觉数据预处理

### 通道映射

从 64 个 ADC 通道中提取 20 个有效通道，按以下顺序排列：

```
[左总压(1), 右总压(1), 左矩阵(3×3=9), 右矩阵(3×3=9)] = 20 通道
```

| 位置 | 通道编号 |
|------|---------|
| 左总压 | CH19 |
| 右总压 | CH18 |
| 左矩阵第1行 | CH1, CH16, CH15 |
| 左矩阵第2行 | CH14, CH13, CH12 |
| 左矩阵第3行 | CH11, CH10, CH9 |
| 右矩阵第1行 | CH17, CH32, CH31 |
| 右矩阵第2行 | CH30, CH29, CH28 |
| 右矩阵第3行 | CH27, CH26, CH25 |

### 处理流程

```
原始 CSV (N, 64)
    ↓ 提取 20 个有效通道
(N, 20)
    ↓ 动态基线消除
(N, 20)   ΔP = B₀ - P_raw,  clip ≥ 0
    ↓ Min-Max 归一化
(N, 20)   P_norm = ΔP / 3500,  clip [0, 1]
    ↓ 滑动窗口切片
(Samples, 16, 20)   window=16, stride=1
```

#### Step 1: 提取有效通道

从 64 通道中按固定顺序取出 20 个有效通道。

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

## 时间同步说明

各模态数据由不同采集线程写入，统一依赖时间戳做后处理对齐：

| 模态 | 文件 | 时间戳字段 | 目标采样率 |
|------|------|------------|------------|
| DJI 图像 | `dji/XXXX.jpg` | 文件序号对应视觉采样顺序 | 20 Hz |
| RealSense 图像 | `realsense_rgb/XXXX.jpg` | 文件序号对应视觉采样顺序 | 20 Hz |
| 压力 | `pressure/pressure.csv` | `timestamp_us`，来自压力数据包 | 200 Hz |
| 机械臂 | `robot_state/robot_state.csv` | `timestamp_us`，采集端系统时间 | 200 Hz |
| 夹爪 | `robot_state/gripper_state.csv` | `timestamp_us`，采集端系统时间 | 200 Hz 目标，实际取决于 Modbus 读取耗时 |

对齐建议：

- 图像以帧序号和采集 session 时间作为低频参考。
- 压力、机械臂、夹爪使用 `timestamp_us` 做最近邻或线性插值对齐。
- 夹爪数据需要同时检查 `position_read_code == 0`。
- 如果 `position_read_latency_ms` 或 `deadline_late_ms` 较大，说明该行夹爪数据的实时性较差。

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

# TransVTLA-RealDataCollect

面向 **机械臂 + 双 RealSense RGB-D 相机 + 压力传感器** 的多模态数据采集项目。

这个仓库的目标是把采集、回放、调试和机械臂交互整理成一套脚本工具，方便在实验现场快速记录同步数据，并在采集完成后离线检查每个 session。

## 项目内容

- **Intel RealSense RGB-D 相机 x2**：分别作为 world camera 和 wrist camera，同时采集 RGB 与深度图。
- **睿尔曼机械臂**：采集当前位姿、关节状态和软件状态。
- **压力数据**：通过 UDP 接收下位机发送的数据包并保存为 CSV。

## 目录说明

- `Robotic_Arm/`：机械臂 Python SDK，本仓库已内置。
- `collect_data.py`：多模态数据采集主程序（双 RealSense RGB-D + 机械臂 + 压力 + 夹爪），PySide6 界面。
- `test_frequency.py`：逐项测试压力、机械臂、夹爪和双 RealSense 的采集频率。
- `data_viwer.py`：离线浏览已采集 session 的工具。
- `calibration/world_realsense_calibration.py`：固定 world RealSense 相机到机械臂基座的外参标定。
- `calibration/hand_eye_calibration.py`：wrist RealSense 手眼标定，读取标准 profile 下的 SDK 内参。
- `scripts/build_sync_index.py`：为每个 session 生成标准多模态时间同步索引。
- `scripts/trim_pressure_channels.py`：备份并裁剪历史压力 CSV，只保留标准 20 个触觉通道。
- `Phase2_build_RLDSdata.py`：把采集 session 转换为 RGB-D RLDS/TFDS 数据。
- `keybordControl.py`：键盘控制机械臂并记录位姿。
- `connect_robot.py`：快速连接机械臂并打印当前状态。

## 环境依赖

建议使用 Python 虚拟环境，并安装以下依赖：

```bash
pip install opencv-python numpy keyboard
pip install pyrealsense2
```

说明：

- `pyrealsense2` 需要根据你的 RealSense 驱动和 Python 版本安装。
- 机械臂相关脚本依赖本仓库内的 `Robotic_Arm` 包。

## 数据输出结构

常见采集结果会按时间戳创建 session，例如：

```text
sessions/
	session_YYYYMMDD_HHMMSS/
		world_camera/
			rgb/
			depth/
		wrist_camera/
			rgb/
			depth/
		camera_metadata.json
		robot_state/
		pressure/
```

不同脚本保存的内容略有差异：

- `world_camera/rgb/`：world RealSense RGB 图像帧。
- `world_camera/depth/`：world RealSense 深度图，uint16 PNG，单位毫米，已对齐 RGB。
- `wrist_camera/rgb/`：wrist RealSense RGB 图像帧。
- `wrist_camera/depth/`：wrist RealSense 深度图，uint16 PNG，单位毫米，已对齐 RGB。
- `camera_metadata.json`：两台 RealSense 的 `848x480@30` profile、SDK 内参、畸变、序列号和 depth scale。
- `robot_state/`：机械臂状态 JSON。
- `pressure/`：压力数据 CSV，仅保存标准 20 个有效触觉通道。
- `sync/`：标准时间同步索引，由 `scripts/build_sync_index.py` 生成。

旧版 `dji/` + `realsense_rgb/` + `realsense_depth/` 会话仍可由回放、标注和转换脚本读取。

RLDS/TFDS 转换会输出 `primary_image`、`wrist_image`、`primary_depth`、`wrist_depth`。两路深度为 `uint16`，形状 `224x224x1`，单位毫米。

## 脚本说明与用法

### `test_frequency.py`

用途：正式采集前检查各传感器和两台 RealSense 同时运行时的实际频率。

常用示例：

```bash
python test_frequency.py --sensor dual_realsense --duration 30
python test_frequency.py --sensor world_camera --world-serial 1234567890
python test_frequency.py --sensor wrist_camera --wrist-serial 0987654321
```

### `collect_data.py`

用途：采集 world/wrist 两台 RealSense 的 RGB-D 数据 + 机械臂状态 + 压力 UDP 数据 + 夹爪官方状态 API 数据。

新增内容：

- 录制期间调用 `rm_get_rm_plus_state_info()`，将夹爪实时状态写入 `robot_state/gripper_state.csv`。
- 夹爪开合值记录为 `gripper_pos`，来源于 `dist["pos"][0]`。
- 视觉帧保存频率为 `30Hz`，图像写盘和界面显示异步，保存性能优先。
- 压力 CSV 只保存 `channel_config.py` 中定义的 20 个有效触觉通道，并保留 `sensor_timestamp_us` / `host_monotonic_us`。
- 数据采集模块独立为 `collectors/` 包，每个采集器可单独调用。

示例：

```bash
python collect_data.py --arm-host 172.25.5.243 --arm-port 8080 \
  --world-serial 1234567890 --wrist-serial 0987654321
```

常用参数：

- `--world-serial`：world RealSense 序列号；不填时会从已连接设备自动选择。
- `--wrist-serial`：wrist RealSense 序列号；不填时会从已连接设备自动选择。
- `--width` / `--height`：RGB 和深度流标准分辨率，默认 `848x480`。
- `--rs-fps`：RealSense 设备标准采集 FPS，默认 `30`。
- `--disable-gripper`：关闭夹爪状态采集。

### 标准同步与历史压力数据裁剪

正式转 RLDS 前先生成同步索引：

```bash
python scripts/build_sync_index.py --data-root sessions
```

历史压力 CSV 如仍为 64 通道格式，先 dry-run，再执行带备份裁剪：

```bash
python scripts/trim_pressure_channels.py --data-root sessions
python scripts/trim_pressure_channels.py --data-root sessions --apply
```

执行后每个被裁剪的文件会保留 `pressure/pressure.full64.backup.csv`，并写入 `pressure/channel_trim_manifest.json`。

### `data_viwer.py`

用途：离线浏览 `sessions/` 下的采集结果。

浏览方式：

- 使用滑块切换帧。
- 使用左右方向键前后切换。
- `Q` 或 `ESC` 退出。

示例：

```bash
python data_viwer.py --sessions sessions
python data_viwer.py --sessions sessions --session session_20260429_120000
```

### `pressure_data.py`

用途：接收压力 UDP 数据并保存到 CSV 文件。

它会：

- 向下位机发送一次 `HELLO` 握手。
- 持续监听 UDP 数据包。
- 定期把 `left/right` 压力值写入 `pressure_logs/`。

示例：

```bash
python pressure_data.py
```

### `herong_9_pressure_data.py`

用途：在终端内实时显示 64 路压力通道数据。

特点：

- 不会持续刷屏，而是原地刷新固定区域。
- 适合现场快速查看压力数据是否正常。

示例：

```bash
python herong_9_pressure_data.py
```

### `keybordControl.py`

用途：键盘控制机械臂，并高频记录位姿到 `pose_logs/`。

默认按键：

- `W` / `↑`：Z 轴正方向。
- `S` / `↓`：Z 轴负方向。
- `A` / `←`：Y 轴正方向。
- `D` / `→`：Y 轴负方向。
- `Q`：X 轴正方向。
- `E`：X 轴负方向。
- `ESC`：退出并保存日志。

示例：

```bash
python keybordControl.py
```

### `connect_robot.py`

用途：最小化的机械臂连接示例，适合先验证 IP、端口和 SDK 是否可用。

示例：

```bash
python connect_robot.py
```

### `test.py`

用途：机械臂 SDK 的调试脚本，会打印软件信息和当前位姿。

示例：

```bash
python test.py
```

## 推荐工作流

1. 先运行 `connect_robot.py` 或 `test.py`，确认机械臂可连接。
2. 再用 `test_frequency.py --sensor dual_realsense` 检查两台 RealSense 是否能同时出 30Hz RGB-D 帧对。
3. 使用 `test_frequency.py --sensor pressure` 验证压力通信。
4. 运行 `collect_data.py` 开始正式采集。
5. 采集后运行 `scripts/build_sync_index.py`，再运行压力预处理和 RLDS 转换。

## 采集注意事项

- 请确认两台 RealSense、压力下位机和机械臂的 IP / 端口配置正确。
- 采集前建议先用序列号绑定 world/wrist 两台相机，避免设备枚举顺序变化。
- 压力脚本依赖下位机发送符合 `<Q64h>` 格式的数据包。
- `sessions/`、`pose_logs/`、`pressure_logs/`、`sample_output/` 等目录属于运行输出，通常不需要提交到 Git。

## 许可证与说明

本仓库包含机械臂二次开发相关代码和数据采集脚本，适合实验环境下的数据同步采集与调试。

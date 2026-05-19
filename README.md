# TransVTLA-RealDataCollect

面向 **机械臂 + DJI 相机 + RealSense + 压力传感器** 的多模态数据采集项目。

这个仓库的目标是把采集、回放、调试和机械臂交互整理成一套脚本工具，方便在实验现场快速记录同步数据，并在采集完成后离线检查每个 session。

## 项目内容

- **DJI Osmo Action**：通过 UVC / OpenCV 读取视频流。
- **Intel RealSense D435**：采集 RGB 图像，部分脚本也支持深度图。
- **睿尔曼机械臂**：采集当前位姿、关节状态和软件状态。
- **压力数据**：通过 UDP 接收下位机发送的数据包并保存为 CSV。

## 目录说明

- `Robotic_Arm/`：机械臂 Python SDK，本仓库已内置。
- `camera_data.py`：DJI + RealSense 的预览/小批量采集工具。
- `collect_vision.py`：DJI + RealSense 的交互式录制工具。
- `collect_robot.py`：DJI + RealSense + 机械臂状态的同步采集工具。
- `collect_pressure.py`：DJI + RealSense + 机械臂状态 + 压力 UDP 数据保存。
- `collect_gripper.py`：在压力采集基础上增加夹爪官方状态 API 读取。
- `data_viwer.py`：离线浏览已采集 session 的工具。
- `pressure_data.py`：压力 UDP 数据接收并保存为 CSV。
- `herong_9_pressure_data.py`：压力数据的终端实时显示工具。
- `keybordControl.py`：键盘控制机械臂并记录位姿。
- `connect_robot.py`：快速连接机械臂并打印当前状态。
- `test.py`：机械臂 SDK 的快速测试脚本。

## 环境依赖

建议使用 Python 虚拟环境，并安装以下依赖：

```bash
pip install opencv-python numpy keyboard
pip install pyrealsense2
```

说明：

- `pyrealsense2` 需要根据你的 RealSense 驱动和 Python 版本安装。
- Windows 下 DJI 相机脚本使用 `cv2.CAP_DSHOW` 打开摄像头。
- 机械臂相关脚本依赖本仓库内的 `Robotic_Arm` 包。

## 数据输出结构

常见采集结果会按时间戳创建 session，例如：

```text
sessions/
	session_YYYYMMDD_HHMMSS/
		dji/
		realsense_rgb/
		robot_state/
		pressure/
```

不同脚本保存的内容略有差异：

- `dji/`：DJI 图像帧。
- `realsense_rgb/`：RealSense RGB 图像帧。
- `robot_state/`：机械臂状态 JSON。
- `pressure/`：压力数据 CSV。

## 脚本说明与用法

### `camera_data.py`

用途：预览或小批量保存 DJI + RealSense 的图像数据。

常用示例：

```bash
python camera_data.py --mode preview
python camera_data.py --mode capture --output sample_output
python camera_data.py --list-cameras
```

主要参数：

- `--dji-index`：DJI 摄像头索引。
- `--mode`：`preview` 或 `capture`。
- `--output`：输出目录。
- `--frame-count`：采集帧组数。
- `--depth-preview`：是否显示/保存深度伪彩色图。

### `collect_vision.py`

用途：交互式采集 DJI + RealSense RGB，同步保存到按时间戳命名的 session 目录。

操作方式：

- `SPACE`：开始 / 暂停录制。
- `Q` 或 `ESC`：退出。

示例：

```bash
python collect_vision.py --output sessions
```

### `collect_robot.py`

用途：采集 DJI + RealSense RGB + 机械臂状态。

录制时会额外保存：

- 机械臂当前状态 JSON。
- 当前帧对应的关节信息与位姿信息。

示例：

```bash
python collect_robot.py --arm-host 192.168.31.92 --arm-port 8080
```

常用参数：

- `--arm-host`：机械臂控制器 IP。
- `--arm-port`：机械臂控制器端口。
- `--output`：session 保存目录。

### `collect_pressure.py`

用途：采集 DJI + RealSense RGB + 机械臂状态 + 压力 UDP 数据。

新增内容：

- 启动压力 UDP 监听。
- 录制期间将压力数据写入 `pressure/pressure.csv`。
- 压力数据包含 `timestamp_us`、`left`、`right` 三列。

示例：

```bash
python collect_pressure.py --arm-host 192.168.31.92 --arm-port 8080
```

常用参数：

- `--arm-host`：机械臂控制器 IP。
- `--arm-port`：机械臂控制器端口。
- `--output`：session 保存目录。
- `--dji-index`：DJI 摄像头索引。

### `collect_gripper.py`

用途：采集 DJI + RealSense RGB + 机械臂状态 + 压力 UDP 数据 + 夹爪官方状态 API 数据。

新增内容：

- 录制期间调用 `rm_get_rm_plus_state_info()`，将夹爪实时状态写入 `robot_state/gripper_state.csv`。
- 夹爪开合值记录为 `gripper_pos`，来源于 `dist["pos"][0]`。

示例：

```bash
python collect_gripper.py --arm-host 172.25.5.243 --arm-port 8080
```

常用参数：

- `--disable-gripper`：关闭夹爪状态采集。

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
2. 再用 `camera_data.py` 检查 DJI 和 RealSense 是否正常出图。
3. 使用 `pressure_data.py` 或 `herong_9_pressure_data.py` 验证压力通信。
4. 最后按需要运行 `collect_robot.py`、`collect_pressure.py` 或 `collect_gripper.py` 开始正式采集。

## 采集注意事项

- 请确认 DJI 摄像头、RealSense 和机械臂的 IP / 端口配置正确。
- 采集前建议先检查相机索引，避免打开错误设备。
- 压力脚本依赖下位机发送符合 `<Q64h>` 格式的数据包。
- `sessions/`、`pose_logs/`、`pressure_logs/`、`sample_output/` 等目录属于运行输出，通常不需要提交到 Git。

## 许可证与说明

本仓库包含机械臂二次开发相关代码和数据采集脚本，适合实验环境下的数据同步采集与调试。
如果你希望，我还可以继续帮你补一版 **英文 README** 或者把这些脚本整理成一个更清晰的命令行入口说明。

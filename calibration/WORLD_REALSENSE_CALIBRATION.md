# World RealSense 标定方案

## 目标

双 RealSense 采集后，`fuser/camera.py` 里的 `world_camera` 不能继续使用旧 DJI 参数。新的标定目标是求固定 world RealSense 相机在机器人基座坐标系下的位姿：

```text
T_base_cam = [R_base_cam | t_base_cam]
```

`fuser` 当前约定：

- `CameraParams.R` = `R_base_cam`，即相机坐标轴在机器人基座/world 坐标系中的方向。
- `CameraParams.t` = `t_base_cam`，即相机原点在机器人基座/world 坐标系中的位置。
- 投影时会使用 `R.T` 和 `-R.T @ t` 得到 world-to-camera。

本项目统一 RealSense 标准 profile：

- RGB/depth profile：`848x480@30`
- 内参 `K`：用 RealSense SDK 在上述 profile 下读取
- `fuser` 的 `image_size_orig`：`(480, 848)`
- 深度图：对齐到 RGB/color 像素后保存为 `uint16` PNG，单位毫米

## 两种可用布置

world RealSense 必须固定在最终采集位置，不要再移动。棋盘格不一定要装在末端，但每张图都必须知道棋盘格在机器人基座坐标系下的位姿。

### 方案 A：棋盘格装在末端（推荐）

1. 棋盘格刚性安装到末端执行器或夹爪上。
2. 测量 `T_ee_target`：棋盘格 target 坐标系在末端执行器坐标系中的位姿。
3. 移动机械臂采集 15-30 个姿态，覆盖不同距离、图像区域和角度。

优点是每一帧的 `T_base_target = T_base_ee @ T_ee_target` 由机器人状态自动给出，可以采很多不同姿态，精度更稳定。

### 方案 B：棋盘格固定在基座坐标下的已知位置

1. 把棋盘格固定在桌面/工装上，不要移动。
2. 用尺、工装或已知夹具测出 `T_base_target`。
3. 采集一组图像，求 `T_base_cam = T_base_target @ inv(T_cam_target)`。

这种方式不需要连接机械臂，可以用 `capture --no-robot`。但如果相机和棋盘格都固定，图像视角变化有限，结果更依赖你对 `T_base_target` 的测量精度。

不能直接手持棋盘格随便移动然后求相机相对基座的位置，除非你同时记录每一帧棋盘格的 `T_base_target`。否则 PnP 只能得到 `T_cam_target`，缺少和机器人基座坐标的连接，绝对外参不可观。

棋盘格 target 坐标系由代码定义为：

- 原点：棋盘格第一个内角点。
- X/Y：沿棋盘格内角点行列方向。
- Z：棋盘格平面法向，由 OpenCV PnP 输出决定。

## 使用流程

列出 RealSense：

```bash
python calibration/world_realsense_calibration.py list
```

方案 A 采集样本：

```bash
python calibration/world_realsense_calibration.py capture \
  --serial <WORLD_REALSENSE_SERIAL> \
  --arm-host 172.25.5.243 \
  --arm-port 8080 \
  --width 848 \
  --height 480
```

方案 B 采集样本：

```bash
python calibration/world_realsense_calibration.py capture \
  --serial <WORLD_REALSENSE_SERIAL> \
  --no-robot \
  --width 848 \
  --height 480
```

界面按键：

- `SPACE`：当前棋盘格检测成功时保存一帧。
- `S`：保存 JSON，不退出。
- `Q` / `ESC`：退出。

方案 A 求解外参。若 `T_ee_target` 近似为单位阵：

```bash
python calibration/world_realsense_calibration.py solve \
  --data calibration/world_realsense_calibration.json \
  --reference ee
```

若已测得棋盘格相对末端的平移和 RPY，单位分别为米和度：

```bash
python calibration/world_realsense_calibration.py solve \
  --data calibration/world_realsense_calibration.json \
  --reference ee \
  --ee-target-xyz 0.032 0.000 0.085 \
  --ee-target-rpy-deg 0 0 0
```

也可以把 4x4 矩阵写入 JSON：

```json
{
  "T_ee_target": [
    [1, 0, 0, 0.032],
    [0, 1, 0, 0.000],
    [0, 0, 1, 0.085],
    [0, 0, 0, 1]
  ]
}
```

然后运行：

```bash
python calibration/world_realsense_calibration.py solve \
  --reference ee \
  --ee-target-json calibration/ee_to_checkerboard.json
```

方案 B 求解外参，直接给棋盘格在基座坐标系下的位姿：

```bash
python calibration/world_realsense_calibration.py solve \
  --reference base \
  --base-target-xyz 0.420 0.180 0.030 \
  --base-target-rpy-deg 0 0 90
```

或者使用 JSON：

```json
{
  "T_base_target": [
    [0, -1, 0, 0.420],
    [1,  0, 0, 0.180],
    [0,  0, 1, 0.030],
    [0,  0, 0, 1]
  ]
}
```

```bash
python calibration/world_realsense_calibration.py solve \
  --reference base \
  --base-target-json calibration/base_to_checkerboard.json
```

脚本会输出：

- `calibration/world_realsense_extrinsics.json`
- 可直接粘贴到 `fuser/camera.py` 的 `"world_camera": CameraParams(...)` 片段

## 质量标准

建议采集时满足：

- 棋盘格覆盖图像中心、四角、近距离和远距离。
- 机械臂姿态旋转轴不要都接近平行。
- PnP 平均重投影误差最好低于 `0.5 px`。
- 求解后的平均平移误差最好低于 `10 mm`，平均旋转误差低于 `2 deg`。

如果误差偏大，优先检查：

- `T_ee_target` 是否测量方向正确。
- 方案 B 的 `T_base_target` 是否测量方向正确。
- 棋盘格尺寸 `--square-size` 是否与实物一致。
- world RealSense 在采集样本后是否移动过。
- 是否有模糊、反光或棋盘格只出现在画面很小区域的样本。

# 数据来源、标注、配置与归一化位置总览

本文档描述当前 TransVTLA 数据从原始 session 到标注 manifest，再到 RLDS/TFDS 的完整
来源和文件位置。统计基于 2026-07-13 的仓库状态。

## 1. 一句话入口

| 目的 | 推荐入口 |
| --- | --- |
| 查全部 2112 条最终标注 | `dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/annotation_manifest.csv` |
| 查单个 episode 的完整标注 | `dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/session_annotations/` |
| 查 855 条真机任务划分 | `dataset/task_splits/task_session_manifest.csv` |
| 查真机原始数据 | `dataset/sessions/` 和 `dataset/phase2_realdata_sessions/sessions/` |
| 查仿真原始数据 | `dataset/franka_sim_10k_70f_848x480/shards/` |
| 执行标注到 RLDS 转换 | `scripts/build_annotated_rlds.py` |
| 查当前真机 RLDS | `dataset/rlds_real1000_strict/transvtla_annotated_rlds/` |
| 查当前仿真 RLDS | `dataset/rlds_sim_gripper01_strict/transvtla_annotated_rlds/` |
| 查夹爪归一化规则 | `scripts/normalize_gripper_tfrecords.py` 和各 RLDS 版本下的 `gripper_normalization_summary.json` |

## 2. 数据总量和来源

最终标注 manifest 共 2112 条：

| 数据源 | 原始 episode/session | 原始根目录 |
| --- | ---: | --- |
| 真机 | 855 | `dataset/sessions/`：549；`dataset/phase2_realdata_sessions/sessions/`：306 |
| 仿真 | 1257 | `dataset/franka_sim_10k_70f_848x480/` |
| **总计** | **2112** |  |

仿真数据来自：

```text
dataset/franka_sim_10k_70f_848x480/shards/
  franka_20env_10000env_realsense848_70f_shard00_gpu0/
    session_ep*_env*/
```

当前原始仿真有 1257 条，但现有 strict sim RLDS 中是 1256 episodes；已知
`session_ep428_env1.BAD_20260624/world_camera/depth` 曾出现 I/O error，应视为 1 条未进入
strict sim RLDS 的首要排查对象。

## 3. 真机数据有多少天

855 条真机数据跨 **7 个实际采集日**：

| 采集日期 | Sessions | 主要任务 |
| --- | ---: | --- |
| 2026-05-26 | 59 | task1=39，task4=17，unknown=3 |
| 2026-06-03 | 306 | task1=306 |
| 2026-06-05 | 6 | task2=6 |
| 2026-06-08 | 8 | task2=8 |
| 2026-06-10 | 7 | task2=7 |
| 2026-06-15 | 211 | task1=99，task3=45，task4=67 |
| 2026-06-16 | 258 | task2=163，task3=44，task4=51 |
| **总计** | **855** |  |

## 4. 真机原始格式

| 数据格式/硬件 split | Sessions | 原始位置 | 主要视觉数据 |
| --- | ---: | --- | --- |
| `dual_realsense` | 469 | 主要在 `dataset/sessions/` | world + wrist 双 RealSense RGB-D |
| `legacy_dji_realsense` | 327 | 主要在 `dataset/phase2_realdata_sessions/sessions/` | DJI world RGB/伪深度 + wrist RealSense RGB-D |
| `legacy_dji_realsense_rgb_only` | 59 | `dataset/sessions/session_20260526_*` | DJI RGB-only |

进一步按相机位置划分：

| Split | Sessions |
| --- | ---: |
| `dual_realsense_repositioned` | 417 |
| `dual_realsense_initial_position_begin_end` | 52 |
| `legacy_dji_realsense_rgbd` | 327 |
| `legacy_dji_realsense_rgb_only` | 59 |

标准双 RealSense session 的核心文件：

```text
world_camera/rgb/          # 848x480 JPEG
world_camera/depth/        # 848x480 uint16 PNG，毫米
wrist_camera/rgb/
wrist_camera/depth/
camera_metadata.json       # 相机内参、depth scale、序列号等
frames.csv                 # 视觉主时间线
aligned_timesteps.csv      # robot/gripper/pressure 对齐结果
pressure/pressure.csv      # 20 通道触觉
preprocessed_pressure/*.npz
robot_state/robot_state.csv
robot_state/gripper_state.csv
```

旧 DJI 数据使用 `sync/sync_index.csv` 做跨模态对齐。`dji_depth/` 是 Depth Anything V2
生成的 per-image relative depth，不是标定后的 metric depth。

## 5. 任务划分在哪里

任务人工复核与范围标注：

```text
dataset/task_review/
  task_classification_report.md
  task_range_annotations.csv
  previews/preview_manifest.csv
```

最终 session 级任务划分：

```text
dataset/task_splits/
  task_session_manifest.csv
  task_split_summary.json
  by_task/<task_id>/manifest.csv
  by_hardware_split/<split>/<task_id>/manifest.csv
```

任务数量：

| Task ID | Sessions |
| --- | ---: |
| `task1_simple_put_object_in_box` | 444 |
| `task2_complex_put_object_or_beaker_tripod` | 184 |
| `task3_simple_beaker_tripod` | 89 |
| `task4_test_tube_rack_insertion` | 135 |
| `unknown_or_mixed` | 3 |

## 6. 最终标注在哪里

推荐且已合并 task4 人工确认结果的 canonical 标注：

```text
dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/
  annotation_manifest.csv                 # 2112 条统一入口
  annotation_summary.json
  session_annotations/*.json              # 每个 episode 一个完整 sidecar
  validation/annotation_manifest_consistency_checked.csv
  real_1000_train_view_strict/manifest.csv
  real_1000_train_view_with_review/manifest.csv
```

task4 单独确认表：

```text
docs/task4_slot_annotations_confirmed.csv
```

task4 合并后的指令分布：

| Slot pair | Sessions | Language instruction |
| --- | ---: | --- |
| `1-2` | 68 | `Pick up the test tube from slot 1 and insert it into slot 2 of the test tube rack.` |
| `1-3` | 23 | `Pick up the test tube from slot 1 and insert it into slot 3 of the test tube rack.` |
| `1-5` | 27 | `Pick up the test tube from slot 1 and insert it into slot 5 of the test tube rack.` |
| `2-3` | 17 | `Pick up the test tube from slot 2 and insert it into slot 3 of the test tube rack.` |

合并前备份：

```text
dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/backups/
  pre_task4_slot_merge_20260713.tar.gz
```

## 7. 标注是如何生成的

标注主程序是 `scripts/auto_annotate_dataset.py`，主要输入和输出关系如下：

```text
task_session_manifest.csv
        +
Qwen3-VL vision results / consistency review / third-pass overrides
        +
pressure.csv 接触阶段估计
        ↓
annotation_manifest.csv + session_annotations/*.json
```

标注字段包括：任务、语言指令、物体类别和属性、slot pair、阶段分段、接触帧、视觉复核、
质量状态等。task4 的 135 条 slot 指令已同时写入 canonical CSV、派生 manifest 和单 session
JSON，方便任何入口得到一致标签。

## 8. RLDS 转换配置在哪里

基础转换器：

```text
scripts/build_annotated_rlds.py
```

它的配置来自命令行参数，没有单独的 YAML 配置文件。主要参数：

| 参数 | 默认/作用 |
| --- | --- |
| `--manifest` | 必填，指定标注 manifest |
| `--output-dir` | 默认 `dataset/rlds_annotated` |
| `--data-source` | `all/real/sim` |
| `--allow-review` | 是否保留 `needs_review=1` |
| `--include-unusable` | 是否保留不可做 aligned training 的记录 |
| `--image-size` | 默认 224 |
| `--no-mask-tactile-before-stage1` | 是否取消 stage1 前的触觉置零 |
| `--version` | 默认 `3.0.0` |
| `--dry-run` / `--build` | 校验或正式构建 |

转换逻辑：

- 一个 session 生成一个 episode。
- 一个共同视觉帧生成一个 step。
- RGB 缩放为 `224x224`。
- depth 使用最近邻缩放为 `224x224x1 uint16`。
- `state = pose_1..pose_6 + gripper_position`。
- `action[:6] = next_pose - current_pose`。
- 基础构建中的 `action[6] = next_gripper - current_gripper`；后续夹爪归一化重写会将其改为下一帧归一化夹爪位置。
- tactile 来自 `preprocessed_pressure/*.npz`，shape 为 `(16,20)`；默认 stage1 前置零。
- 语言和 slot metadata 直接来自传入的 manifest。

相机原始配置在每个 session 的 `camera_metadata.json`。3.4.0 点云候选还引用外部配置：

```text
/media/files/Test_TransVTLA/configs/camera_groups/
  transvtla_world_camera_groups_formal_metric_groups_20260710.json
```

对应候选 manifest 和报告在：

```text
/media/files/Test_TransVTLA/reports/
  rlds_3_4_2_candidate_manifest_formal_metric_groups_20260710.csv
  rlds_3_4_0_groupfix_build_20260711.json
  rlds_3_4_0_groupfix_build_20260711.md
```

## 9. 归一化在哪里

### 9.1 夹爪归一化

脚本：

```text
scripts/normalize_gripper_tfrecords.py
scripts/normalize_gripper_rlds.py
```

当前规则：

```text
observation.state[6] = clip(raw_gripper_position / 1000, 0, 1)
action[6] = 下一视觉帧的 normalized gripper position
terminal action[6] = 当前/最后一帧 normalized gripper position
```

注意：归一化后的 `action[6]` 是下一帧位置，不再是基础构建器产生的 gripper delta。

实际运行参数和 min/max 在：

```text
dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0/
  gripper_normalization_summary.json

dataset/rlds_sim_gripper01_strict/transvtla_annotated_rlds/3.0.0/
  gripper_normalization_summary.json
```

真机：1000 examples、318421 steps、scale=1000、归一化范围 `[0,1]`。

仿真：1256 examples、87920 steps、scale=1000；当前原始夹爪值约为 280，因此归一化值约为
0.28。

### 9.2 Robot pose/action

基础 RLDS 中 `pose_1..pose_6` 保持原始物理尺度，没有在转换器中做 dataset mean/std 或
quantile 归一化。`action[:6]` 是相邻视觉帧 pose 差分。模型训练侧若做统计归一化，应使用
TFDS 目录下的 `dataset_statistics_*.json`，不要把它与数据生成时的夹爪除 1000 混为一谈。

### 9.3 触觉

- 原始数据在 `pressure/pressure.csv`，已清洗为 20 个通道。
- RLDS 读取 `preprocessed_pressure/*.npz` 的 `(16,20)` float32 窗口。
- 基础转换器不在加载时再做全局 mean/std；默认只在 stage1 前将 tactile 置零。
- contact-memory 使用原始 pressure CSV 单独计算 baseline：预接触/前 50 行中位数，MAD
  估计噪声，压力变化为 `max(baseline_adc - current_adc, 0)`。
- contact-memory 的实际参数写在相应版本目录的 `contact_memory_generation_summary.json`。

### 9.4 图像和深度

- RGB 只做 resize 到 `224x224`，存为 uint8；模型侧归一化不在 RLDS 转换脚本内。
- RealSense depth 保持 uint16 毫米值，resize 使用 nearest-neighbor。
- 缺失 depth 会填零。
- DJI pseudo depth 是相对深度，不应进入严格 metric 点云训练。

## 10. 当前 RLDS 版本和来源

| RLDS | Episodes | Shards | 原始来源/manifest |
| --- | ---: | ---: | --- |
| `dataset/rlds_real1000_strict/.../3.0.0` | 1000 | 512 | `real_1000_train_view_strict/manifest.csv`，由 456 个 strict 真机 session 重采样到 1000 |
| `dataset/rlds_sim_strict/.../3.0.0` | 1256 | 256 | 1257 条仿真原始 session 中成功进入 strict 的 1256 条 |
| `dataset/rlds_sim_gripper01_strict/.../3.0.0` | 1256 | 256 | `rlds_sim_strict` 经夹爪除 1000 重写 |

真机 3.0.0 后续增强链：

```text
3.0.0 基础标注 RLDS + gripper normalization
  → 3.1.0 point_cloud / next_point_cloud
  → 3.2.0 contact memory
  → 3.3.0 candidate: gripper_xyz + current_timestep
  → 3.4.0 candidate: camera group 修正与 metric LC 筛选
```

3.4.0 candidate 共 1000 examples、318421 steps；其中 898 examples 被列为 metric LC
candidate，102 examples 因负漂移/待人工确认而排除。该版本状态仍是 candidate，不是严格标定
release，也未标记为可直接真机 rollout。

## 11. 重要使用提醒

1. canonical 标注已包含最新 task4 指令，但**已经构建好的 RLDS TFRecord 不会自动改变**。
   若训练需要新 task4 语言监督，必须从更新后的 manifest 重新构建或重写 TFRecord 文本字段。
2. `real_1000_train_view_strict` 是 456 个基础真机 session 重采样成 1000 episodes，不是 1000
   个唯一真实采集 session。
3. 全部唯一真机数据是 855 条；全部唯一 real+sim 标注是 2112 条。
4. `needs_review` 与 `usable_for_aligned_training` 是不同概念，筛选时不要混用。
5. DJI pseudo depth 只能作为相对深度/视觉辅助，不能当 RealSense metric depth 使用。


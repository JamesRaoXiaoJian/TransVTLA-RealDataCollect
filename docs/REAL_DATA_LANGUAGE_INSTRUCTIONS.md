# 真实数据语言指令统计

本文档汇总
`dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/annotation_manifest.csv`
中的 855 条真实数据语言指令。

## 汇总

| 任务 | Task ID | 数据量 | 指令种类数 | 当前状态 |
| --- | --- | ---: | ---: | --- |
| 任务 1：简单场景放入盒子 | `task1_simple_put_object_in_box` | 444 | 5 | 已具体到物体类别 |
| 任务 2：复杂场景 | `task2_complex_put_object_or_beaker_tripod` | 184 | 5 | 烧杯目标位置需要复核 |
| 任务 3：简单烧杯放三脚架 | `task3_simple_beaker_tripod` | 89 | 1 | 目标动作明确 |
| 任务 4：试管插入试管架 | `task4_test_tube_rack_insertion` | 135 | 1 | 插孔位置需要复核 |
| 未确定 | `unknown_or_mixed` | 3 | 1（空指令） | 需要人工确认 |
| **总计** |  | **855** |  |  |

## 任务 1：简单场景放入盒子

Task ID：`task1_simple_put_object_in_box`

共 444 条数据、5 种指令。

| 数量 | `language_instruction` |
| ---: | --- |
| 218 | `Pick up the small hollow transparent cube and place it into the box.` |
| 105 | `Pick up the big transparent cube and place it into the box.` |
| 73 | `Pick up the dropper bottle and place it into the box.` |
| 32 | `Pick up the cola cup and place it into the box.` |
| 16 | `Pick up the large beaker and place it into the box.` |

该任务的语言指令已经具体到物体类别。

## 任务 2：复杂场景

Task ID：`task2_complex_put_object_or_beaker_tripod`

共 184 条数据、5 种指令。

| 数量 | `language_instruction` |
| ---: | --- |
| 69 | `Find the large beaker in the current environment, pick it up, and place it into the box.` |
| 63 | `Find the big transparent cube in the current environment, pick it up, and place it into the box.` |
| 35 | `Find the small hollow transparent cube in the current environment, pick it up, and place it into the box.` |
| 15 | `Find the small beaker in the current environment, pick it up, and place it into the box.` |
| 2 | `Find the dropper bottle in the current environment, pick it up, and place it into the box.` |

其中有 84 条烧杯数据：

| 物体类别 | 数量 | 当前目标位置 |
| --- | ---: | --- |
| Large beaker | 69 | Box |
| Small beaker | 15 | Box |
| **总计** | **84** |  |

这 84 条烧杯数据目前全部写成放进盒子，没有任何一条写成放到三脚架。
`scripts/auto_annotate_dataset.py` 中的任务 2 模板会统一生成
`place it into the box`。构建最终 RLDS 前，应复核这些数据的真实目标位置。

## 任务 3：简单烧杯放三脚架

Task ID：`task3_simple_beaker_tripod`

共 89 条数据、1 种指令。

| 数量 | `language_instruction` |
| ---: | --- |
| 89 | `Pick up the beaker and place it on the tripod.` |

该指令没有区分烧杯大小，但目标动作明确为放到三脚架。

## 任务 4：试管插入试管架

Task ID：`task4_test_tube_rack_insertion`

原始 manifest 共 135 条数据、1 种指令，全部被标记为 `1-2`。

| 数量 | `slot_pair` | `language_instruction` |
| ---: | --- | --- |
| 135 | `1-2` | `Pick up the test tube from slot 1 and insert it into slot 2 of the test tube rack.` |

标注程序支持 `1-2`、`1-3`、`1-5` 和 `2-3` 四种插孔组合，但当前
135 条数据全部标记为 `1-2`。构建最终 RLDS 前，应根据实际视频复核每条数据的
源插孔和目标插孔。

### 多 Agent 视觉复核与采集者确认结果

三个独立分片对全部 135 个 session 逐条查看了动作前、阶段起点、接触和末段图像。
场景、背景和物理起终孔模式显示数据由四个连续采集配置组成：

| 全局索引 | 数据量 | 场景配置候选 | 候选指令 |
| --- | ---: | --- | --- |
| 1–68 | 68 | `1-2` | `Pick up the test tube from slot 1 and insert it into slot 2 of the test tube rack.` |
| 69–91 | 23 | `1-3` | `Pick up the test tube from slot 1 and insert it into slot 3 of the test tube rack.` |
| 92–118 | 27 | `1-5` | `Pick up the test tube from slot 1 and insert it into slot 5 of the test tube rack.` |
| 119–135 | 17 | `2-3` | `Pick up the test tube from slot 2 and insert it into slot 3 of the test tube rack.` |
| **总计** | **135** |  |  |

完整逐 session 确认结果保存在 `docs/task4_slot_annotations_confirmed.csv`。图像验证了每组内
场景配置和物理起终孔模式一致；采集者于 2026-07-13 确认四个配置依次对应 `1-2`、
`1-3`、`1-5`、`2-3`。结果已经合并到 canonical manifest、派生训练 manifest 和单 session
annotation JSON，可以用于后续 RLDS 重建。

## 未确定数据

Task ID：`unknown_or_mixed`

共 3 条数据，`language_instruction` 均为空。这些数据需要人工确认任务类型并补充
具体指令。

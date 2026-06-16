# 数据修复计划

> 基于 2026-06-10 全量审计结果，针对 306 个 session 的已有数据制定修复方案。

---

## 1. 修复项总览

| 编号 | 修复项 | 影响范围 | 优先级 | 状态 |
|------|--------|---------|--------|------|
| F1 | Pressure 时间戳排序 | 19 个 session | P0 | 待执行 |
| F2 | Gripper 空值填充 | 2 个 session | P1 | 待执行 |
| F3 | Gripper 时间戳去重 | 110/306 session (82,842行重复) | P0 | 待执行 |
| F4 | Pressure ADC 异常值裁剪 | 19 个 session | P2 | 待执行 |
| F5 | 百度网盘残留文件清理 | 2 个 session | P2 | 待执行 |
| F6 | 压力数据预处理（生成 .npz） | 全部 306 个 session | P1 | 待执行 |
| F7 | 频率不匹配的标注与文档更新 | 全局 | P0 | 待执行 |

---

## 2. 各修复项详细方案

### F1: Pressure 时间戳排序（P0）

**问题**：19 个 session 的 `pressure.csv` 中 UDP 包到达顺序不一致，导致时间戳非单调递增。

**影响 session**：
```
session_20260603_192900   (1 次反转)
session_20260603_193800  (14 次反转)
session_20260603_193938   (1 次反转)
session_20260603_200021   (1 次反转)
session_20260603_200101   (1 次反转)
session_20260603_200912  (24 次反转)
session_20260603_201918  (30 次反转)
session_20260603_202327   (1 次反转)
session_20260603_205025  (25 次反转)
session_20260603_205139   (1 次反转)
session_20260603_205508   (1 次反转)
session_20260603_210050   (1 次反转)
session_20260603_211039   (1 次反转)
session_20260603_211226   (1 次反转)
session_20260603_211552   (1 次反转)
session_20260603_212954   (1 次反转)
session_20260603_213514   (1 次反转)
session_20260603_214151   (1 次反转)
session_20260603_215152   (1 次反转)
```

**修复方案**：
1. 读取 `pressure.csv`，保留 header
2. 按 `timestamp_us` 列升序排序所有数据行
3. 写回原文件（覆盖）
4. 验证排序后时间戳单调递增

**修复脚本**：`scripts/fix_f1_pressure_sort.py`（见下方）

**风险**：排序会改变行顺序，但不改变数据内容。对于 UDP 数据，到达顺序本就无物理意义，排序后更符合实际时间线。

---

### F2: Gripper 空值填充（P1）

**问题**：2 个 session 的 `gripper_state.csv` 中 `sys_state` 列有空字符串。

**影响 session**：
- `session_20260603_202005`：row 82、row 125 的 `sys_state` 为空
- `session_20260603_205508`：row 645、row 649 的 `sys_state` 为空

**修复方案**：
1. 对 `sys_state` 列的空值，用前一个非空值填充（forward fill）
2. 如果是第一行就为空，用后一个非空值填充（backward fill）

**风险**：极低。`sys_state` 为辅助信息列，不影响核心位姿/触觉数据。

---

### F3: Gripper 时间戳去重（P1）

**问题**：110/306 个 session 的 `gripper_state.csv` 存在大量重复时间戳行（同一微秒戳写入多次）。去重后数据量平均减少 ~60%。

**影响规模**：
- 110 个 session 受影响
- 总计 82,842 行重复数据需要删除
- 典型例子：`session_20260603_202235` 从 2004 行去重到 686 行（减少 66%）

**根因**：`GripperStateCollector` 的 `_poll_loop()` 中 `rm_get_rm_plus_state_info()` 调用返回后，时间戳已过，下一次循环生成了相同的时间戳。采集线程的竞争导致同一批数据被多次写入 buffer。

**修复方案**：
1. 读取 `gripper_state.csv`
2. 按 `timestamp_us` 去重，保留每组重复的第一行
3. 写回原文件

**风险**：低。重复行本身是采集线程竞争导致的冗余数据，去重后保留了每个时间戳的完整信息。

---

### F4: Pressure ADC 异常值裁剪（P2）

**问题**：19 个 session 中部分通道出现超出 [0, 5007] 范围的 ADC 值（负值如 -7500，或超上限如 8125）。

**异常通道分布**：集中在 CH42-CH48，均不在有效 20 通道映射中（有效通道为 CH1,9,10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31,32）。

**修复方案**：
- **方案 A（推荐）**：不修复原始 CSV，在预处理阶段（`preprocess_pressure.py`）只提取有效 20 通道，异常通道自然被过滤
- **方案 B**：将超出 [0, 5007] 的值裁剪到范围内（`np.clip`）

**建议**：采用方案 A，保持原始数据不变。

---

### F5: 百度网盘残留文件清理（P2）

**问题**：2 个 session 中有 100 个 `.baiduyun.uploading.cfg` 文件。

**影响 session**：
- `session_20260603_192201`：91 个（主要在 dji/ 目录）
- `session_20260603_192130`：9 个

**修复命令**：
```bash
find /media/files/dataset/TransVTLA-RealDataCollect/dataset/phase2_realdata_sessions/sessions \
  -name "*.baiduyun.uploading.cfg" -delete
```

**风险**：零。这些是同步工具的临时配置文件，与数据无关。

---

### F6: 压力数据预处理（P1）

**问题**：全部 306 个 session 尚未运行 `preprocess_pressure.py` 生成 `.npz` 文件。

**修复方案**：
```bash
python preprocess_pressure.py \
  --source-root dataset/phase2_realdata_sessions/sessions \
  --output-root dataset/phase2_realdata_sessions/sessions
```

**处理流程**（每个 session）：
1. 读取标准 `pressure/pressure.csv`（20 个有效通道）
2. 按 `channel_mapping.json` 校验通道顺序
3. 动态基线去除（前 50 行均值）
4. 归一化（÷3500，clip 到 [0,1]）
5. 滑动窗口（size=16, stride=1）→ shape `(N, 16, 20)`
6. 保存为 `preprocessed_pressure/pressure.npz`

**前置条件**：先完成 F1（时间戳排序）。

---

### F7: 频率标注与文档更新（P0）

**问题**：代码和文档中的目标频率与实际不符。

| 模态 | 文档/代码标注 | 实际值 | 修正 |
|------|-------------|--------|------|
| Pressure | 200 Hz | ~140 Hz | 更新为 140 Hz |
| Robot state | 200 Hz（代码 100Hz） | ~16 Hz | 更新为 16 Hz |
| Gripper state | 200 Hz | ~19 Hz | 更新为 19 Hz |
| DJI camera | 20 Hz | ~8 fps | 更新为 8 fps |
| RealSense | 20 Hz | ~8 fps | 更新为 8 fps |

**修改文件**：
- `data_description.md`：更新频率表格
- `README.md`：更新频率说明
- `collectors/robot_arm.py`：注释更新
- `collectors/gripper_state.py`：注释更新

---

## 3. 修复执行顺序

```
Step 1: F7  ── 更新文档（无数据变更，立即可做）
Step 2: F5  ── 清理残留文件（一行命令）
Step 3: F1  ── Pressure 时间戳排序（19 个 session）
Step 4: F3  ── Gripper 时间戳去重（110 个 session，82,842 行重复）  ← 最大修复量
Step 5: F2  ── Gripper 空值填充（2 个 session，4 个空值）
Step 6: F6  ── 压力数据预处理（依赖 F1 完成）
         F4  ── ADC 裁剪（可选，预处理已过滤异常通道）
```

---

## 4. 修复脚本

### `scripts/fix_f1_pressure_sort.py`

```python
#!/usr/bin/env python3
"""F1: Sort pressure.csv by timestamp_us for sessions with non-monotonic timestamps."""

import csv
import os
from pathlib import Path

SESSIONS_ROOT = Path("dataset/phase2_realdata_sessions/sessions")

AFFECTED_SESSIONS = [
    "session_20260603_192900", "session_20260603_193800", "session_20260603_193938",
    "session_20260603_200021", "session_20260603_200101", "session_20260603_200912",
    "session_20260603_201918", "session_20260603_202327", "session_20260603_205025",
    "session_20260603_205139", "session_20260603_205508", "session_20260603_210050",
    "session_20260603_211039", "session_20260603_211226", "session_20260603_211552",
    "session_20260603_212954", "session_20260603_213514", "session_20260603_214151",
    "session_20260603_215152",
]

def fix_pressure_sort(session_name: str) -> bool:
    csv_path = SESSIONS_ROOT / session_name / "pressure" / "pressure.csv"
    if not csv_path.exists():
        print(f"  SKIP {session_name}: file not found")
        return False

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Sort by timestamp_us (first column)
    rows.sort(key=lambda r: int(r[0]))

    # Verify monotonic
    for i in range(len(rows) - 1):
        if int(rows[i + 1][0]) < int(rows[i][0]):
            print(f"  FAIL {session_name}: still non-monotonic after sort")
            return False

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  OK   {session_name}: sorted {len(rows)} rows")
    return True

if __name__ == "__main__":
    print("F1: Sorting pressure.csv timestamps")
    fixed = 0
    for s in AFFECTED_SESSIONS:
        if fix_pressure_sort(s):
            fixed += 1
    print(f"\nDone: {fixed}/{len(AFFECTED_SESSIONS)} sessions fixed")
```

### `scripts/fix_f2_f3_gripper.py`

```python
#!/usr/bin/env python3
"""F2+F3: Fix gripper_state.csv — deduplicate and fill empty sys_state."""

import csv
import os
from pathlib import Path

SESSIONS_ROOT = Path("dataset/phase2_realdata_sessions/sessions")

def fix_gripper(session_path: Path) -> dict:
    csv_path = session_path / "robot_state" / "gripper_state.csv"
    if not csv_path.exists():
        return {"skipped": True}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    original_count = len(rows)
    fixes = {"deduped": 0, "filled_empty": 0}

    # F3: Deduplicate by timestamp_us
    seen_ts = set()
    deduped = []
    for row in rows:
        ts = row[0]
        if ts in seen_ts:
            fixes["deduped"] += 1
            continue
        seen_ts.add(ts)
        deduped.append(row)
    rows = deduped

    # F2: Fill empty sys_state (column index 4)
    sys_state_idx = header.index("sys_state") if "sys_state" in header else 4
    last_valid = ""
    for row in rows:
        if row[sys_state_idx] == "":
            if last_valid:
                row[sys_state_idx] = last_valid
                fixes["filled_empty"] += 1
        else:
            last_valid = row[sys_state_idx]

    # Backward fill if first rows are still empty
    for row in reversed(rows):
        if row[sys_state_idx] == "":
            row[sys_state_idx] = last_valid
        else:
            last_valid = row[sys_state_idx]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return {"original": original_count, "final": len(rows), **fixes}

if __name__ == "__main__":
    print("F2+F3: Fixing gripper_state.csv")
    for sdir in sorted(SESSIONS_ROOT.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("session_"):
            continue
        result = fix_gripper(sdir)
        if result.get("deduped", 0) > 0 or result.get("filled_empty", 0) > 0:
            print(f"  {sdir.name}: deduped={result['deduped']}, filled={result['filled_empty']}, "
                  f"{result['original']}→{result['final']} rows")
    print("\nDone")
```

### `scripts/fix_f5_cleanup.sh`

```bash
#!/bin/bash
# F5: Remove Baidu Yun sync config files
find dataset/phase2_realdata_sessions/sessions \
  -name "*.baiduyun.uploading.cfg" -type f -print -delete
echo "Cleanup complete."
```

---

## 5. 验证清单

每个修复步骤完成后，运行以下验证：

```bash
# 验证 F1: 无时间戳反转
python3 -c "
import csv, os
root = 'dataset/phase2_realdata_sessions/sessions'
for s in sorted(os.listdir(root)):
    p = os.path.join(root, s, 'pressure', 'pressure.csv')
    if not os.path.isfile(p): continue
    with open(p) as f:
        r = csv.reader(f); next(r)
        ts = [int(row[0]) for row in r]
    for i in range(len(ts)-1):
        if ts[i+1] < ts[i]:
            print(f'FAIL: {s} at row {i}')
            break
else:
    print('PASS: All pressure timestamps monotonic')
"

# 验证 F2+F3: 无空值、无重复
python3 -c "
import csv, os
root = 'dataset/phase2_realdata_sessions/sessions'
for s in sorted(os.listdir(root)):
    p = os.path.join(root, s, 'robot_state', 'gripper_state.csv')
    if not os.path.isfile(p): continue
    with open(p) as f:
        r = csv.reader(f); header = next(r)
        rows = list(r)
    # Check empty
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if v == '':
                print(f'EMPTY: {s} row {i} col {j}')
    # Check dupes
    tss = [row[0] for row in rows]
    if len(tss) != len(set(tss)):
        print(f'DUPE: {s}')
else:
    print('PASS: All gripper_state clean')
"

# 验证 F5: 无残留文件
find dataset/ -name "*.baiduyun.uploading.cfg" | head -5
echo "(should be empty)"
```

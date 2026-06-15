"""Session data quality check."""
import json, csv, statistics
from pathlib import Path

session = Path("d:/realman/sessions/session_20260615_193300")

print("=" * 70)
print("  1. 文件结构")
print("=" * 70)

world_rgb = sorted((session / "world_camera" / "rgb").glob("*.jpg"))
world_depth = sorted((session / "world_camera" / "depth").glob("*.png"))
wrist_rgb = sorted((session / "wrist_camera" / "rgb").glob("*.jpg"))
wrist_depth = sorted((session / "wrist_camera" / "depth").glob("*.png"))

print(f"  world_camera/rgb:   {len(world_rgb)} jpg")
print(f"  world_camera/depth: {len(world_depth)} png")
print(f"  wrist_camera/rgb:   {len(wrist_rgb)} jpg")
print(f"  wrist_camera/depth: {len(wrist_depth)} png")

if world_rgb:
    sizes = [f.stat().st_size for f in world_rgb[:5]]
    print(f"  world RGB sample sizes: {sizes}")
if world_depth:
    sizes = [f.stat().st_size for f in world_depth[:5]]
    print(f"  world depth sample sizes: {sizes}")

def check_continuity(files, label):
    ids = sorted(int(f.stem) for f in files)
    if not ids:
        print(f"  {label}: NO FILES")
        return
    expected = set(range(ids[0], ids[-1] + 1))
    missing = expected - set(ids)
    print(f"  {label}: {ids[0]:04d}~{ids[-1]:04d}, total={len(ids)}, missing={len(missing)}")
    if missing and len(missing) <= 20:
        print(f"    missing frames: {sorted(missing)}")

print("\n  帧连续性:")
check_continuity(world_rgb, "world_rgb")
check_continuity(world_depth, "world_depth")
check_continuity(wrist_rgb, "wrist_rgb")
check_continuity(wrist_depth, "wrist_depth")

# ========== 2. frames.csv ==========
print(f"\n{'=' * 70}")
print("  2. frames.csv")
print("=" * 70)

with open(session / "frames.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    fheader = next(reader)
    frows = list(reader)

ts_col = fheader.index("capture_monotonic_us")
fts = [int(r[ts_col]) for r in frows]
fintervals = [(fts[i+1] - fts[i]) / 1000.0 for i in range(len(fts) - 1)]

print(f"  rows={len(frows)}, duration={(fts[-1]-fts[0])/1e6:.2f}s")
print(f"  mean={statistics.mean(fintervals):.2f}ms ({1000/statistics.mean(fintervals):.1f}Hz)")
print(f"  median={statistics.median(fintervals):.2f}ms, std={statistics.stdev(fintervals):.2f}ms")
print(f"  min={min(fintervals):.2f}ms, max={max(fintervals):.2f}ms")

abnormal = [(i, iv) for i, iv in enumerate(fintervals) if iv > 50 or iv < 10]
if abnormal:
    print(f"  异常间隔 (>50ms/<10ms): {len(abnormal)}")
    for i, iv in abnormal[:10]:
        print(f"    frame {i}: {iv:.2f}ms")

# ========== 3. robot_state.csv ==========
print(f"\n{'=' * 70}")
print("  3. robot_state.csv")
print("=" * 70)

with open(session / "robot_state" / "robot_state.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    rheader = next(reader)
    rrows = list(reader)

rts_col = rheader.index("timestamp_us")
rts = [int(r[rts_col]) for r in rrows]
rintervals = [(rts[i+1] - rts[i]) / 1000.0 for i in range(len(rts) - 1)]

print(f"  rows={len(rrows)}, duration={(rts[-1]-rts[0])/1e6:.2f}s")
print(f"  mean={statistics.mean(rintervals):.2f}ms ({1000/statistics.mean(rintervals):.1f}Hz)")
print(f"  std={statistics.stdev(rintervals):.2f}ms, min={min(rintervals):.2f}ms, max={max(rintervals):.2f}ms")

joint_cols = [i for i, c in enumerate(rheader) if c.startswith("joint_")]
pose_cols = [i for i, c in enumerate(rheader) if c.startswith("pose_")]
print(f"  joints: {[rheader[i] for i in joint_cols]}")
print(f"  poses: {[rheader[i] for i in pose_cols]}")

zero_rows = sum(1 for row in rrows if all(float(row[i]) == 0 for i in joint_cols))
if zero_rows:
    print(f"  WARNING: {zero_rows}/{len(rrows)} rows have all-zero joints")
else:
    print(f"  Joint data: OK (no all-zero rows)")

print(f"  First joints: {[rrows[0][i] for i in joint_cols]}")
print(f"  Last joints:  {[rrows[-1][i] for i in joint_cols]}")

# ========== 4. gripper_state.csv ==========
print(f"\n{'=' * 70}")
print("  4. gripper_state.csv")
print("=" * 70)

with open(session / "robot_state" / "gripper_state.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    gheader = next(reader)
    grows = list(reader)

gts_col = gheader.index("timestamp_us")
gts = [int(r[gts_col]) for r in grows]
gintervals = [(gts[i+1] - gts[i]) / 1000.0 for i in range(len(gts) - 1)]

print(f"  rows={len(grows)}, duration={(gts[-1]-gts[0])/1e6:.2f}s")
print(f"  mean={statistics.mean(gintervals):.2f}ms ({1000/statistics.mean(gintervals):.1f}Hz)")

pos_col = gheader.index("gripper_pos")
positions = []
for row in grows:
    try:
        positions.append(float(row[pos_col]))
    except (ValueError, IndexError):
        pass
if positions:
    print(f"  gripper_pos: [{min(positions):.0f}, {max(positions):.0f}], mean={statistics.mean(positions):.1f}")

# ========== 5. pressure.csv ==========
print(f"\n{'=' * 70}")
print("  5. pressure.csv")
print("=" * 70)

with open(session / "pressure" / "pressure.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    pheader = next(reader)
    prows = list(reader)

pts_col = pheader.index("host_monotonic_us")
pts = [int(r[pts_col]) for r in prows]
pintervals = [(pts[i+1] - pts[i]) / 1000.0 for i in range(len(pts) - 1)]

print(f"  rows={len(prows)}, duration={(pts[-1]-pts[0])/1e6:.2f}s")
print(f"  mean={statistics.mean(pintervals):.2f}ms ({1000/statistics.mean(pintervals):.1f}Hz)")
print(f"  cols={len(pheader)}: {pheader[:6]}...")

# ========== 6. aligned_timesteps.csv ==========
print(f"\n{'=' * 70}")
print("  6. aligned_timesteps.csv")
print("=" * 70)

with open(session / "aligned_timesteps.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    aheader = next(reader)
    arows = list(reader)

print(f"  rows={len(arows)}, cols={len(aheader)}")
print(f"  header: {aheader[:8]}...")

def offset_stats(col_name):
    col = aheader.index(col_name)
    vals = []
    for row in arows:
        try:
            vals.append(abs(float(row[col])))
        except (ValueError, IndexError):
            pass
    if not vals:
        return
    s = sorted(vals)
    n = len(s)
    p95 = s[int(n * 0.95)]
    over10 = sum(1 for v in vals if v > 10)
    print(f"  {col_name}: mean={statistics.mean(vals):.2f}ms, p95={p95:.2f}ms, max={s[-1]:.2f}ms, >10ms={over10}/{n}")

offset_stats("robot_offset_ms")
offset_stats("gripper_offset_ms")
offset_stats("pressure_offset_ms")

# 检查插值值是否合理（非空、非异常）
robot_data_cols = [i for i, c in enumerate(aheader) if c.startswith("robot_joint_")]
if robot_data_cols:
    sample = [arows[len(arows)//2][i] for i in robot_data_cols]
    print(f"  Robot mid-sample joints: {sample}")

# ========== 7. 时间覆盖 ==========
print(f"\n{'=' * 70}")
print("  7. 时间覆盖")
print("=" * 70)

vs, ve = fts[0], fts[-1]
for name, ts_list_sensor in [("Robot", rts), ("Gripper", gts), ("Pressure", pts)]:
    ss, se = ts_list_sensor[0], ts_list_sensor[-1]
    gap_s = (vs - ss) / 1000.0
    gap_e = (se - ve) / 1000.0
    ok = "OK" if (ss <= vs and se >= ve) else "GAP"
    print(f"  {name}: start={gap_s:+.1f}ms, end={gap_e:+.1f}ms [{ok}]")

# ========== 8. camera_metadata.json ==========
print(f"\n{'=' * 70}")
print("  8. camera_metadata.json")
print("=" * 70)

with open(session / "camera_metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

print(f"  schema: {meta.get('schema')}")
print(f"  visual_fps: {meta.get('visual_recording_fps')}")
for cname, cdata in meta.get("cameras", {}).items():
    avail = cdata.get("available")
    serial = cdata.get("serial_number")
    prof = cdata.get("profile", {})
    print(f"  {cname}: available={avail}, serial={serial}, {prof.get('width')}x{prof.get('height')}@{prof.get('fps')}fps")

print(f"\n{'=' * 70}")
print("  DONE")
print("=" * 70)

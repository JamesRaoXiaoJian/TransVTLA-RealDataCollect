"""Analyze old CSV data with baseline-corrected grip force extraction."""
import csv
import numpy as np

rows = []
with open(r"C:\isaac-sim\script\gripper_force_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# --- Step 1: compute baseline from approach phase (phase 0) ---
approach_rows = [r for r in rows if int(r["phase"]) == 0]
f1_fx_base = np.mean([float(r["finger1_fx"]) for r in approach_rows])
f1_fy_base = np.mean([float(r["finger1_fy"]) for r in approach_rows])
f1_fz_base = np.mean([float(r["finger1_fz"]) for r in approach_rows])
f2_fx_base = np.mean([float(r["finger2_fx"]) for r in approach_rows])
f2_fy_base = np.mean([float(r["finger2_fy"]) for r in approach_rows])
f2_fz_base = np.mean([float(r["finger2_fz"]) for r in approach_rows])

print("=" * 70)
print("BASELINE (averaged from approach phase, no contact):")
print(f"  Finger1: fx={f1_fx_base:.3f}  fy={f1_fy_base:.3f}  fz={f1_fz_base:.3f}")
print(f"  Finger2: fx={f2_fx_base:.3f}  fy={f2_fy_base:.3f}  fz={f2_fz_base:.3f}")

# --- Step 2: compute net grip force for all rows ---
for r in rows:
    net_f1_fx = float(r["finger1_fx"]) - f1_fx_base
    net_f2_fx = float(r["finger2_fx"]) - f2_fx_base
    r["net_f1_fx"] = net_f1_fx
    r["net_f2_fx"] = net_f2_fx
    # Grip force = |net fx| (the pinch axis component, baseline removed)
    r["grip_f1"] = abs(net_f1_fx)
    r["grip_f2"] = abs(net_f2_fx)
    r["grip_total"] = r["grip_f1"] + r["grip_f2"]

# --- Step 3: per-phase summary ---
print("\n" + "=" * 70)
print(f"{'Phase':>5} {'Name':>20} {'Steps':>6} | {'Grip Total':>11} | {'F1 |net fx|':>12} | {'F2 |net fx|':>12}")
print(f"{'':>5} {'':>20} {'':>6} | {'mean':>5} {'max':>5} | {'mean':>5} {'max':>5} | {'mean':>5} {'max':>5}")
print("-" * 70)

phases = {}
for r in rows:
    p = int(r["phase"])
    phases.setdefault(p, []).append(r)

for p in sorted(phases.keys()):
    data = phases[p]
    name = data[0]["phase_name"]
    gt = [d["grip_total"] for d in data]
    g1 = [d["grip_f1"] for d in data]
    g2 = [d["grip_f2"] for d in data]
    print(f"{p:>5} {name:>20} {len(data):>6} | {np.mean(gt):>5.1f} {max(gt):>5.1f} | {np.mean(g1):>5.1f} {max(g1):>5.1f} | {np.mean(g2):>5.1f} {max(g2):>5.1f}")

# --- Step 4: detail for grasp-related phases ---
for phase_id, label in [(3, "close_gripper"), (4, "lift")]:
    data = phases.get(phase_id, [])
    if not data:
        continue
    print(f"\n{'=' * 70}")
    print(f"Phase {phase_id} ({label}) — net fx detail (first 5 + last 5):")
    print(f"{'step':>5} {'net_f1_fx':>10} {'net_f2_fx':>10} | {'grip_f1':>8} {'grip_f2':>8} {'total':>8}")
    print("-" * 60)
    samples = data[:5] + data[-5:] if len(data) > 10 else data
    for d in samples:
        print(f"{d['step']:>5} {d['net_f1_fx']:>10.3f} {d['net_f2_fx']:>10.3f} | {d['grip_f1']:>8.3f} {d['grip_f2']:>8.3f} {d['grip_total']:>8.3f}")

# --- Step 5: verify grasp profile makes physical sense ---
print(f"\n{'=' * 70}")
print("GRASP PROFILE SUMMARY (should follow this pattern):")
print("  approach/descend/settle  -> grip_force near 0 (no contact)")
print("  close_gripper            -> grip_force rises sharply")
print("  lift                     -> grip_force holds steady")
print("  move/descend_to_place    -> grip_force holds steady")
print("  open_gripper             -> grip_force drops to ~0")
print()

grasp_start = phases.get(3, [])
grasp_hold = phases.get(5, [])
grasp_release = phases.get(7, [])
if grasp_start and grasp_hold:
    close_max = max(d["grip_total"] for d in grasp_start)
    hold_mean = np.mean([d["grip_total"] for d in grasp_hold])
    release_val = np.mean([d["grip_total"] for d in grasp_release]) if grasp_release else 0
    print(f"  Close phase peak grip:   {close_max:.2f} N")
    print(f"  Carry phase mean grip:   {hold_mean:.2f} N")
    print(f"  Release phase mean grip: {release_val:.2f} N")
    if close_max > 10 and hold_mean < 5 and release_val < 1:
        print("  -> PROFILE CORRECT: grip peaks during close, drops after release")
    else:
        print("  -> NOTE: values may need further calibration depending on object")

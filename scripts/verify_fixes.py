#!/usr/bin/env python3
"""Verify all data fixes have been applied correctly.

Usage:
    python scripts/verify_fixes.py
"""

import csv
import os
from pathlib import Path

SESSIONS_ROOT = Path("dataset/phase2_realdata_sessions/sessions")


def verify_f1_pressure_monotonic():
    """F1: Verify all pressure.csv timestamps are monotonically increasing."""
    print("F1: Checking pressure.csv timestamp monotonicity...")
    failures = []
    total = 0

    for sdir in sorted(SESSIONS_ROOT.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("session_"):
            continue
        total += 1
        csv_path = sdir / "pressure" / "pressure.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            ts = [int(row[0]) for row in reader]

        for i in range(len(ts) - 1):
            if ts[i + 1] < ts[i]:
                failures.append(f"{sdir.name}: reversal at row {i} ({ts[i]} → {ts[i+1]})")
                break

    if failures:
        print(f"  FAIL: {len(failures)}/{total} sessions have non-monotonic timestamps")
        for f in failures[:10]:
            print(f"    {f}")
    else:
        print(f"  PASS: All {total} sessions have monotonically increasing timestamps")
    return len(failures) == 0


def verify_f2_f3_gripper_clean():
    """F2+F3: Verify gripper_state.csv has no empty cells and no duplicate timestamps."""
    print("\nF2+F3: Checking gripper_state.csv for empty cells and duplicates...")
    empty_failures = []
    dupe_failures = []
    total = 0

    for sdir in sorted(SESSIONS_ROOT.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("session_"):
            continue
        total += 1
        csv_path = sdir / "robot_state" / "gripper_state.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Check empty cells — skip gripper_pos when read code indicates failure (code=-2)
        # Empty gripper_pos with code=-2 is a legitimate "read failed" signal
        code_idx = header.index("rm_plus_read_code") if "rm_plus_read_code" in header else 2
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if v == "":
                    # Allow empty gripper fields when read code is -2 (read failure)
                    if j >= 5 and code_idx < len(row) and row[code_idx] == "-2":
                        continue  # legitimate empty
                    empty_failures.append(f"{sdir.name}: row {i} col {j} ({header[j]}) is empty")
                    break
            else:
                continue
            break

        # Check duplicate timestamps
        tss = [row[0] for row in rows]
        if len(tss) != len(set(tss)):
            dupe_failures.append(f"{sdir.name}: {len(tss) - len(set(tss))} duplicate timestamps")

    ok = True
    if empty_failures:
        print(f"  FAIL: {len(empty_failures)}/{total} sessions have empty cells")
        for f in empty_failures[:5]:
            print(f"    {f}")
        ok = False
    else:
        print(f"  PASS: No empty cells in {total} sessions")

    if dupe_failures:
        print(f"  FAIL: {len(dupe_failures)}/{total} sessions have duplicate timestamps")
        for f in dupe_failures[:5]:
            print(f"    {f}")
        ok = False
    else:
        print(f"  PASS: No duplicate timestamps in {total} sessions")

    return ok


def verify_f5_no_baidu_files():
    """F5: Verify no .baiduyun.uploading.cfg files remain."""
    print("\nF5: Checking for Baidu Yun config files...")
    cfg_files = list(SESSIONS_ROOT.rglob("*.baiduyun.uploading.cfg"))
    if cfg_files:
        print(f"  FAIL: {len(cfg_files)} .baiduyun.uploading.cfg files found")
        for f in cfg_files[:5]:
            print(f"    {f}")
        return False
    else:
        print("  PASS: No .baiduyun.uploading.cfg files found")
        return True


def verify_structure_integrity():
    """Verify all sessions have required directories and files."""
    print("\nStructure: Checking session integrity...")
    issues = []
    total = 0

    required_dirs = ["dji", "realsense_rgb", "realsense_depth", "pressure", "robot_state"]
    required_files = [
        ("pressure", "pressure.csv"),
        ("robot_state", "robot_state.csv"),
        ("robot_state", "gripper_state.csv"),
    ]

    for sdir in sorted(SESSIONS_ROOT.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("session_"):
            continue
        total += 1

        for d in required_dirs:
            if not (sdir / d).is_dir():
                issues.append(f"{sdir.name}: missing directory {d}/")

        for subdir, fname in required_files:
            fpath = sdir / subdir / fname
            if not fpath.exists():
                issues.append(f"{sdir.name}: missing {subdir}/{fname}")
            elif fpath.stat().st_size == 0:
                issues.append(f"{sdir.name}: empty {subdir}/{fname}")

    if issues:
        print(f"  FAIL: {len(issues)} issues in {total} sessions")
        for i in issues[:10]:
            print(f"    {i}")
        return False
    else:
        print(f"  PASS: All {total} sessions have complete structure")
        return True


def main():
    print("=" * 60)
    print("DATA FIX VERIFICATION")
    print("=" * 60)
    print()

    results = {
        "F1 pressure monotonic": verify_f1_pressure_monotonic(),
        "F2+F3 gripper clean": verify_f2_f3_gripper_clean(),
        "F5 no baidu files": verify_f5_no_baidu_files(),
        "Structure integrity": verify_structure_integrity(),
    }

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED — review output above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())

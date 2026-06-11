#!/usr/bin/env python3
"""F2+F3: Fix gripper_state.csv — deduplicate by timestamp and fill empty sys_state.

Usage:
    python scripts/fix_f2_f3_gripper.py [--dry-run]
"""

import csv
import sys
from pathlib import Path

SESSIONS_ROOT = Path("dataset/phase2_realdata_sessions/sessions")


def fix_gripper(session_path: Path, dry_run: bool = False) -> dict:
    csv_path = session_path / "robot_state" / "gripper_state.csv"
    if not csv_path.exists():
        return {"skipped": True}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    original_count = len(rows)
    fixes = {"deduped": 0, "filled_empty": 0, "original": original_count}

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

    # Forward fill
    last_valid = ""
    for row in rows:
        if row[sys_state_idx] == "":
            if last_valid:
                row[sys_state_idx] = last_valid
                fixes["filled_empty"] += 1
        else:
            last_valid = row[sys_state_idx]

    # Backward fill for any remaining empty at the start
    next_valid = ""
    for row in reversed(rows):
        if row[sys_state_idx] == "":
            if next_valid:
                row[sys_state_idx] = next_valid
                fixes["filled_empty"] += 1
        else:
            next_valid = row[sys_state_idx]

    fixes["final"] = len(rows)

    if fixes["deduped"] == 0 and fixes["filled_empty"] == 0:
        return fixes  # No changes needed

    if dry_run:
        return fixes

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return fixes


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN MODE — no files will be modified\n")

    print("F2+F3: Fixing gripper_state.csv (dedup + fill empty sys_state)\n")

    total_deduped = 0
    total_filled = 0
    sessions_fixed = 0

    for sdir in sorted(SESSIONS_ROOT.iterdir()):
        if not sdir.is_dir() or not sdir.name.startswith("session_"):
            continue
        result = fix_gripper(sdir, dry_run=dry_run)
        if result.get("deduped", 0) > 0 or result.get("filled_empty", 0) > 0:
            sessions_fixed += 1
            total_deduped += result["deduped"]
            total_filled += result["filled_empty"]
            print(f"  {sdir.name}: deduped={result['deduped']}, filled={result['filled_empty']}, "
                  f"{result['original']}→{result['final']} rows")

    action = "would fix" if dry_run else "fixed"
    print(f"\nDone: {sessions_fixed} sessions {action}")
    print(f"  Total deduped rows: {total_deduped}")
    print(f"  Total filled empty cells: {total_filled}")


if __name__ == "__main__":
    main()

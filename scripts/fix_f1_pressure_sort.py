#!/usr/bin/env python3
"""F1: Sort pressure.csv by timestamp_us for sessions with non-monotonic timestamps.

Usage:
    python scripts/fix_f1_pressure_sort.py [--dry-run]
"""

import csv
import sys
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


def count_reversals(timestamps):
    """Count non-monotonic transitions in a list of timestamps."""
    return sum(1 for i in range(len(timestamps) - 1) if timestamps[i + 1] < timestamps[i])


def fix_pressure_sort(session_name: str, dry_run: bool = False) -> bool:
    csv_path = SESSIONS_ROOT / session_name / "pressure" / "pressure.csv"
    if not csv_path.exists():
        print(f"  SKIP {session_name}: file not found")
        return False

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Parse timestamps
    timestamps = [int(row[0]) for row in rows]
    reversals_before = count_reversals(timestamps)

    if reversals_before == 0:
        print(f"  OK   {session_name}: already monotonic")
        return True

    # Sort by timestamp_us (first column)
    rows.sort(key=lambda r: int(r[0]))

    # Verify monotonic after sort
    timestamps_after = [int(row[0]) for row in rows]
    reversals_after = count_reversals(timestamps_after)

    if reversals_after > 0:
        print(f"  FAIL {session_name}: still {reversals_after} reversals after sort")
        return False

    if dry_run:
        print(f"  DRY  {session_name}: would fix {reversals_before} reversals ({len(rows)} rows)")
        return True

    # Write back
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  OK   {session_name}: fixed {reversals_before} reversals ({len(rows)} rows)")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN MODE — no files will be modified\n")

    print("F1: Sorting pressure.csv timestamps\n")
    fixed = 0
    for s in AFFECTED_SESSIONS:
        if fix_pressure_sort(s, dry_run=dry_run):
            fixed += 1

    print(f"\nDone: {fixed}/{len(AFFECTED_SESSIONS)} sessions {'checked' if dry_run else 'fixed'}")


if __name__ == "__main__":
    main()

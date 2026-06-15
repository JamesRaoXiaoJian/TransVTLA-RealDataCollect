#!/usr/bin/env python3
"""Trim pressure CSV files to the standard 20 tactile channels.

Default mode is dry-run. Use --apply to rewrite pressure.csv after creating a
full-column backup next to the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channel_config import VALID_CHANNELS

BACKUP_NAME = "pressure.full64.backup.csv"
MANIFEST_NAME = "channel_trim_manifest.json"


@dataclass
class TrimResult:
    path: Path
    status: str
    input_columns: int = 0
    output_columns: int = 0
    rows: int = 0
    message: str = ""


def _timestamp_columns(header: list[str]) -> list[str]:
    cols = [c for c in ("sensor_timestamp_us", "host_monotonic_us", "timestamp_us") if c in header]
    if cols:
        return cols
    return header[:1]


def _count_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def trim_one(csv_path: Path, apply: bool) -> TrimResult:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []

    channel_cols = [c for c in header if c.startswith("CH")]
    valid_cols = [f"CH{ch}" for ch in VALID_CHANNELS]
    timestamp_cols = _timestamp_columns(header)
    output_header = timestamp_cols + valid_cols

    missing = [c for c in output_header if c not in header]
    if missing:
        return TrimResult(
            path=csv_path,
            status="error",
            input_columns=len(header),
            message=f"missing required columns: {missing}",
        )

    extra_channels = [c for c in channel_cols if c not in valid_cols]
    if not extra_channels and header == output_header:
        return TrimResult(
            path=csv_path,
            status="already_standard",
            input_columns=len(header),
            output_columns=len(output_header),
            rows=_count_rows(csv_path),
        )

    if not apply:
        return TrimResult(
            path=csv_path,
            status="would_trim",
            input_columns=len(header),
            output_columns=len(output_header),
            rows=_count_rows(csv_path),
            message=f"remove {len(extra_channels)} non-standard channel columns",
        )

    backup_path = csv_path.parent / BACKUP_NAME
    if backup_path.exists():
        return TrimResult(
            path=csv_path,
            status="skipped",
            input_columns=len(header),
            output_columns=len(output_header),
            message=f"backup already exists: {backup_path.name}",
        )

    shutil.copy2(csv_path, backup_path)
    source_rows = _count_rows(csv_path)
    backup_rows = _count_rows(backup_path)
    if source_rows != backup_rows:
        return TrimResult(
            path=csv_path,
            status="error",
            input_columns=len(header),
            output_columns=len(output_header),
            rows=source_rows,
            message=f"backup row count mismatch: source={source_rows}, backup={backup_rows}",
        )

    tmp_path = csv_path.with_suffix(".trim.tmp")
    written = 0
    with open(csv_path, newline="", encoding="utf-8") as src, open(
        tmp_path, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=output_header)
        writer.writeheader()
        for row in reader:
            writer.writerow({name: row.get(name, "") for name in output_header})
            written += 1
    tmp_path.replace(csv_path)

    manifest = {
        "schema": "pressure_channel_trim/v1",
        "trimmed_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(csv_path),
        "backup_file": str(backup_path),
        "input_columns": header,
        "output_columns": output_header,
        "valid_channels": VALID_CHANNELS,
        "rows": written,
    }
    with open(csv_path.parent / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return TrimResult(
        path=csv_path,
        status="trimmed",
        input_columns=len(header),
        output_columns=len(output_header),
        rows=written,
        message=f"backup={backup_path.name}",
    )


def iter_pressure_csvs(data_root: Path) -> list[Path]:
    return sorted(data_root.rglob("pressure/pressure.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim pressure.csv files to standard 20 tactile channels.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing sessions.")
    parser.add_argument("--apply", action="store_true", help="Rewrite files after creating backups.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of files to process.")
    args = parser.parse_args()

    csv_files = iter_pressure_csvs(args.data_root)
    if args.limit is not None:
        csv_files = csv_files[: args.limit]

    print(f"mode={'apply' if args.apply else 'dry-run'} files={len(csv_files)} root={args.data_root}")
    counts: dict[str, int] = {}
    failures: list[TrimResult] = []

    for csv_path in csv_files:
        result = trim_one(csv_path, apply=args.apply)
        counts[result.status] = counts.get(result.status, 0) + 1
        rel = csv_path.relative_to(args.data_root)
        detail = f" {result.message}" if result.message else ""
        print(
            f"[{result.status}] {rel} "
            f"cols {result.input_columns}->{result.output_columns} rows={result.rows}{detail}"
        )
        if result.status == "error":
            failures.append(result)

    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

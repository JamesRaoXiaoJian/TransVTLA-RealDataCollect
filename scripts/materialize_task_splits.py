#!/usr/bin/env python3
"""Expand task range annotations into session manifests and symlink views.

Input annotations are ranges over visual review sheets created by
build_task_preview_sheets.py. The output keeps the existing hardware split and
adds a task-level subdivision without moving or deleting source sessions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_MANIFEST = Path("dataset/audit_reports/clean_split_manifest.csv")
DEFAULT_PREVIEW_MANIFEST = Path("dataset/task_review/previews/preview_manifest.csv")
DEFAULT_ANNOTATIONS = Path("dataset/task_review/task_range_annotations.csv")
DEFAULT_OUT_DIR = Path("dataset/task_splits")

SPLIT_ORDER = [
    "dual_realsense_repositioned",
    "dual_realsense_initial_position_begin_end",
    "legacy_dji_realsense_rgbd",
    "legacy_dji_realsense_rgb_only",
]

TASKS = {
    "task1_simple_put_object_in_box": "Simple scene: put target object into box",
    "task2_complex_put_object_or_beaker_tripod": "Complex scene: put object into box or put beaker on tripod",
    "task3_simple_beaker_tripod": "Simple scene: put beaker on tripod",
    "task4_test_tube_rack_insertion": "Insert test tube into test tube rack",
    "unknown_or_mixed": "Uncertain, mixed, or insufficient visual evidence",
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        return str(path.absolute().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_out_dir(out_dir: Path, clean: bool) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _sorted_sessions(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["split"]].append(row)
    for split_rows in result.values():
        split_rows.sort(key=lambda row: (row.get("date", ""), row.get("time", ""), row.get("session", "")))
    return result


def _sheet_session_map(clean_rows: list[dict[str, str]], preview_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_split = _sorted_sessions(clean_rows)
    mapping: dict[str, list[dict[str, str]]] = {}
    for preview in preview_rows:
        split = preview["split"]
        session_count = int(preview["session_count"])
        split_sessions = by_split[split]
        start = next(
            (idx for idx, row in enumerate(split_sessions) if row["session"] == preview["start_session"]),
            -1,
        )
        if start < 0:
            raise ValueError(f"No start session found for sheet {preview['sheet_id']}: {preview['start_session']}")
        chunk = split_sessions[start : start + session_count]
        if not chunk or len(chunk) != session_count:
            raise ValueError(f"No full session range found for sheet {preview['sheet_id']}")
        if chunk[0]["session"] != preview["start_session"] or chunk[-1]["session"] != preview["end_session"]:
            raise ValueError(
                f"Preview/session order mismatch for {preview['sheet_id']}: "
                f"{chunk[0]['session']}..{chunk[-1]['session']} vs "
                f"{preview['start_session']}..{preview['end_session']}"
            )
        mapping[preview["sheet_id"]] = chunk
    return mapping


def _annotation_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["sheet_id"], int(row["row_start"]), int(row["row_end"])


def _normalize_sheet_id(split: str, sheet_id: str) -> str:
    sheet_id = sheet_id.strip()
    if sheet_id.startswith(f"{split}_"):
        return sheet_id
    if sheet_id.startswith("sheet_"):
        return f"{split}_{sheet_id.removeprefix('sheet_')}"
    if sheet_id.isdigit():
        return f"{split}_{int(sheet_id):03d}"
    return sheet_id


def expand_annotations(
    clean_rows: list[dict[str, str]],
    preview_rows: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    sheet_map = _sheet_session_map(clean_rows, preview_rows)
    assignment_by_session: dict[str, dict[str, str]] = {}

    for annotation in sorted(annotation_rows, key=_annotation_key):
        sheet_id = _normalize_sheet_id(annotation["split"], annotation["sheet_id"])
        if sheet_id not in sheet_map:
            raise ValueError(f"Unknown sheet_id in annotations: {sheet_id}")
        task_id = annotation["task_id"]
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id for {sheet_id}: {task_id}")
        row_start = int(annotation["row_start"])
        row_end = int(annotation["row_end"])
        sessions = sheet_map[sheet_id]
        if row_start < 1 or row_end > len(sessions) or row_start > row_end:
            raise ValueError(f"Invalid row range for {sheet_id}: {row_start}-{row_end}")
        for row_index in range(row_start, row_end + 1):
            session = sessions[row_index - 1]
            name = session["session"]
            if name in assignment_by_session:
                raise ValueError(f"Duplicate task assignment for {name}")
            assignment_by_session[name] = {
                "task_id": task_id,
                "task_name": TASKS[task_id],
                "task_confidence": annotation.get("confidence", ""),
                "task_evidence": annotation.get("evidence", ""),
                "task_notes": annotation.get("notes", ""),
                "task_sheet_id": sheet_id,
                "task_row_in_sheet": str(row_index),
            }

    missing = [row["session"] for row in clean_rows if row["session"] not in assignment_by_session]
    if missing:
        raise ValueError(f"Missing task annotations for {len(missing)} sessions, first={missing[:5]}")

    expanded: list[dict[str, str]] = []
    for row in clean_rows:
        merged = dict(row)
        merged.update(assignment_by_session[row["session"]])
        expanded.append(merged)
    expanded.sort(
        key=lambda row: (
            SPLIT_ORDER.index(row["split"]) if row["split"] in SPLIT_ORDER else 999,
            row["task_id"],
            row.get("date", ""),
            row.get("time", ""),
            row["session"],
        )
    )
    return expanded


def _link_session(link_path: Path, target_path: Path) -> str:
    rel_target = os.path.relpath(target_path, link_path.parent)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link_path):
        if link_path.is_symlink() and os.readlink(link_path) == rel_target:
            return "already_linked"
        return "conflict"
    link_path.symlink_to(rel_target, target_is_directory=True)
    return "linked"


def _write_view(view_dir: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Counter[str]:
    counters: Counter[str] = Counter()
    _write_csv(view_dir / "manifest.csv", rows, fieldnames)
    _write_text(view_dir / "sessions.txt", "".join(f"{row['rel_path']}\n" for row in rows))
    for row in rows:
        target = _repo_path(Path(row["rel_path"]))
        link = view_dir / "sessions" / row["session"]
        counters[_link_session(link, target)] += 1
    return counters


def materialize(expanded: list[dict[str, str]], out_dir: Path, clean: bool) -> dict[str, object]:
    _clean_out_dir(out_dir, clean)
    fieldnames = list(expanded[0].keys()) if expanded else []
    _write_csv(out_dir / "task_session_manifest.csv", expanded, fieldnames)

    link_counters: Counter[str] = Counter()
    summary: dict[str, object] = {
        "schema": "transvtla_task_splits/v1",
        "tasks": TASKS,
        "total_sessions": len(expanded),
        "by_hardware_split": {},
        "by_task": {},
    }

    for split in SPLIT_ORDER:
        split_rows = [row for row in expanded if row["split"] == split]
        split_summary: dict[str, object] = {}
        for task_id in TASKS:
            task_rows = [row for row in split_rows if row["task_id"] == task_id]
            if not task_rows:
                continue
            view_dir = out_dir / "by_hardware_split" / split / task_id
            link_counters.update(_write_view(view_dir, task_rows, fieldnames))
            split_summary[task_id] = {
                "sessions": len(task_rows),
                "visual_complete_frames": sum(int(row.get("visual_complete_frames") or 0) for row in task_rows),
                "manifest": _rel(view_dir / "manifest.csv"),
                "sessions_dir": _rel(view_dir / "sessions"),
            }
        summary["by_hardware_split"][split] = split_summary

    for task_id in TASKS:
        task_rows = [row for row in expanded if row["task_id"] == task_id]
        if not task_rows:
            continue
        view_dir = out_dir / "by_task" / task_id
        link_counters.update(_write_view(view_dir, task_rows, fieldnames))
        summary["by_task"][task_id] = {
            "sessions": len(task_rows),
            "visual_complete_frames": sum(int(row.get("visual_complete_frames") or 0) for row in task_rows),
            "manifest": _rel(view_dir / "manifest.csv"),
            "sessions_dir": _rel(view_dir / "sessions"),
        }

    summary["link_status_counts"] = dict(sorted(link_counters.items()))
    with open(out_dir / "task_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_text(out_dir / "README.md", _readme(summary))
    return summary


def _readme(summary: dict[str, object]) -> str:
    lines = [
        "# Task Split Views",
        "",
        "This directory adds task-level splits on top of the hardware splits.",
        "All `sessions/` entries are symlinks to original sessions.",
        "",
        "Task IDs:",
    ]
    for task_id, task_name in TASKS.items():
        lines.append(f"- `{task_id}`: {task_name}")
    lines.extend(["", "Global task counts:", "", "| task | sessions | complete visual frames |", "| --- | ---: | ---: |"])
    by_task = summary.get("by_task", {})
    if isinstance(by_task, dict):
        for task_id in TASKS:
            info = by_task.get(task_id)
            if isinstance(info, dict):
                lines.append(f"| `{task_id}` | {info['sessions']} | {info['visual_complete_frames']:,} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize task-level split views from range annotations.")
    parser.add_argument("--clean-manifest", type=Path, default=DEFAULT_CLEAN_MANIFEST)
    parser.add_argument("--preview-manifest", type=Path, default=DEFAULT_PREVIEW_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Remove existing output dir before writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clean_rows = _read_csv(_repo_path(args.clean_manifest))
    preview_rows = _read_csv(_repo_path(args.preview_manifest))
    annotation_rows = _read_csv(_repo_path(args.annotations))
    expanded = expand_annotations(clean_rows, preview_rows, annotation_rows)
    summary = materialize(expanded, _repo_path(args.out_dir), clean=args.clean)
    print(f"sessions={summary['total_sessions']} out_dir={_rel(_repo_path(args.out_dir))}")
    for task_id, info in summary["by_task"].items():
        print(f"{task_id}: sessions={info['sessions']} frames={info['visual_complete_frames']}")
    print(f"links={summary['link_status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

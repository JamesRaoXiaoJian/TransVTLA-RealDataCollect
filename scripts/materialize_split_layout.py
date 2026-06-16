#!/usr/bin/env python3
"""Create a non-destructive split-view layout for cleaned sessions.

The source dataset layout is left untouched. This script reads the clean split
manifest and materializes a view directory containing per-split manifests plus
relative symlinks to the original session directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("dataset/audit_reports/clean_split_manifest.csv")
DEFAULT_OUT_DIR = Path("dataset/split_views")

SPLIT_ORDER = [
    "dual_realsense_repositioned",
    "dual_realsense_initial_position_begin_end",
    "legacy_dji_realsense_rgbd",
    "legacy_dji_realsense_rgb_only",
]


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        return str(path.absolute().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str], apply: bool) -> None:
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str, apply: bool) -> None:
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _split_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_ORDER}
    for row in rows:
        split = row.get("split", "unknown") or "unknown"
        result.setdefault(split, []).append(row)
    for split_rows in result.values():
        split_rows.sort(key=lambda row: (row.get("date", ""), row.get("time", ""), row.get("session", "")))
    return result


def _link_session(link_path: Path, target_path: Path, apply: bool) -> tuple[str, str]:
    rel_target = os.path.relpath(target_path, link_path.parent)
    if not apply:
        return "would_link", rel_target

    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link_path):
        if link_path.is_symlink() and os.readlink(link_path) == rel_target:
            return "already_linked", rel_target
        return "conflict", rel_target

    link_path.symlink_to(rel_target, target_is_directory=True)
    return "linked", rel_target


def _split_readme(split: str, rows: list[dict[str, str]]) -> str:
    frames = sum(int(row.get("visual_complete_frames") or 0) for row in rows)
    rgbd = sum(int(row.get("usable_for_rgbd") or 0) for row in rows)
    aligned = sum(int(row.get("usable_for_aligned_training") or 0) for row in rows)
    return (
        f"# {split}\n\n"
        "This directory is a non-destructive split view. The `sessions/` entries\n"
        "are symlinks to the original session directories; source data remains in\n"
        "`dataset/sessions` and `dataset/phase2_realdata_sessions/sessions`.\n\n"
        f"- Sessions: {len(rows)}\n"
        f"- Complete visual frames: {frames}\n"
        f"- RGB-D usable sessions: {rgbd}\n"
        f"- Aligned dual-RealSense sessions: {aligned}\n\n"
        "Files:\n"
        "- `manifest.csv`: per-session metadata from the clean split manifest.\n"
        "- `sessions.txt`: original session paths, one per line.\n"
        "- `sessions/`: symlink view for filesystem-based loaders.\n"
    )


def _root_readme(out_dir: Path, split_map: dict[str, list[dict[str, str]]]) -> str:
    lines = [
        "# Clean Split Views",
        "",
        "This directory reorganizes the cleaned dataset by split without moving,",
        "copying, or deleting the original session directories.",
        "",
        "Layout:",
        "",
        "```text",
        f"{_rel(out_dir)}/",
        "  split_summary.json",
        "  all_manifest.csv",
    ]
    for split in SPLIT_ORDER:
        lines.extend(
            [
                f"  {split}/",
                "    manifest.csv",
                "    sessions.txt",
                "    sessions/",
            ]
        )
    lines.extend(["```", ""])
    lines.append("| split | sessions | complete visual frames |")
    lines.append("| --- | ---: | ---: |")
    for split in SPLIT_ORDER:
        rows = split_map.get(split, [])
        frames = sum(int(row.get("visual_complete_frames") or 0) for row in rows)
        lines.append(f"| `{split}` | {len(rows)} | {frames:,} |")
    lines.append("")
    lines.append("Use `manifest.csv` when a training pipeline should avoid symlink traversal.")
    return "\n".join(lines) + "\n"


def materialize(manifest: Path, out_dir: Path, apply: bool) -> dict[str, object]:
    rows = _read_manifest(manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    split_map = _split_rows(rows)

    counters: Counter[str] = Counter()
    split_summary: dict[str, dict[str, object]] = {}

    _write_csv(out_dir / "all_manifest.csv", rows, fieldnames, apply)

    for split in SPLIT_ORDER:
        split_rows = split_map.get(split, [])
        split_dir = out_dir / split
        sessions_dir = split_dir / "sessions"
        _write_csv(split_dir / "manifest.csv", split_rows, fieldnames, apply)
        _write_text(
            split_dir / "sessions.txt",
            "".join(f"{row['rel_path']}\n" for row in split_rows),
            apply,
        )
        _write_text(split_dir / "README.md", _split_readme(split, split_rows), apply)

        linked_sessions = []
        for row in split_rows:
            target = _repo_path(Path(row["rel_path"]))
            link = sessions_dir / row["session"]
            status, rel_target = _link_session(link, target, apply)
            counters[status] += 1
            linked_sessions.append(
                {
                    "session": row["session"],
                    "source": row["rel_path"],
                    "link": _rel(link),
                    "target": rel_target,
                    "status": status,
                }
            )

        split_summary[split] = {
            "sessions": len(split_rows),
            "visual_complete_frames": sum(int(row.get("visual_complete_frames") or 0) for row in split_rows),
            "usable_for_rgb": sum(int(row.get("usable_for_rgb") or 0) for row in split_rows),
            "usable_for_rgbd": sum(int(row.get("usable_for_rgbd") or 0) for row in split_rows),
            "usable_for_aligned_training": sum(
                int(row.get("usable_for_aligned_training") or 0) for row in split_rows
            ),
            "manifest": _rel(split_dir / "manifest.csv"),
            "sessions_txt": _rel(split_dir / "sessions.txt"),
            "sessions_dir": _rel(sessions_dir),
            "links": linked_sessions,
        }

    payload = {
        "schema": "transvtla_split_views/v1",
        "mode": "apply" if apply else "dry-run",
        "source_manifest": _rel(manifest),
        "out_dir": _rel(out_dir),
        "splits": split_summary,
        "link_status_counts": dict(sorted(counters.items())),
    }

    _write_text(out_dir / "README.md", _root_readme(out_dir, split_map), apply)
    if apply:
        with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-destructive split-view directories.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--apply", action="store_true", help="Create directories, manifests, and symlinks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _repo_path(args.manifest)
    out_dir = _repo_path(args.out_dir)
    payload = materialize(manifest, out_dir, args.apply)
    print(f"mode={payload['mode']} out_dir={payload['out_dir']}")
    for split in SPLIT_ORDER:
        info = payload["splits"][split]
        print(
            f"{split}: sessions={info['sessions']} "
            f"frames={info['visual_complete_frames']} "
            f"rgbd={info['usable_for_rgbd']} aligned={info['usable_for_aligned_training']}"
        )
    print(f"links: {payload['link_status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

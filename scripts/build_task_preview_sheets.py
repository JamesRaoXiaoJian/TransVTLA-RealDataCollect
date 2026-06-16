#!/usr/bin/env python3
"""Build image contact sheets for task-level session classification.

The sheets are only review artifacts. They do not move or modify source data.
Each row is one session, ordered by split/date/time. The thumbnails show three
frames from the primary view and one middle frame from the secondary view.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("dataset/audit_reports/clean_split_manifest.csv")
DEFAULT_OUT_DIR = Path("dataset/task_review/previews")

SPLIT_ORDER = [
    "dual_realsense_repositioned",
    "dual_realsense_initial_position_begin_end",
    "legacy_dji_realsense_rgbd",
    "legacy_dji_realsense_rgb_only",
]

VIEW_BY_FORMAT = {
    "dual_realsense": {
        "primary": ("world_camera/rgb", ".jpg", "world"),
        "secondary": ("wrist_camera/rgb", ".jpg", "wrist"),
    },
    "legacy_dji_realsense": {
        "primary": ("dji", ".jpg", "dji"),
        "secondary": ("realsense_rgb", ".jpg", "rs"),
    },
    "legacy_dji_realsense_rgb_only": {
        "primary": ("dji", ".jpg", "dji"),
        "secondary": ("realsense_rgb", ".jpg", "rs"),
    },
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        return str(path.absolute().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _sort_key(path: Path) -> tuple[int, str]:
    number = _safe_int(path.stem)
    return (number if number is not None else 10**12, path.name)


def _list_images(session: Path, rel_dir: str, suffix: str) -> list[Path]:
    directory = session / rel_dir
    if not directory.is_dir():
        return []
    return sorted((p for p in directory.glob(f"*{suffix}") if p.is_file()), key=_sort_key)


def _pick(paths: list[Path], fraction: float) -> Path | None:
    if not paths:
        return None
    index = min(len(paths) - 1, max(0, round((len(paths) - 1) * fraction)))
    return paths[index]


def _load_thumb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None:
        image = Image.new("RGB", size, (225, 225, 225))
        return image
    with Image.open(path) as src:
        src = src.convert("RGB")
        src.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (245, 245, 245))
        x = (size[0] - src.width) // 2
        y = (size[1] - src.height) // 2
        canvas.paste(src, (x, y))
        return canvas


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_rows(manifest: Path) -> list[dict[str, str]]:
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(
        key=lambda row: (
            SPLIT_ORDER.index(row["split"]) if row.get("split") in SPLIT_ORDER else 999,
            row.get("date", ""),
            row.get("time", ""),
            row.get("session", ""),
        )
    )
    return rows


def _short_split(split: str) -> str:
    mapping = {
        "dual_realsense_repositioned": "dual-repos",
        "dual_realsense_initial_position_begin_end": "dual-initial",
        "legacy_dji_realsense_rgbd": "legacy-rgbd",
        "legacy_dji_realsense_rgb_only": "legacy-rgb",
    }
    return mapping.get(split, split)


def _session_images(row: dict[str, str]) -> list[tuple[str, Path | None]]:
    session = _repo_path(Path(row["rel_path"]))
    config = VIEW_BY_FORMAT.get(row.get("data_format", ""), VIEW_BY_FORMAT["dual_realsense"])
    primary_dir, primary_suffix, primary_label = config["primary"]
    secondary_dir, secondary_suffix, secondary_label = config["secondary"]
    primary = _list_images(session, primary_dir, primary_suffix)
    secondary = _list_images(session, secondary_dir, secondary_suffix)
    return [
        (f"{primary_label}_start", _pick(primary, 0.08)),
        (f"{primary_label}_mid", _pick(primary, 0.50)),
        (f"{primary_label}_end", _pick(primary, 0.92)),
        (f"{secondary_label}_mid", _pick(secondary, 0.50)),
    ]


def _draw_sheet(
    rows: list[dict[str, str]],
    out_path: Path,
    sheet_id: str,
    thumb_size: tuple[int, int],
    label_width: int,
) -> None:
    row_height = thumb_size[1] + 46
    header_height = 54
    gutter = 10
    width = label_width + 4 * thumb_size[0] + 5 * gutter
    height = header_height + len(rows) * row_height + gutter
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(20)
    label_font = _font(13)
    small_font = _font(11)

    draw.rectangle((0, 0, width, header_height), fill=(30, 36, 44))
    draw.text((14, 14), sheet_id, fill=(255, 255, 255), font=title_font)

    for row_index, row in enumerate(rows):
        y = header_height + row_index * row_height
        if row_index % 2:
            draw.rectangle((0, y, width, y + row_height), fill=(248, 250, 252))
        split = _short_split(row.get("split", ""))
        label_lines = [
            f"{row_index + 1:02d} {row['session']}",
            f"{split}  {row.get('date', '')} {row.get('time', '')}",
            f"frames={row.get('visual_complete_frames', '')} notes={row.get('notes', '')}",
        ]
        for offset, text in enumerate(label_lines):
            draw.text((12, y + 12 + offset * 18), text[:42], fill=(20, 23, 28), font=label_font)

        for col, (view_label, frame_path) in enumerate(_session_images(row)):
            x = label_width + gutter + col * (thumb_size[0] + gutter)
            thumb = _load_thumb(frame_path, thumb_size)
            image.paste(thumb, (x, y + 8))
            draw.rectangle((x, y + 8, x + thumb_size[0], y + 8 + thumb_size[1]), outline=(170, 176, 186))
            draw.text((x + 4, y + 14 + thumb_size[1]), view_label, fill=(50, 56, 66), font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)


def build_sheets(
    manifest: Path,
    out_dir: Path,
    sessions_per_sheet: int,
    thumb_size: tuple[int, int],
    clean: bool,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(manifest)
    preview_rows: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}

    for split in SPLIT_ORDER:
        split_rows = [row for row in rows if row.get("split") == split]
        summary[split] = {
            "sessions": len(split_rows),
            "sheets": math.ceil(len(split_rows) / sessions_per_sheet) if split_rows else 0,
            "visual_complete_frames": sum(int(row.get("visual_complete_frames") or 0) for row in split_rows),
        }
        for sheet_index in range(0, len(split_rows), sessions_per_sheet):
            chunk = split_rows[sheet_index : sheet_index + sessions_per_sheet]
            sheet_number = sheet_index // sessions_per_sheet
            sheet_id = f"{split} sheet {sheet_number:03d}"
            out_path = out_dir / split / f"sheet_{sheet_number:03d}.jpg"
            _draw_sheet(chunk, out_path, sheet_id, thumb_size=thumb_size, label_width=330)
            preview_rows.append(
                {
                    "sheet_id": f"{split}_{sheet_number:03d}",
                    "split": split,
                    "sheet_index": str(sheet_number),
                    "sheet_path": _rel(out_path),
                    "session_count": str(len(chunk)),
                    "start_session": chunk[0]["session"],
                    "end_session": chunk[-1]["session"],
                    "start_date": chunk[0].get("date", ""),
                    "start_time": chunk[0].get("time", ""),
                    "end_date": chunk[-1].get("date", ""),
                    "end_time": chunk[-1].get("time", ""),
                }
            )

    manifest_path = out_dir / "preview_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sheet_id",
            "split",
            "sheet_index",
            "sheet_path",
            "session_count",
            "start_session",
            "end_session",
            "start_date",
            "start_time",
            "end_date",
            "end_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(preview_rows)

    return preview_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual review contact sheets for task splitting.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sessions-per-sheet", type=int, default=12)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--thumb-height", type=int, default=150)
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory before writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _repo_path(args.manifest)
    out_dir = _repo_path(args.out_dir)
    preview_rows, summary = build_sheets(
        manifest,
        out_dir,
        sessions_per_sheet=args.sessions_per_sheet,
        thumb_size=(args.thumb_width, args.thumb_height),
        clean=args.clean,
    )
    print(f"wrote {len(preview_rows)} sheets to {_rel(out_dir)}")
    for split in SPLIT_ORDER:
        info = summary[split]
        print(f"{split}: sessions={info['sessions']} sheets={info['sheets']} frames={info['visual_complete_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

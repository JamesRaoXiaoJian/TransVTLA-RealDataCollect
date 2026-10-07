#!/usr/bin/env python3
"""Audit Isaac Sim Franka real-schema collection output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SESSION_RE = re.compile(r"^session_ep(?P<episode>\d+)_env(?P<env>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/media/data"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-total-env-sessions", type=int, default=10000)
    parser.add_argument("--expected-episodes-per-shard", type=int, default=500)
    parser.add_argument("--expected-envs-per-shard", type=int, default=5)
    parser.add_argument("--expected-frames-per-session", type=int, default=70)
    parser.add_argument("--expected-width", type=int, default=848)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-save-image-interval", type=int, default=5)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--max-errors", type=int, default=500)
    parser.add_argument("--sample-contact-sheets", action="store_true", default=True)
    parser.add_argument(
        "--quick-images",
        action="store_true",
        help="Only open first/middle/last image per session. Counts are still checked fully.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_csv_rows(path: Path, timestamp_columns: tuple[int, ...] = (0,)) -> tuple[int, bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return 0, False, [f"missing csv: {path}"]
    previous: dict[int, int] = {}
    rows = 0
    monotonic = True
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0, False, [f"empty csv: {path}"]
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            rows += 1
            for col in timestamp_columns:
                if col >= len(row):
                    continue
                try:
                    value = int(float(row[col]))
                except ValueError:
                    errors.append(f"{path}:{line_no}: invalid timestamp column {col}: {row[col]!r}")
                    monotonic = False
                    continue
                old = previous.get(col)
                if old is not None and value < old:
                    errors.append(f"{path}:{line_no}: non-monotonic timestamp column {col}: {value} < {old}")
                    monotonic = False
                    break
                previous[col] = value
    return rows, monotonic, errors


def read_frames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], [f"missing frames.csv: {path}"]
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    errors: list[str] = []
    expected_fields = {
        "frame_id",
        "capture_monotonic_us",
        "world_rgb_save_us",
        "world_depth_save_us",
        "wrist_rgb_save_us",
        "wrist_depth_save_us",
        "save_complete_us",
        "save_queue_depth",
    }
    if not expected_fields.issubset(set(reader.fieldnames or [])):
        errors.append(f"{path}: missing frame columns: {sorted(expected_fields - set(reader.fieldnames or []))}")
    last_capture = -1
    for idx, row in enumerate(rows, start=1):
        try:
            frame_id = int(row["frame_id"])
            capture = int(row["capture_monotonic_us"])
        except Exception as exc:
            errors.append(f"{path}: invalid frame row {idx}: {exc}")
            continue
        if frame_id != idx:
            errors.append(f"{path}: expected frame_id {idx}, got {frame_id}")
        if capture < last_capture:
            errors.append(f"{path}: non-monotonic capture timestamp at frame {idx}")
        last_capture = capture
    return rows, errors


def image_indices(total: int, quick: bool) -> list[int]:
    if total <= 0:
        return []
    if not quick:
        return list(range(total))
    return sorted({0, total // 2, total - 1})


def audit_images(
    files: list[Path],
    *,
    expected_count: int,
    expected_size: tuple[int, int],
    depth: bool,
    quick: bool,
    stats: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if len(files) != expected_count:
        errors.append(f"{files[0].parent if files else 'missing-image-dir'}: expected {expected_count} files, got {len(files)}")
    for idx, path in enumerate(files, start=1):
        expected_name = f"{idx:04d}.{'png' if depth else 'jpg'}"
        if path.name != expected_name:
            errors.append(f"{path}: unexpected name, expected {expected_name}")
            if len(errors) > 20:
                break
    for idx in image_indices(len(files), quick):
        path = files[idx]
        try:
            with Image.open(path) as im:
                if im.size != expected_size:
                    errors.append(f"{path}: size {im.size}, expected {expected_size}")
                arr = np.asarray(im)
        except Exception as exc:
            errors.append(f"{path}: open failed: {exc}")
            continue
        if depth:
            if arr.shape[:2] != (expected_size[1], expected_size[0]):
                errors.append(f"{path}: depth shape {arr.shape}, expected {(expected_size[1], expected_size[0])}")
            if arr.dtype != np.uint16:
                errors.append(f"{path}: depth dtype {arr.dtype}, expected uint16")
            nonzero = arr[arr > 0]
            stats["depth_images_checked"] += 1
            stats["depth_zero_pixels"] += int((arr == 0).sum())
            stats["depth_saturated_pixels"] += int((arr == 65535).sum())
            if nonzero.size == 0:
                errors.append(f"{path}: all-zero depth")
            else:
                stats["depth_min_mm"] = min(stats["depth_min_mm"], int(nonzero.min()))
                stats["depth_max_mm"] = max(stats["depth_max_mm"], int(nonzero.max()))
        else:
            stats["rgb_images_checked"] += 1
            if arr.ndim != 3 or arr.shape[2] < 3:
                errors.append(f"{path}: unexpected rgb shape {arr.shape}")
            elif int(arr.max()) == 0:
                errors.append(f"{path}: all-black rgb")
    return errors


def make_contact_sheets(run_dirs: list[Path], report_dir: Path, expected_prefix: str) -> list[str]:
    outputs: list[str] = []
    items: list[tuple[str, str, int, Path, Path, Path]] = []
    for run_dir in run_dirs:
        shard_label = run_dir.name.replace(expected_prefix + "_", "")
        for session in sorted(run_dir.glob("session_ep*_env*"))[:2]:
            world = sorted((session / "world_camera" / "rgb").glob("*.jpg"))
            wrist = sorted((session / "wrist_camera" / "rgb").glob("*.jpg"))
            wrist_depth = sorted((session / "wrist_camera" / "depth").glob("*.png"))
            if not world or not wrist or not wrist_depth:
                continue
            for idx in sorted({0, len(world) // 2, len(world) - 1}):
                if idx < len(wrist) and idx < len(wrist_depth):
                    items.append((shard_label, session.name, idx + 1, world[idx], wrist[idx], wrist_depth[idx]))
            if len(items) >= 24:
                break
        if len(items) >= 24:
            break
    if not items:
        return outputs

    thumb_w, thumb_h, label_h, cols = 212, 120, 34, 6
    rows = math.ceil(len(items) / cols)
    font = ImageFont.load_default()
    specs = [
        ("world_rgb", 3),
        ("wrist_rgb", 4),
        ("wrist_depth", 5),
    ]
    for name, item_index in specs:
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for i, item in enumerate(items):
            shard, session_name, frame_no = item[:3]
            path = item[item_index]
            if name.endswith("depth"):
                arr = np.asarray(Image.open(path)).astype(np.float32)
                valid = arr[arr > 0]
                lo = float(np.percentile(valid, 2)) if valid.size else 0.0
                hi = float(np.percentile(valid, 98)) if valid.size else 1.0
                vis = np.clip((arr - lo) / max(1.0, hi - lo), 0, 1)
                img = Image.fromarray((vis * 255).astype(np.uint8), "L").convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            img = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (i % cols) * thumb_w
            y = (i // cols) * (thumb_h + label_h)
            sheet.paste(img, (x, y))
            draw.text((x + 4, y + thumb_h + 3), f"{shard}\n{session_name} f{frame_no:02d}", fill=(235, 235, 235), font=font)
        out = report_dir / f"{name}_contact.jpg"
        sheet.save(out, quality=92)
        outputs.append(str(out))
    return outputs


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = args.report_dir or (args.output_dir / f"{args.run_prefix}_audit_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(p for p in args.output_dir.glob(f"{args.run_prefix}_shard*_gpu*") if p.is_dir())
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "run_prefix": args.run_prefix,
        "audit_started_at": timestamp,
        "run_dirs": [str(p) for p in run_dirs],
        "sessions": 0,
        "frames": 0,
        "rgb_images_checked": 0,
        "depth_images_checked": 0,
        "depth_zero_pixels": 0,
        "depth_saturated_pixels": 0,
        "depth_min_mm": 2**31 - 1,
        "depth_max_mm": 0,
        "frame_count_histogram": Counter(),
        "csv_row_ranges": defaultdict(lambda: [10**18, 0]),
        "per_shard": {},
    }

    if len(run_dirs) != args.expected_shards:
        errors.append(f"expected {args.expected_shards} shard dirs, got {len(run_dirs)}")

    expected_size = (args.expected_width, args.expected_height)
    expected_sessions_per_shard = args.expected_episodes_per_shard * args.expected_envs_per_shard

    for run_dir in run_dirs:
        shard_stats: dict[str, Any] = {"sessions": 0, "frames": 0, "episodes": set(), "envs": set()}
        stats["per_shard"][run_dir.name] = shard_stats
        cfg_path = run_dir / "collection_config.json"
        if not cfg_path.exists():
            errors.append(f"{run_dir}: missing collection_config.json")
            continue
        try:
            cfg = read_json(cfg_path)
        except Exception as exc:
            errors.append(f"{cfg_path}: cannot parse json: {exc}")
            continue
        expected_cfg = {
            "num_envs": args.expected_envs_per_shard,
            "episodes": args.expected_episodes_per_shard,
            "world_camera_width": args.expected_width,
            "world_camera_height": args.expected_height,
            "wrist_camera_width": args.expected_width,
            "wrist_camera_height": args.expected_height,
            "save_image_interval": args.expected_save_image_interval,
        }
        for key, expected in expected_cfg.items():
            if cfg.get(key) != expected:
                errors.append(f"{cfg_path}: {key}={cfg.get(key)!r}, expected {expected!r}")

        sessions = sorted(p for p in run_dir.glob("session_ep*_env*") if p.is_dir())
        if len(sessions) != expected_sessions_per_shard:
            errors.append(f"{run_dir}: expected {expected_sessions_per_shard} sessions, got {len(sessions)}")

        seen_pairs: set[tuple[int, int]] = set()
        for session in sessions:
            match = SESSION_RE.match(session.name)
            if not match:
                errors.append(f"{session}: bad session name")
                continue
            episode = int(match.group("episode"))
            env_id = int(match.group("env"))
            seen_pairs.add((episode, env_id))
            shard_stats["episodes"].add(episode)
            shard_stats["envs"].add(env_id)

            frame_rows, frame_errors = read_frames(session / "frames.csv")
            errors.extend(frame_errors)
            frame_count = len(frame_rows)
            stats["sessions"] += 1
            stats["frames"] += frame_count
            shard_stats["sessions"] += 1
            shard_stats["frames"] += frame_count
            stats["frame_count_histogram"][frame_count] += 1
            if frame_count != args.expected_frames_per_session:
                errors.append(f"{session}: expected {args.expected_frames_per_session} frames, got {frame_count}")

            for rel, ext, depth in [
                ("world_camera/rgb", "*.jpg", False),
                ("world_camera/depth", "*.png", True),
                ("wrist_camera/rgb", "*.jpg", False),
                ("wrist_camera/depth", "*.png", True),
            ]:
                files = sorted((session / rel).glob(ext))
                errors.extend(
                    audit_images(
                        files,
                        expected_count=frame_count,
                        expected_size=expected_size,
                        depth=depth,
                        quick=args.quick_images,
                        stats=stats,
                    )
                )

            for name, path, timestamp_cols in [
                ("robot_state", session / "robot_state" / "robot_state.csv", (0,)),
                ("gripper_state", session / "robot_state" / "gripper_state.csv", (0,)),
                ("pressure", session / "pressure" / "pressure.csv", (0, 1)),
            ]:
                rows, _monotonic, csv_errors = count_csv_rows(path, timestamp_cols)
                errors.extend(csv_errors)
                row_range = stats["csv_row_ranges"][name]
                row_range[0] = min(row_range[0], rows)
                row_range[1] = max(row_range[1], rows)
                if rows <= 0:
                    errors.append(f"{path}: no rows")

            for json_name in ["camera_metadata.json", "sim_metadata.json"]:
                path = session / json_name
                if not path.exists():
                    errors.append(f"{session}: missing {json_name}")
                    continue
                try:
                    payload = read_json(path)
                except Exception as exc:
                    errors.append(f"{path}: cannot parse json: {exc}")
                    continue
                if json_name == "camera_metadata.json":
                    cameras = payload.get("cameras", {})
                    for cam_name in ["world_camera", "wrist_camera"]:
                        profile = cameras.get(cam_name, {}).get("profile", {})
                        if profile.get("width") != args.expected_width or profile.get("height") != args.expected_height:
                            errors.append(f"{path}: {cam_name} profile={profile}")

            if len(errors) >= args.max_errors:
                warnings.append(f"stopped collecting errors at max_errors={args.max_errors}")
                break

        expected_pairs = {
            (episode, env_id)
            for episode in range(1, args.expected_episodes_per_shard + 1)
            for env_id in range(args.expected_envs_per_shard)
        }
        missing_pairs = sorted(expected_pairs - seen_pairs)
        extra_pairs = sorted(seen_pairs - expected_pairs)
        if missing_pairs:
            errors.append(f"{run_dir}: missing session pairs count={len(missing_pairs)}, first={missing_pairs[:10]}")
        if extra_pairs:
            errors.append(f"{run_dir}: extra session pairs count={len(extra_pairs)}, first={extra_pairs[:10]}")

        shard_stats["episodes"] = [min(shard_stats["episodes"] or [0]), max(shard_stats["episodes"] or [0]), len(shard_stats["episodes"])]
        shard_stats["envs"] = sorted(shard_stats["envs"])
        if len(errors) >= args.max_errors:
            break

    if stats["sessions"] != args.expected_total_env_sessions:
        errors.append(f"expected total sessions {args.expected_total_env_sessions}, got {stats['sessions']}")

    if stats["depth_min_mm"] == 2**31 - 1:
        stats["depth_min_mm"] = None
    stats["frame_count_histogram"] = dict(stats["frame_count_histogram"])
    stats["csv_row_ranges"] = dict(stats["csv_row_ranges"])
    stats["audit_finished_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats["errors"] = errors[: args.max_errors]
    stats["warnings"] = warnings
    stats["passed"] = not errors

    contact_sheets = make_contact_sheets(run_dirs, report_dir, args.run_prefix) if args.sample_contact_sheets else []
    stats["contact_sheets"] = contact_sheets

    report_json = report_dir / "audit_report.json"
    report_txt = report_dir / "audit_summary.txt"
    report_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"Franka collection audit: {args.run_prefix}",
        f"passed: {stats['passed']}",
        f"sessions: {stats['sessions']} / {args.expected_total_env_sessions}",
        f"frames: {stats['frames']}",
        f"frame_count_histogram: {stats['frame_count_histogram']}",
        f"rgb_images_checked: {stats['rgb_images_checked']}",
        f"depth_images_checked: {stats['depth_images_checked']}",
        f"depth_min_mm: {stats['depth_min_mm']}",
        f"depth_max_mm: {stats['depth_max_mm']}",
        f"depth_zero_pixels: {stats['depth_zero_pixels']}",
        f"depth_saturated_pixels: {stats['depth_saturated_pixels']}",
        f"csv_row_ranges: {stats['csv_row_ranges']}",
        f"contact_sheets: {contact_sheets}",
        f"report_json: {report_json}",
        f"errors: {len(errors)}",
    ]
    for error in errors[:50]:
        lines.append(f"ERROR: {error}")
    for warning in warnings:
        lines.append(f"WARNING: {warning}")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

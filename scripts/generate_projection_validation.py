#!/usr/bin/env python3
"""Generate projection validation figures for LC RLDS fields."""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuser.camera import get_camera_params  # noqa: E402


DATASET_NAME = "transvtla_annotated_rlds"
OBS = "steps/observation"


def project_world_to_image(
    xyz: np.ndarray,
    image_size_orig: tuple[int, int] = (480, 848),
    image_size_resize: tuple[int, int] = (672, 672),
) -> dict[str, np.ndarray]:
    """Project base/world-frame xyz using the same convention as fuser.contrastive."""

    camera = get_camera_params("world_camera")
    K = camera.K.detach().cpu().numpy().astype(np.float64)
    R = camera.R.detach().cpu().numpy().astype(np.float64)
    t = camera.t.detach().cpu().numpy().astype(np.float64)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)

    scale_x = image_size_resize[1] / image_size_orig[1]
    scale_y = image_size_resize[0] / image_size_orig[0]
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    R_world_to_cam = R.T
    t_world_to_cam = -R_world_to_cam @ t
    xyz_cam = xyz @ R_world_to_cam.T + t_world_to_cam
    uvw = xyz_cam @ K_scaled.T
    z = uvw[:, 2]
    xy = uvw[:, :2] / (z[:, None] + 1e-6)
    valid = (
        (z > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image_size_resize[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image_size_resize[0])
    )
    return {"xy_672": xy.astype(np.float32), "z": z.astype(np.float32), "valid": valid}


def get_float(features: Any, name: str) -> np.ndarray:
    return np.asarray(features[f"{OBS}/{name}"].float_list.value, dtype=np.float32)


def get_int(features: Any, name: str) -> np.ndarray:
    return np.asarray(features[f"{OBS}/{name}"].int64_list.value, dtype=np.int64)


def decode_image(features: Any, index: int) -> Image.Image:
    data = features[f"{OBS}/primary_image"].bytes_list.value[index]
    return Image.open(BytesIO(data)).convert("RGB")


def choose_timestep(valid: np.ndarray, current: np.ndarray, contact_time: np.ndarray) -> int:
    causal_valid = valid & (contact_time <= current[:, None]) & (contact_time >= 0)
    counts = causal_valid.sum(axis=1)
    if counts.max() > 0:
        return int(np.argmax(counts))
    return int(len(current) // 2)


def draw_cross(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: tuple[int, int, int], r: int = 5) -> None:
    x, y = xy
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)


def analyze_example(raw: bytes, tf: Any, out_dir: Path, sample_id: int, point_sample: int) -> dict[str, Any] | None:
    ex = tf.train.Example.FromString(raw)
    f = ex.features.feature
    required = [
        f"{OBS}/primary_image",
        f"{OBS}/point_cloud",
        f"{OBS}/gripper_xyz",
        f"{OBS}/contact_memory",
        f"{OBS}/valid_contact_mask",
        f"{OBS}/contact_type",
        f"{OBS}/contact_reliability",
        f"{OBS}/contact_time",
        f"{OBS}/current_timestep",
    ]
    if any(key not in f for key in required):
        return None
    current = get_int(f, "current_timestep")
    n = int(current.size)
    if n == 0:
        return None
    point_cloud = get_float(f, "point_cloud").reshape(n, -1, 3)
    gripper_xyz = get_float(f, "gripper_xyz").reshape(n, 3)
    contact_memory = get_float(f, "contact_memory").reshape(n, 6, 3)
    valid_mask = get_int(f, "valid_contact_mask").reshape(n, 6).astype(bool)
    contact_type = get_int(f, "contact_type").reshape(n, 6)
    reliability = get_float(f, "contact_reliability").reshape(n, 6)
    contact_time = get_int(f, "contact_time").reshape(n, 6)
    t = choose_timestep(valid_mask, current, contact_time)

    image = decode_image(f, t)
    draw = ImageDraw.Draw(image, "RGBA")
    scale_to_image = image.size[0] / 672.0

    pc = point_cloud[t]
    pc_indices = np.linspace(0, pc.shape[0] - 1, min(point_sample, pc.shape[0]), dtype=np.int64)
    pc_proj = project_world_to_image(pc[pc_indices])
    pc_xy_img = pc_proj["xy_672"] * scale_to_image
    for xy, ok in zip(pc_xy_img, pc_proj["valid"]):
        if ok:
            x, y = float(xy[0]), float(xy[1])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 120, 255, 90))

    gripper_proj = project_world_to_image(gripper_xyz[t : t + 1])
    if bool(gripper_proj["valid"][0]):
        xy = tuple((gripper_proj["xy_672"][0] * scale_to_image).astype(float).tolist())
        draw_cross(draw, xy, (255, 0, 0, 255), r=6)

    causal_contact = valid_mask[t] & (contact_time[t] <= current[t]) & (contact_time[t] >= 0)
    contact_proj = project_world_to_image(contact_memory[t])
    for slot in range(6):
        if not causal_contact[slot] or not bool(contact_proj["valid"][slot]):
            continue
        xy = tuple((contact_proj["xy_672"][slot] * scale_to_image).astype(float).tolist())
        color = (255, 180, 0, 255) if int(contact_type[t, slot]) == 0 else (0, 220, 80, 255)
        draw_cross(draw, xy, color, r=4)
        draw.text((xy[0] + 4, xy[1] + 4), f"{slot}:{reliability[t, slot]:.2f}", fill=color)

    draw.text((6, 6), f"sample {sample_id} t={t} valid={int(causal_contact.sum())}", fill=(255, 255, 255, 255))
    png_path = out_dir / f"projection_sample_{sample_id:02d}.png"
    image.save(png_path)

    return {
        "sample_id": sample_id,
        "image_path": str(png_path),
        "timestep": t,
        "episode_steps": n,
        "pointcloud": {
            "status": "fallback_sampled_point_cloud",
            "sampled_points": int(len(pc_indices)),
            "valid_projected": int(np.count_nonzero(pc_proj["valid"])),
            "valid_ratio": float(np.mean(pc_proj["valid"])) if pc_proj["valid"].size else 0.0,
        },
        "gripper_xyz": {
            "xyz": gripper_xyz[t].astype(float).tolist(),
            "xy_672": gripper_proj["xy_672"][0].astype(float).tolist(),
            "valid": bool(gripper_proj["valid"][0]),
            "depth": float(gripper_proj["z"][0]),
        },
        "contact_memory": [
            {
                "slot": slot,
                "xyz": contact_memory[t, slot].astype(float).tolist(),
                "valid_contact_mask": bool(valid_mask[t, slot]),
                "causal_visible": bool(causal_contact[slot]),
                "contact_type": int(contact_type[t, slot]),
                "contact_reliability": float(reliability[t, slot]),
                "contact_time": int(contact_time[t, slot]),
                "xy_672": contact_proj["xy_672"][slot].astype(float).tolist(),
                "projection_valid": bool(contact_proj["valid"][slot]),
                "depth": float(contact_proj["z"][slot]),
            }
            for slot in range(6)
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Projection Validation",
        "",
        f"Dataset: `{payload['dataset']}`",
        f"Projection camera: `{payload['projection_camera']}`",
        f"point_cloud_centers_status: `{payload['point_cloud_centers_status']}`",
        "",
        "Images:",
    ]
    for sample in payload["samples"]:
        lines.append(
            f"- `{sample['image_path']}`: t={sample['timestep']}, "
            f"pc_valid_ratio={sample['pointcloud']['valid_ratio']:.3f}, "
            f"gripper_valid={sample['gripper_xyz']['valid']}"
        )
    lines += [
        "",
        "Color legend: blue=sampled point_cloud, red=gripper_xyz, yellow=active contact, green=grasp contact.",
        "Coordinates are base/world-frame xyz projected with the same world_camera K/R/t convention used by MLA fuser.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.3.0_candidate"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/media/files/TransVTLA-RealDataCollect/debug/projection_validation"))
    parser.add_argument("--report-dir", type=Path, default=Path("/media/files/TransVTLA-RealDataCollect/reports"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--point-sample", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import tensorflow as tf

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    shards = sorted(args.version_dir.glob(f"{DATASET_NAME}-train.tfrecord-*"))
    for shard in shards:
        for raw in tf.compat.v1.io.tf_record_iterator(str(shard)):
            result = analyze_example(raw, tf, args.out_dir, len(samples), args.point_sample)
            if result is not None:
                samples.append(result)
            if len(samples) >= args.num_samples:
                break
        if len(samples) >= args.num_samples:
            break
    payload = {
        "dataset": str(args.version_dir),
        "projection_camera": "world_camera",
        "projection_frame_expected": "robot_base / TransVTLA world-base xyz in meters",
        "point_cloud_centers_status": "fallback_sampled_point_cloud",
        "samples": samples,
    }
    json_path = args.report_dir / "projection_validation.json"
    md_path = args.report_dir / "projection_validation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"samples={len(samples)}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())

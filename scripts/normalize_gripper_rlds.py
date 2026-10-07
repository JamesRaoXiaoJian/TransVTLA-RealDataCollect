#!/usr/bin/env python3
"""Rewrite a TransVTLA RLDS/TFDS dataset with normalized gripper values.

The source real dataset stores observation.state[6] as raw gripper position
in [0, 1000], and action[6] as the frame-to-frame gripper position delta.
OpenVLA EEF_POS treats the final action dimension as an absolute gripper
command, so this converter writes:

  observation.state[6] = clip(raw_pos / 1000, 0, 1)
  action[6]            = next normalized gripper position

The original dataset directory is read-only from this script's perspective.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow_datasets as tfds


TACTILE_SHAPE = (16, 20)


def decode_text(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: decode_text(v) for k, v in value.items()}
    return value


def normalize_raw_gripper(raw: np.ndarray | float, scale: float) -> np.ndarray | float:
    return np.clip(np.asarray(raw, dtype=np.float32) / float(scale), 0.0, 1.0)


def clone_step(step: dict[str, Any]) -> dict[str, Any]:
    observation = step["observation"]
    return {
        "observation": {
            "primary_image": observation["primary_image"],
            "wrist_image": observation["wrist_image"],
            "primary_depth": observation["primary_depth"],
            "wrist_depth": observation["wrist_depth"],
            "state": np.asarray(observation["state"], dtype=np.float32).copy(),
            "tactile": np.asarray(observation["tactile"], dtype=np.float32),
        },
        "action": np.asarray(step["action"], dtype=np.float32).copy(),
        "reward": np.float32(step["reward"]),
        "is_first": bool(step["is_first"]),
        "is_last": bool(step["is_last"]),
        "is_terminal": bool(step["is_terminal"]),
        "language_instruction": decode_text(step["language_instruction"]),
        "action_prompt": decode_text(step["action_prompt"]),
        "object_type": decode_text(step["object_type"]),
        "object_label": decode_text(step["object_label"]),
        "item_type": decode_text(step["item_type"]),
        "material": decode_text(step["material"]),
        "phase": decode_text(step["phase"]),
        "stage_label": np.int64(step["stage_label"]),
        "use_tactile": bool(step["use_tactile"]),
        "fragility": np.float32(step["fragility"]),
        "slip_risk": np.float32(step["slip_risk"]),
        "depth_uncertainty": np.float32(step["depth_uncertainty"]),
        "force_min": np.float32(step["force_min"]),
        "force_max": np.float32(step["force_max"]),
    }


def transform_steps(source_steps: Iterable[dict[str, Any]], gripper_scale: float) -> list[dict[str, Any]]:
    steps = [clone_step(step) for step in source_steps]
    if not steps:
        return steps

    raw_gripper = np.asarray([step["observation"]["state"][6] for step in steps], dtype=np.float32)
    normalized_gripper = normalize_raw_gripper(raw_gripper, gripper_scale).astype(np.float32)

    for idx, step in enumerate(steps):
        step["observation"]["state"][6] = normalized_gripper[idx]
        target_idx = min(idx + 1, len(steps) - 1)
        step["action"][6] = normalized_gripper[target_idx]

    return steps


def make_features(image_size: int) -> tfds.features.FeaturesDict:
    return tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "observation": tfds.features.FeaturesDict({
                "primary_image": tfds.features.Image(shape=(image_size, image_size, 3), dtype=np.uint8),
                "wrist_image": tfds.features.Image(shape=(image_size, image_size, 3), dtype=np.uint8),
                "primary_depth": tfds.features.Tensor(shape=(image_size, image_size, 1), dtype=np.uint16),
                "wrist_depth": tfds.features.Tensor(shape=(image_size, image_size, 1), dtype=np.uint16),
                "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                "tactile": tfds.features.Tensor(shape=TACTILE_SHAPE, dtype=np.float32),
            }),
            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "reward": tfds.features.Scalar(dtype=np.float32),
            "is_first": tfds.features.Scalar(dtype=np.bool_),
            "is_last": tfds.features.Scalar(dtype=np.bool_),
            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
            "language_instruction": tfds.features.Text(),
            "action_prompt": tfds.features.Text(),
            "object_type": tfds.features.Text(),
            "object_label": tfds.features.Text(),
            "item_type": tfds.features.Text(),
            "material": tfds.features.Text(),
            "phase": tfds.features.Text(),
            "stage_label": tfds.features.Scalar(dtype=np.int64),
            "use_tactile": tfds.features.Scalar(dtype=np.bool_),
            "fragility": tfds.features.Scalar(dtype=np.float32),
            "slip_risk": tfds.features.Scalar(dtype=np.float32),
            "depth_uncertainty": tfds.features.Scalar(dtype=np.float32),
            "force_min": tfds.features.Scalar(dtype=np.float32),
            "force_max": tfds.features.Scalar(dtype=np.float32),
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "episode_id": tfds.features.Scalar(dtype=np.int64),
            "source_file": tfds.features.Text(),
            "annotation_key": tfds.features.Text(),
            "data_source": tfds.features.Text(),
            "task_id": tfds.features.Text(),
            "split": tfds.features.Text(),
            "slot_pair": tfds.features.Text(),
            "success": tfds.features.Scalar(dtype=np.bool_),
            "needs_review": tfds.features.Scalar(dtype=np.bool_),
            "review_reasons": tfds.features.Text(),
        }),
    })


class TransvtlaAnnotatedRlds(tfds.core.GeneratorBasedBuilder):
    """Same TFDS builder name as the source dataset for OpenVLA compatibility."""

    BUILDER_CONFIGS = []

    def __init__(
        self,
        *,
        data_dir: str | Path,
        source_builder: tfds.core.DatasetBuilder,
        version: str,
        image_size: int,
        gripper_scale: float,
        max_episodes: int | None,
        **kwargs: Any,
    ) -> None:
        self.VERSION = tfds.core.Version(version)
        self.source_builder = source_builder
        self.image_size = image_size
        self.gripper_scale = gripper_scale
        self.max_episodes = max_episodes
        super().__init__(data_dir=data_dir, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "TransVTLA annotated RLDS dataset with gripper state/action "
                "normalized from raw [0,1000] position to [0,1]."
            ),
            features=make_features(self.image_size),
        )

    def _split_generators(self, dl_manager: Any) -> dict[str, Iterable[tuple[str, dict[str, Any]]]]:
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterable[tuple[str, dict[str, Any]]]:
        ds = self.source_builder.as_dataset(split="train", shuffle_files=False)
        for ordinal, episode in enumerate(tfds.as_numpy(ds)):
            if self.max_episodes is not None and ordinal >= self.max_episodes:
                break
            metadata = decode_text(episode["episode_metadata"])
            steps = transform_steps(episode["steps"], self.gripper_scale)
            key = f"transvtla_gripper01_{ordinal:06d}"
            yield key, {
                "steps": steps,
                "episode_metadata": metadata,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0"),
        help="Path to source TFDS builder directory, e.g. .../transvtla_annotated_rlds/3.0.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_gripper01_strict"),
        help="Output TFDS data_dir. The builder subdir transvtla_annotated_rlds/3.0.0 is created under it.",
    )
    parser.add_argument("--version", default="3.0.0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--gripper-scale", type=float, default=1000.0)
    parser.add_argument("--max-episodes", type=int, default=None, help="Debug option for smoke builds.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_dir.is_dir():
        raise SystemExit(f"Source builder directory does not exist: {args.source_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory already exists and is non-empty: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_builder = tfds.builder_from_directory(str(args.source_dir))
    builder = TransvtlaAnnotatedRlds(
        data_dir=args.output_dir,
        source_builder=source_builder,
        version=args.version,
        image_size=args.image_size,
        gripper_scale=args.gripper_scale,
        max_episodes=args.max_episodes,
    )
    print(f"source={args.source_dir}")
    print(f"output={args.output_dir}")
    print(f"version={args.version} gripper_scale={args.gripper_scale}")
    builder.download_and_prepare()
    print(f"done: {builder.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

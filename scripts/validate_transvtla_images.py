#!/usr/bin/env python3
"""Decode every RGB image payload in a TransVTLA TFDS/RLDS directory."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


DEFAULT_VERSION_DIR = Path(
    "/media/files/TransVTLA-RealDataCollect/dataset/"
    "rlds_real1000_gripper01_strict/transvtla_annotated_rlds/3.0.0"
)

IMAGE_FIELDS = [
    "steps/observation/primary_image",
    "steps/observation/wrist_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-dir", type=Path, default=DEFAULT_VERSION_DIR)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shards = sorted(args.version_dir.glob("transvtla_annotated_rlds-train.tfrecord-*-of-*"))
    if not shards:
        raise SystemExit(f"No TFRecord shards found: {args.version_dir}")

    counts: Counter[str] = Counter()
    errors: list[str] = []

    for shard_idx, shard in enumerate(shards, start=1):
        shard_examples = 0
        for raw in tf.compat.v1.io.tf_record_iterator(str(shard)):
            shard_examples += 1
            counts["examples"] += 1
            example = tf.train.Example.FromString(raw)
            features = example.features.feature
            for key in IMAGE_FIELDS:
                if key not in features:
                    errors.append(f"{shard.name} example {shard_examples}: missing {key}")
                    continue
                payloads = features[key].bytes_list.value
                for payload_idx, payload in enumerate(payloads):
                    counts[key] += 1
                    try:
                        image = tf.io.decode_image(payload, channels=3, expand_animations=False)
                        shape = image.shape.as_list()
                        if shape != [224, 224, 3]:
                            errors.append(
                                f"{shard.name} example {shard_examples}: "
                                f"{key}[{payload_idx}] shape={shape}, expected=[224, 224, 3]"
                            )
                    except Exception as exc:  # noqa: BLE001 - retain exact decode failure.
                        errors.append(f"{shard.name} example {shard_examples}: {key}[{payload_idx}] decode failed: {exc}")

        if shard_idx == 1 or shard_idx % args.progress_every == 0 or shard_idx == len(shards):
            print(
                f"[{shard_idx:04d}/{len(shards):04d}] examples={counts['examples']} "
                f"primary={counts['steps/observation/primary_image']} "
                f"wrist={counts['steps/observation/wrist_image']} errors={len(errors)}",
                flush=True,
            )

    summary = {
        "version_dir": str(args.version_dir),
        "num_shards": len(shards),
        "num_examples": counts["examples"],
        "decoded_primary_images": counts["steps/observation/primary_image"],
        "decoded_wrist_images": counts["steps/observation/wrist_image"],
        "num_errors": len(errors),
        "errors": errors[:200],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        print(f"IMAGE_VALIDATION_FAILED errors={len(errors)}")
        return 1
    print("IMAGE_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

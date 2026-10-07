# RLDS Semantic Annotation Patch Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely replace four stale semantic annotation features in two existing RLDS versions while preserving every sensor, point-cloud, state, action, and version-specific feature.

**Architecture:** A new standalone patcher reads the strict manifest, matches examples by episode ID plus two identity assertions, and streams each TFRecord into a sibling candidate. Verification hashes every immutable feature and validates semantic distributions. A separate recoverable publication command swaps both verified candidates into place using a durable journal.

**Tech Stack:** Python 3.11.15, TensorFlow `tf.train.Example` and `TFRecordWriter`, CSV/JSON, pytest.

---

## Chunk 1: Semantic patch core

### Task 1: Manifest mapping and strict prompt transformation

**Files:**
- Create: `scripts/patch_rlds_semantic_annotations.py`
- Create: `tests/test_patch_rlds_semantic_annotations.py`

- [ ] Write failing tests for strict manifest selection; exactly 1000 unique ordinals with no missing/extra IDs; triple `episode_id`/`annotation_key`/`source_file` mismatch rejection; exact per-step instruction/prompt value counts; exact prompt-prefix validation; suffix preservation; malformed prompt rejection; and canonical task/instruction/slot distributions.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_patch_rlds_semantic_annotations.py` and confirm failure because the patcher does not exist.
- [ ] Implement the minimal pure functions for manifest loading and four-feature patching.
- [ ] Re-run the targeted tests and confirm they pass.

### Task 2: Immutable-feature verification and shard metadata

**Files:**
- Modify: `scripts/patch_rlds_semantic_annotations.py`
- Modify: `tests/test_patch_rlds_semantic_annotations.py`

- [ ] Write failing tests that compare source/candidate examples, allow only the four semantic features to differ, detect added/removed/changed immutable fields, preserve version-specific features and metadata, preserve shard filenames/order/counts, update only dataset byte counts/shard lengths, refuse non-empty candidates, and prove the source is unchanged after every candidate-generation failure.
- [ ] Confirm the new tests fail for the expected missing behavior.
- [ ] Implement streaming comparison, coverage/distribution summaries, metadata copying, and stale-statistics exclusion.
- [ ] Re-run all patcher tests.

## Chunk 2: Recoverable publication

### Task 3: Durable two-version publication transaction

**Files:**
- Modify: `scripts/patch_rlds_semantic_annotations.py`
- Modify: `tests/test_patch_rlds_semantic_annotations.py`

- [ ] Write failing temporary-directory tests covering: same-filesystem/no-alias/backup-absent preflight; journal-file and journal-parent fsync; affected-parent fsync after every rename and before commit; simulated termination at every rename/journal boundary; recovery before every other subcommand; recovery based on recorded identities rather than path existence; reverse-order rollback; rollback-failure hard error; and committed post-swap identity behavior.
- [ ] Confirm failure for missing transaction behavior.
- [ ] Implement candidate generation and publication as separate commands; require explicit `--publish` for renames.
- [ ] Re-run targeted tests, then run the exact full suite: `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/transvtla-matplotlib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python -m pytest -q tests`. Expected: exit 0 and all tests pass; any unrelated failure is reported and resolved or explicitly surfaced before dataset execution.

## Chunk 3: Dataset execution

### Task 4: Small real-data candidate validation

**Files:**
- Modify: `scripts/patch_rlds_semantic_annotations.py`
- Modify: `tests/test_patch_rlds_semantic_annotations.py`

- [ ] Write failing tests for a low-level `test-one-shard` subcommand that accepts copied shard paths, emits no publishable `dataset_info.json`, cannot call publication code, and cannot weaken the 1000-ID requirement of `patch`, `verify`, or `publish`. Run the targeted test and confirm the expected missing-subcommand failure.
- [ ] Implement only that test harness and re-run the targeted test to green.
- [ ] Copy exact test inputs: `cp dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0/transvtla_annotated_rlds-train.tfrecord-00000-of-00512 /tmp/rlds_semantic_patch_source_3.0.0.tfrecord` and `cp dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05/transvtla_annotated_rlds-train.tfrecord-00000-of-00512 /tmp/rlds_semantic_patch_source_3.4.0.tfrecord`.
- [ ] Run for 3.0.0:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py test-one-shard --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-shard /tmp/rlds_semantic_patch_source_3.0.0.tfrecord --output-shard /tmp/rlds_semantic_patch_3.0.0.tfrecord --report /tmp/rlds_semantic_patch_3.0.0.json`
- [ ] Run for 3.4.0:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py test-one-shard --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-shard /tmp/rlds_semantic_patch_source_3.4.0.tfrecord --output-shard /tmp/rlds_semantic_patch_3.4.0.tfrecord --report /tmp/rlds_semantic_patch_3.4.0.json`
- [ ] Expected reports: `identity_mismatches=0`, `unexpected_changed_features=0`, and only the four allowlisted feature names; 3.4.0 point-cloud/contact-memory hashes unchanged.

### Task 5: Full candidate generation and verification

**Files:**
- Output: `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.annotation_patch_candidate_20260713`
- Output: `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.annotation_patch_candidate_20260713`
- Report: `dataset/logs/rlds_semantic_patch_3.0.0_20260713.json`
- Report: `dataset/logs/rlds_semantic_patch_3.4.0_20260713.json`
- Log: `dataset/logs/rlds_semantic_patch_full_20260713.log`

- [ ] Write failing tests for `patch-batch` and `wait-status`: the batch must run jobs sequentially, atomically write and fsync a durable JSON status with captured failure/success, and never report complete if a job fails; `wait-status` must poll the actual state and return the recorded exit code. Run red, implement minimally, and run green.
- [ ] Launch one exact batch command in `screen`:
  `screen -dmS rlds_semantic_patch_full bash -lc 'exec /home/james/.venvs/TransVTLA-RealDataCollect/bin/python /media/files/TransVTLA-RealDataCollect/scripts/patch_rlds_semantic_annotations.py patch-batch --manifest /media/files/TransVTLA-RealDataCollect/dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --status /media/files/TransVTLA-RealDataCollect/dataset/logs/rlds_semantic_patch_full_20260713.status.json --job /media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0,/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.annotation_patch_candidate_20260713,/media/files/TransVTLA-RealDataCollect/dataset/logs/rlds_semantic_patch_3.0.0_20260713.json --job /media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05,/media/files/TransVTLA-RealDataCollect/dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.annotation_patch_candidate_20260713,/media/files/TransVTLA-RealDataCollect/dataset/logs/rlds_semantic_patch_3.4.0_20260713.json > /media/files/TransVTLA-RealDataCollect/dataset/logs/rlds_semantic_patch_full_20260713.log 2>&1'`
- [ ] Gate verification on the durable condition, not elapsed time or screen disappearance:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py wait-status --status dataset/logs/rlds_semantic_patch_full_20260713.status.json --require complete --poll-seconds 5 --timeout-seconds 21600`. Expected: exit 0 and `PATCH_BATCH_COMPLETE jobs=2`; a recorded failed state exits nonzero immediately with the job error.
- [ ] Verify 3.0.0 exactly:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py verify --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0 --candidate-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.annotation_patch_candidate_20260713 --report dataset/logs/rlds_semantic_patch_3.0.0_20260713.json`
- [ ] Verify 3.4.0 exactly:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py verify --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05 --candidate-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.annotation_patch_candidate_20260713 --report dataset/logs/rlds_semantic_patch_3.4.0_20260713.json`
- [ ] Expected for both: `examples=1000`, `unique_episode_ids=1000`, `missing_ids=[]`, `extra_ids=[]`, `identity_mismatches=0`, `unexpected_changed_features=0`, shard filenames/order/counts preserved, and task/instruction/slot distributions equal the manifest.

### Task 6: Publish both versions transactionally

**Files:**
- Original version paths become verified patched datasets.
- Timestamped backup paths retain both originals.

- Backup: `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.pre_semantic_patch_20260713`
- Backup: `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.pre_semantic_patch_20260713`
- Journal: `dataset/rlds_real1000_strict/transvtla_annotated_rlds/.semantic_annotation_patch_transaction_20260713.json`

- [ ] Run exact preflight:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py publish --preflight-only --journal dataset/rlds_real1000_strict/transvtla_annotated_rlds/.semantic_annotation_patch_transaction_20260713.json --swap dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0,dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.annotation_patch_candidate_20260713,dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.pre_semantic_patch_20260713,dataset/logs/rlds_semantic_patch_3.0.0_20260713.json --swap dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05,dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.annotation_patch_candidate_20260713,dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.pre_semantic_patch_20260713,dataset/logs/rlds_semantic_patch_3.4.0_20260713.json`. Expected: exit 0 and `PREFLIGHT_OK versions=2`.
- [ ] Run the identical command without `--preflight-only`. Expected: exit 0 and `PUBLISH_COMMITTED versions=2` only after both post-swap identity checks.
- [ ] Verify published 3.0.0 against its exact backup:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py verify --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0.pre_semantic_patch_20260713 --candidate-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0 --report dataset/logs/rlds_semantic_patch_3.0.0_postpublish_20260713.json`
- [ ] Verify published 3.4.0 against its exact backup:
  `CUDA_VISIBLE_DEVICES=-1 /home/james/.venvs/TransVTLA-RealDataCollect/bin/python scripts/patch_rlds_semantic_annotations.py verify --manifest dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv --source-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05.pre_semantic_patch_20260713 --candidate-version dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05 --report dataset/logs/rlds_semantic_patch_3.4.0_postpublish_20260713.json`
- [ ] Confirm the exact journal is committed and records both final identities:
  `/home/james/.venvs/TransVTLA-RealDataCollect/bin/python -c 'import json; from pathlib import Path; p=Path("dataset/rlds_real1000_strict/transvtla_annotated_rlds/.semantic_annotation_patch_transaction_20260713.json"); d=json.loads(p.read_text()); assert d["state"] == "committed"; assert len(d["swaps"]) == 2; assert all(x.get("post_swap_identity_verified") for x in d["swaps"]); print("JOURNAL_COMMITTED versions=2")'`. Expected: exit 0 and the printed confirmation.

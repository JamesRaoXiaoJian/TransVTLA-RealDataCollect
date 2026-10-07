# RLDS Semantic Annotation Patch Design

## Goal

Replace stale semantic language annotations in two existing RLDS versions without regenerating or modifying images, depth, point clouds, contact memory, tactile data, robot state, or actions.

Targets:

- `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.0.0`
- `dataset/rlds_real1000_strict/transvtla_annotated_rlds/3.4.0_candidate_shift2_scale05`

Canonical annotation source:

- `dataset/auto_annotations_qwen3vl_thirdpass/final_annotations/real_1000_train_view_strict/manifest.csv`

## Matching contract

The manifest's selected-row ordinal is the authoritative `episode_id`. Before changing an example, the patcher must require all three values to agree:

- `episode_metadata/episode_id`
- `episode_metadata/annotation_key`
- `episode_metadata/source_file`

Every target must contain exactly 1000 unique episode IDs, with no missing or extra IDs. A mismatch aborts publication and leaves the source untouched.

## Mutation allowlist

Only these serialized TF Example features may change:

- `steps/language_instruction`
- `steps/action_prompt`
- `episode_metadata/task_id`
- `episode_metadata/slot_pair`

The instruction is repeated once per existing target step. Before patching, the patcher must assert that `language_instruction` and `action_prompt` each contain exactly one value per target step, and that every old prompt has exactly the form `What action should the robot take to execute: <old instruction>\n<non-empty CTX suffix>`. It then preserves the suffix byte-for-byte and replaces only the validated instruction prefix. Embedded newlines, missing prefixes, empty suffixes, or step-count mismatches abort the candidate. All other feature payloads must remain byte-for-byte equal after accounting for protobuf serialization order.

## Data flow

1. Load and filter the strict manifest with the same rules as the RLDS builder.
2. Construct the expected metadata and semantic values keyed by ordinal episode ID.
3. Stream each source TFRecord shard without image decoding.
4. Validate the triple match, patch the four allowlisted features, and write a sibling candidate tree.
5. Preserve `features.json` and all version-specific extra features. Copy metadata, update `dataset_info.json` byte counts, and omit stale dataset statistics caches.
6. Verify example count, unique coverage, task/slot distribution, instruction values, and immutable-feature hashes against the exact candidate paths that will be published.
7. Preflight publication: both candidates and sources must exist, all four paths must be on the same filesystem, backup paths must not exist, candidate identity summaries must match the verified summaries, and no candidate/source path may alias another.
8. Publish as a recoverable two-version rename transaction. Before the first rename, create a durable JSON journal containing the exact source, candidate, backup, expected identity, and initial state for both versions; fsync the journal file and its parent directory. Before and after every rename, update and fsync the journal state. After each rename, fsync every parent directory whose entries changed before recording that rename as complete. Success is declared only after both swaps and post-swap identity checks complete, all affected parent directories are fsynced again, and then the committed journal state is written and fsynced. If any caught rename or identity check fails, run the inverse operations in reverse order to restore both original paths and record the rollback durably; surface a hard error if rollback itself cannot restore the preflight state.
9. On every invocation, detect an uncommitted publication journal before doing any other work. Recover idempotently from the recorded state by inspecting all source/candidate/backup identities, completing rollback to the preflight state, fsyncing affected parent directories, and only then allowing a new patch or publication. Never infer recovery from path existence alone.

## Safety and failure behavior

- Never write inside a source version directory.
- Refuse non-empty output directories.
- Keep candidates on any mismatch or failure for inspection.
- Do not publish unless both versions pass full verification.
- Do not begin publication unless the complete two-version transaction preflight passes.
- Roll back every completed rename if either version cannot be published and verified at its original path.
- Make publication restart-safe with a durable, fsynced transaction journal and idempotent startup rollback after SIGKILL or host restart.
- Keep original directories as backups after publication.

## Expected cost

This avoids RGB/depth decoding and point-cloud regeneration, but TFRecord framing requires every source record to be read and rewritten. For roughly 200 GB across both versions, expected elapsed time is 2–4 hours under current shared-disk load.

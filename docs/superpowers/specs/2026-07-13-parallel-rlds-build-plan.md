# Parallel RLDS Build Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce real-data RLDS build time without changing episode content, order, keys, or TFDS metadata.

**Architecture:** Worker processes build complete episodes from independent source sessions. A bounded ordered iterator returns their results to the existing single TFDS writer, which remains the only process that creates shards and metadata. The queue is bounded so full image arrays cannot exhaust memory.

**Tech Stack:** Python stdlib `concurrent.futures`, NumPy, OpenCV, TensorFlow Datasets, pytest.

---

## Chunk 1: Ordered bounded episode preparation

### Task 1: Define and test the worker-result ordering contract

**Files:**
- Modify: `tests/test_contact_memory_generation.py`
- Modify: `scripts/build_annotated_rlds.py`

- [ ] **Step 1: Write the failing test**
  Add a test that submits deliberately out-of-order worker completions and asserts the helper yields episode results in input order while preserving failures.
- [ ] **Step 2: Run test to verify it fails**
  Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_contact_memory_generation.py`
  Expected: FAIL because the ordering helper does not exist.
- [ ] **Step 3: Write minimal implementation**
  Add a module-level bounded process-pool iterator whose workers construct `EpisodeBuildResult`, and whose caller consumes results in ordinal order.
- [ ] **Step 4: Run test to verify it passes**
  Run the same command. Expected: PASS.

### Task 2: Expose conservative build controls

**Files:**
- Modify: `scripts/build_annotated_rlds.py`
- Test: `tests/test_contact_memory_generation.py`

- [ ] **Step 1: Write the failing test**
  Assert `--workers` and `--prefetch-episodes` parse with safe defaults and reject invalid values.
- [ ] **Step 2: Run test to verify it fails**
  Run the targeted pytest command. Expected: FAIL due to missing arguments/validation.
- [ ] **Step 3: Write minimal implementation**
  Add CLI options. Default to a bounded worker count; retain `--workers 1` as the serial-equivalent path.
- [ ] **Step 4: Run test to verify it passes**
  Run targeted tests. Expected: PASS.

## Chunk 2: Integration validation and persistent build

### Task 3: Verify output equivalence on a small subset

**Files:**
- No production-file changes beyond Tasks 1–2.

- [ ] **Step 1: Build a small serial reference**
  Run a 2-episode build in a temporary output directory.
- [ ] **Step 2: Build the same subset with workers enabled**
  Run a 2-episode parallel build in a second temporary output directory.
- [ ] **Step 3: Compare episode count, keys, and task-4 instructions**
  Inspect TFDS metadata and iterate the samples to confirm identical keys and labels.

### Task 4: Launch the confirmed candidate build in screen

**Files:**
- Output: `dataset/rlds_real1000_task4confirmed_candidate_pre_gripper_norm/`
- Log: `dataset/logs/rlds_real1000_task4confirmed_build.log`

- [ ] **Step 1: Preserve the interrupted partial output by renaming it**
- [ ] **Step 2: Start the 1000-view build with a bounded worker count in `screen`**
- [ ] **Step 3: Report the screen session and log path**

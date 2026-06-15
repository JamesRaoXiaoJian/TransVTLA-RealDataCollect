#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

TIMESTAMP_WARNING_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T.*\[Warning\] \[([^\]]+)\]")
EXT_STARTUP_RE = re.compile(r"^\[\d+(?:\.\d+)?s\] \[ext: .*\] startup$")
MATRIX_ROW_RE = re.compile(r"^\s*\d+\s+[\d.]+(?:\s+[\d.]+)+\s*$")

NOISY_WARNING_SOURCES = (
    "carb.crashreporter-breakpad.plugin",
    "carb.windowing-glfw.plugin",
    "omni.platforminfo.plugin",
    "gpu.foundation.plugin",
    "omni.usd",
    "isaacsim.core.simulation_manager.impl.simulation_manager",
    "isaacsim.core.experimental.prims.impl.articulation",
    "omni.physx.fabric.plugin",
    "omni.fabric.plugin",
    "usdrt.population.plugin",
    "rtx.postprocessing.plugin",
    "omni.syntheticdata.plugin",
    "omni.graph.core.plugin",
    "omni.replicator.core.scripts.extension",
    "pxr.Semantics",
    "omni.isaac.cortex",
    "omni.isaac.cortex.sample_behaviors",
    "omni.log",
)

NOISY_EXACT_PREFIXES = (
    "Warning: running in conda env, please deactivate before executing this script",
    "If conda is desired please source setup_conda_env.sh",
    "[Info] [carb] Logging to file:",
    "Starting kit application with the following args:",
    "Passing the following args to the base kit application:",
    "Unidirectional P2P=Enabled Bandwidth",
    "P2P=Enabled Latency",
)

NOISY_SUBSTRINGS = (
    "relationship target </Visual_materials/",
    "omni:rtx:material:db:flattener:",
    "GLFW initialization failed",
    "Failed to startup plugin carb.windowing-glfw.plugin",
    "failed to open the default display",
    "CUDA peer-to-peer observed bandwidth",
    "CUDA peer-to-peer observed latency",
    "Please verify if observed bandwidth and latency are expected",
    "DLSS increasing input dimensions",
    "OgnSdPostRenderVarToHost",
    "Simulation App Starting",
    "Simulation App Startup Complete",
    "Simulation App Shutting Down",
    "app ready",
    "Warp DeprecationWarning",
    "CUDA Toolkit",
    "CUDA peer access",
    "Devices:",
    "Not supported",
    "Kernel cache",
    "mempool enabled",
    "load on device",
    ".cache/warp",
    "DOF types mismatch",
    "USD->Fabric: Unhandled array type string[]",
    "Unhandled attribute type VtArray<std::string>",
    "Prototype prims (instancing prototypes) are present",
)

KEEP_SUBSTRINGS = (
    "Traceback",
    "RuntimeError",
    "Exception",
    "[Error]",
    " Error:",
    "ERROR:",
    "Failed episode",
    "Asynchronous data save failed",
)


def is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if any(token in stripped for token in KEEP_SUBSTRINGS):
        return False
    if EXT_STARTUP_RE.match(stripped):
        return True
    if stripped.startswith("[") and "[ext:" in stripped:
        return True
    if stripped.endswith("] startup"):
        return True
    if stripped.startswith("|"):
        return True
    if stripped.startswith('"cpu"') or stripped.startswith('"cuda:'):
        return True
    if stripped in {r"D\D     0      1", "GPU     0      1", "CPU     0      1"}:
        return True
    if MATRIX_ROW_RE.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in NOISY_EXACT_PREFIXES):
        return True
    if any(token in stripped for token in NOISY_SUBSTRINGS):
        return True
    if TIMESTAMP_WARNING_RE.match(stripped):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter repetitive Isaac/Kit startup noise while preserving collector progress and real errors.")
    parser.add_argument("--summary", action="store_true", help="Print a suppression summary when stdin closes.")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    for line in sys.stdin:
        if is_noise(line):
            key = "other"
            match = TIMESTAMP_WARNING_RE.match(line.strip())
            if match:
                key = match.group(1)
            elif EXT_STARTUP_RE.match(line.strip()):
                key = "extension_startup"
            counts[key] += 1
            continue
        sys.stdout.write(line)
        sys.stdout.flush()

    if args.summary and counts:
        total = sum(counts.values())
        sys.stdout.write(f"[log-filter] suppressed {total} repetitive Isaac/Kit log lines.\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

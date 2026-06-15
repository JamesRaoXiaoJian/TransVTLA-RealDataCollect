#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _latest_npz(data_dir: Path) -> Path:
    files = sorted(data_dir.glob("franka_pressure_test_*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found in: {data_dir}")
    return files[-1]


def _phase_spans(events: np.ndarray) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    if len(events) == 0:
        return spans
    start = 0
    current = int(events[0])
    for i in range(1, len(events)):
        e = int(events[i])
        if e != current:
            spans.append((start, i - 1, current))
            start = i
            current = e
    spans.append((start, len(events) - 1, current))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Franka pressure npz data")
    parser.add_argument("--npz", type=Path, default=None, help="Input npz path (default: latest in pressure_test_data)")
    parser.add_argument("--out", type=Path, default=Path("pressure_test_data/pressure_plot.png"), help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    if args.npz is None:
        args.npz = _latest_npz(Path("pressure_test_data"))
    data = np.load(args.npz, allow_pickle=True)

    sensor_names = [str(x) for x in data["sensor_names"]]
    steps = data["steps"].astype(np.int32)
    forces = data["forces"].astype(np.float32)
    contacts = data["contacts"].astype(np.uint8)
    events = data["events"].astype(np.int32) if "events" in data.files else None

    force_norm = np.linalg.norm(forces, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for i, name in enumerate(sensor_names):
        ax1.plot(steps, forces[:, i], linewidth=1.5, label=name)
    ax1.plot(steps, force_norm, color="black", linewidth=2.0, linestyle="--", label="4-point norm")
    ax1.set_ylabel("Force (N)")
    ax1.set_title(f"Franka pressure timeline: {args.npz.name}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", ncol=2, fontsize=9)

    contact_sum = contacts.sum(axis=1)
    ax2.step(steps, contact_sum, where="post", linewidth=1.8, color="tab:purple")
    ax2.set_xlabel("Simulation step")
    ax2.set_ylabel("Contact count (0-4)")
    ax2.grid(alpha=0.3)

    if events is not None and len(events) == len(steps):
        colors = plt.cm.tab10.colors
        for s, e, phase in _phase_spans(events):
            c = colors[phase % len(colors)] if phase >= 0 else (0.7, 0.7, 0.7)
            ax1.axvspan(steps[s], steps[e], color=c, alpha=0.08)
            ax2.axvspan(steps[s], steps[e], color=c, alpha=0.08)
        phase_counts = {int(k): int(v) for k, v in zip(*np.unique(events, return_counts=True))}
        print("phase_counts:", phase_counts)

    print("samples:", len(steps))
    print("peak_by_sensor:", forces.max(axis=0))
    print("global_peak_norm:", float(force_norm.max()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print("saved:", args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

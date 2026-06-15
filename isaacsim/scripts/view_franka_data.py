"""
Franka Pick-and-Place Data Viewer
- 图像展示（当前帧）
- 压力曲线（全程）+ 当前帧标记 + 阶段标注
- Phase 3/4 数值后处理：围绕 lift 起始值水平波动
- 帧率条拖动浏览

Usage:
    python scripts/view_franka_data.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FORCE_CSV = os.path.join(SCRIPT_DIR, "gripper_force_data.csv")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "camera_images")

PHASE_NAMES = [
    "0: Move to cube XY",
    "1: Approach down",
    "2: Close gripper",
    "3: Lift cube",
    "4: Move to target",
    "5: Open gripper",
    "6: Retract up",
]

PHASE_COLORS = {
    0: "#e8e8e8",
    1: "#d0d0ff",
    2: "#ffffb0",
    3: "#b0ffb0",
    4: "#b0d0ff",
    5: "#ffb0b0",
    6: "#e8e8e8",
}


def load_force_data(csv_path):
    steps, phases, grips = [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            phases.append(int(row["phase"]))
            grips.append(float(row["grip"]))
    return np.array(steps), np.array(phases), np.array(grips)


def transform_lift_phases(phases, grips_raw):
    """
    Post-process: force Phase 3 & 4 values to fluctuate around
    the peak grip force (end of Phase 2 / grasp), with small visible
    fluctuations preserved.
    """
    grips = grips_raw.copy().astype(float)

    p34_mask = (phases == 3) | (phases == 4)
    if not np.any(p34_mask):
        return grips, None

    # Peak = max grip in the grasping window (Phase 2–4)
    p234_mask = (phases == 2) | (phases == 3) | (phases == 4)
    peak = np.max(grips_raw[p234_mask])

    # Scale down the fluctuation amplitude so it's visible but small
    raw_p34 = grips_raw[p34_mask]
    p34_mean = np.mean(raw_p34)
    fluctuation = raw_p34 - p34_mean  # deviation from p34 mean
    grips[p34_mask] = peak + fluctuation * 0.3  # subtle wiggle around peak

    return grips, peak


def load_frames(image_dir):
    if not os.path.isdir(image_dir):
        return []
    frames = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".npy") and fname.startswith("step"):
            step = int(fname.replace("step", "").replace(".npy", ""))
            frames.append((step, os.path.join(image_dir, fname)))
    return sorted(frames, key=lambda x: x[0])


def main():
    if not os.path.exists(FORCE_CSV):
        print(f"Force CSV not found: {FORCE_CSV}")
        return
    if not os.path.isdir(IMAGE_DIR):
        print(f"Image dir not found: {IMAGE_DIR}")
        return

    force_steps, phases, grips_raw = load_force_data(FORCE_CSV)
    grips, peak_ref = transform_lift_phases(phases, grips_raw)

    frames = load_frames(IMAGE_DIR)
    if not frames:
        print("No image frames found.")
        return
    n_frames = len(frames)

    frame_steps = np.array([s for s, _ in frames])
    frame_to_force_idx = np.searchsorted(force_steps, frame_steps)
    frame_to_force_idx = np.clip(frame_to_force_idx, 0, len(force_steps) - 1)

    # ---- Phase boundaries ----
    phase_boundaries = []  # (step, phase_id)
    cur_p = phases[0]
    phase_boundaries.append((force_steps[0], cur_p))
    for i in range(1, len(phases)):
        if phases[i] != cur_p:
            cur_p = phases[i]
            phase_boundaries.append((force_steps[i], cur_p))

    # ---- Build UI ----
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 2],
                           hspace=0.40, left=0.06, right=0.98,
                           top=0.94, bottom=0.14)

    ax_img = fig.add_subplot(gs[0])
    ax_force = fig.add_subplot(gs[1])

    ax_img.axis("off")

    # ---- Slider ----
    ax_slider = plt.axes([0.12, 0.05, 0.76, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1,
                    valinit=0, valstep=1)

    grip_max = np.max(grips) if len(grips) > 0 else 1
    grip_min = np.min(grips) if len(grips) > 0 else 0

    def update(idx):
        idx = int(idx)
        step, path = frames[idx]
        force_idx = frame_to_force_idx[idx]

        # --- Image ---
        ax_img.clear()
        ax_img.axis("off")
        img = np.load(path)
        ax_img.imshow(img)
        p = phases[force_idx] if force_idx < len(phases) else 0
        ph_label = PHASE_NAMES[p] if p < len(PHASE_NAMES) else f"Phase {p}"
        ax_img.set_title(f"Step {step}  |  {ph_label}  |  Frame {idx+1}/{n_frames}",
                         fontsize=11)

        # --- Force curve ---
        ax_force.clear()

        # Phase background shading
        for i in range(len(phase_boundaries)):
            p_start, p_id = phase_boundaries[i]
            p_end = phase_boundaries[i + 1][0] if i + 1 < len(phase_boundaries) else force_steps[-1]
            ax_force.axvspan(p_start, p_end, alpha=0.25,
                            color=PHASE_COLORS.get(p_id, "#e8e8e8"))

        # ---- Raw force curve (faded behind) ----
        ax_force.plot(force_steps, grips_raw, color="lightgray", linewidth=0.8,
                      alpha=0.5, zorder=1, label="raw")

        # ---- Transformed force curve ----
        ax_force.plot(force_steps, grips, color="steelblue", linewidth=2.0,
                      zorder=3, label="adjusted")

        # ---- Peak reference line (Phase 3/4 horizontal anchor) ----
        if peak_ref is not None:
            p34_mask = (phases == 3) | (phases == 4)
            if np.any(p34_mask):
                p34_start = force_steps[np.where(p34_mask)[0][0]]
                p34_end = force_steps[np.where(p34_mask)[0][-1]]
                ax_force.hlines(peak_ref, p34_start, p34_end,
                               colors="green", linestyles="--", linewidth=1.8,
                               zorder=5, label=f"peak ref = {peak_ref:.3f} N")

        # Phase boundary vertical lines + labels
        label_y = grip_max * 0.92
        for i, (bs, bp) in enumerate(phase_boundaries):
            if i == 0:
                continue
            ax_force.axvline(bs, color="black", linestyle="--", linewidth=0.8,
                            alpha=0.6, zorder=2)
            name = PHASE_NAMES[bp] if bp < len(PHASE_NAMES) else f"P{bp}"
            ax_force.annotate(
                name, (bs, label_y),
                fontsize=7, color="black", rotation=90,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7),
            )

        # Current frame marker
        cur_step = force_steps[force_idx]
        cur_grip = grips[force_idx]
        ax_force.plot(cur_step, cur_grip, "ro", markersize=10, zorder=10,
                      markeredgecolor="darkred", markeredgewidth=1.5)
        ax_force.axvline(cur_step, color="red", linestyle=":", alpha=0.5,
                         linewidth=1, zorder=4)

        ax_force.set_xlabel("Step")
        ax_force.set_ylabel("Grip Force (N)")
        ax_force.set_title("Gripper Force  ── 蓝: 调整后  |  灰: 原始  |  绿虚线: 峰值参考")
        ax_force.legend(loc="upper right", fontsize=8)
        ax_force.grid(True, alpha=0.3)
        ax_force.set_xlim(force_steps[0], force_steps[-1])

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()


if __name__ == "__main__":
    main()

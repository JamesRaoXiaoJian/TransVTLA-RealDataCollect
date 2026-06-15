from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "runs" / "collected_data"
MASK_OVERLAY_COLOR = np.array([1.0, 0.5, 0.0], dtype=np.float32)
MASK_OVERLAY_ALPHA = 0.55

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka camera/mask/depth data viewer.")
    parser.add_argument("--dir", type=Path, default=None, help="Collection directory. Defaults to latest Franka dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory used for latest-dataset search.")
    parser.add_argument("--ep", type=int, default=1, help="Episode id, e.g. 1 for episode_001.npz.")
    parser.add_argument("--env", type=int, default=0, help="Environment id.")
    parser.add_argument("--save-snapshot", type=Path, default=None, help="Save one viewer snapshot PNG instead of opening UI.")
    parser.add_argument("--sample", type=int, default=None, help="Sample index. Defaults to first sample with image/depth files.")
    parser.add_argument("--inspect", action="store_true", help="Print all npz fields and sidecar image/depth channels, then exit.")
    return parser.parse_args()


def latest_dataset(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    candidates = [
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "world_camera").is_dir() and next(p.glob("episode_*.npz"), None) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No collection directories found under: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def available_episodes(data_dir: Path) -> list[int]:
    episodes = []
    for path in data_dir.glob("episode_*.npz"):
        try:
            episodes.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(episodes)


def overlay_mask(rgb: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    if rgb is None:
        return None
    if mask is None:
        return rgb
    out = rgb.astype(np.float32) / 255.0
    selected = mask > 0
    for channel in range(3):
        out[:, :, channel][selected] = (
            out[:, :, channel][selected] * (1.0 - MASK_OVERLAY_ALPHA)
            + MASK_OVERLAY_COLOR[channel] * MASK_OVERLAY_ALPHA
        )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def load_npy(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def mask_pixel_count(path: Path) -> int:
    mask = load_npy(path)
    return int(np.sum(mask > 0)) if mask is not None else 0


def array_stats(array: np.ndarray) -> str:
    if array.size == 0:
        return f"shape={array.shape}, dtype={array.dtype}, empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
        if finite.size == 0:
            return f"shape={array.shape}, dtype={array.dtype}, all non-finite"
        return (
            f"shape={array.shape}, dtype={array.dtype}, "
            f"min={float(np.min(finite)):.4g}, max={float(np.max(finite)):.4g}"
        )
    return f"shape={array.shape}, dtype={array.dtype}"


def value_summary(value: Any, precision: int = 3) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return f"{item:.4g}" if isinstance(item, float) else str(item)
    return np.array2string(array, precision=precision, suppress_small=True, max_line_width=120)


def depth_display(depth: np.ndarray | None) -> np.ndarray | None:
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0)


class FrankaMaskData:
    def __init__(self, data_dir: Path, episode: int, env_id: int):
        self.data_dir = data_dir
        self.episode = episode
        self.env_id = env_id
        self.data = np.load(data_dir / f"episode_{episode:03d}.npz")
        self.env_ids = self.data["env_ids"].astype(int)
        self.env_count = int(len(np.unique(self.env_ids)))
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No samples for episode={episode}, env={env_id}")

    def _build_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for row, env_id in enumerate(self.env_ids):
            if int(env_id) != self.env_id:
                continue
            step_id = int(self.data["step_ids"][row])
            stem = f"ep{self.episode:03d}_env{self.env_id}_step{step_id:04d}.npy"
            world_mask_path = self.data_dir / "world_camera_mask" / stem
            wrist_mask_path = self.data_dir / "wrist_camera_mask" / stem
            wrist_depth_path = self.data_dir / "wrist_camera_depth" / stem
            fields = {
                name: self.data[name][row]
                for name in self.data.files
                if self.data[name].shape and self.data[name].shape[0] == len(self.env_ids)
            }
            samples.append(
                {
                    "row": row,
                    "step_id": step_id,
                    "episode_id": int(self.data["episode_ids"][row]) if "episode_ids" in self.data.files else self.episode,
                    "world_img_path": self.data_dir / "world_camera" / stem,
                    "wrist_img_path": self.data_dir / "wrist_camera" / stem,
                    "world_mask_path": world_mask_path,
                    "wrist_mask_path": wrist_mask_path,
                    "wrist_depth_path": wrist_depth_path,
                    "world_mask_px": mask_pixel_count(world_mask_path),
                    "wrist_mask_px": mask_pixel_count(wrist_mask_path),
                    "joint_pos": self.data["joint_positions"][row],
                    "joint_vel": self.data["joint_velocities"][row],
                    "ee_pos": self.data["ee_positions"][row],
                    "ee_ori": self.data["ee_orientations"][row],
                    "target_pos": self.data["target_positions"][row],
                    "cube_pos": self.data["cube_positions"][row],
                    "success": bool(self.data["success"][row]) if "success" in self.data.files else False,
                    "distance": float(self.data["distances"][row]) if "distances" in self.data.files else 0.0,
                    "pressure": int(self.data["pressure"][row]) if "pressure" in self.data.files else 0,
                    "fields": fields,
                    "world_img": None,
                    "wrist_img": None,
                    "world_mask": None,
                    "wrist_mask": None,
                    "wrist_depth": None,
                }
            )
        return samples

    def load_images(self, sample: dict[str, Any]) -> None:
        if sample["world_img"] is None:
            sample["world_img"] = load_npy(sample["world_img_path"])
        if sample["wrist_img"] is None:
            sample["wrist_img"] = load_npy(sample["wrist_img_path"])
        if sample["world_mask"] is None:
            sample["world_mask"] = load_npy(sample["world_mask_path"])
        if sample["wrist_mask"] is None:
            sample["wrist_mask"] = load_npy(sample["wrist_mask_path"])
        if sample["wrist_depth"] is None:
            sample["wrist_depth"] = load_npy(sample["wrist_depth_path"])

    def reload(self, episode: int, env_id: int) -> None:
        self.__init__(self.data_dir, episode, env_id)


class FrankaMaskViewer:
    def __init__(self, dataset: FrankaMaskData, episodes: list[int], initial_idx: int = 0):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.widgets import Button, Slider, TextBox

        self.plt = plt
        self.gridspec = gridspec
        self.Button = Button
        self.Slider = Slider
        self.TextBox = TextBox
        self.dataset = dataset
        self.episodes = episodes
        self.current_idx = min(max(initial_idx, 0), len(dataset.samples) - 1)
        self._build_ui()

    def _build_ui(self) -> None:
        self.fig = self.plt.figure(figsize=(24, 16))
        gs = self.gridspec.GridSpec(4, 5, figure=self.fig, hspace=0.52, wspace=0.40, left=0.05, right=0.98, top=0.92, bottom=0.13)
        self.ax_world_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_world_overlay = self.fig.add_subplot(gs[0, 1])
        self.ax_wrist_rgb = self.fig.add_subplot(gs[0, 2])
        self.ax_wrist_overlay = self.fig.add_subplot(gs[0, 3])
        self.ax_wrist_depth = self.fig.add_subplot(gs[0, 4])
        self.ax_joints = self.fig.add_subplot(gs[1, 0:2])
        self.ax_velocities = self.fig.add_subplot(gs[1, 2:4])
        self.ax_pressure = self.fig.add_subplot(gs[1, 4])
        self.ax_distance = self.fig.add_subplot(gs[2, 0:3])
        self.ax_mask_axis = self.ax_distance.twinx()
        self.ax_trajectory = self.fig.add_subplot(gs[2, 3:5])
        self.ax_stats = self.fig.add_subplot(gs[3, 0:5])

        ax_env = self.plt.axes([0.05, 0.065, 0.07, 0.025])
        self.env_text = self.TextBox(ax_env, "Env: ", initial=str(self.dataset.env_id))
        self.env_text.on_submit(self._on_env)
        ax_ep = self.plt.axes([0.14, 0.065, 0.08, 0.025])
        self.ep_text = self.TextBox(ax_ep, "Episode: ", initial=str(self.dataset.episode))
        self.ep_text.on_submit(self._on_episode)
        ax_slider = self.plt.axes([0.30, 0.065, 0.52, 0.025])
        self.slider = self.Slider(ax_slider, "Sample", 0, max(len(self.dataset.samples) - 1, 1), valinit=self.current_idx, valstep=1)
        self.slider.on_changed(self.update)
        ax_prev = self.plt.axes([0.24, 0.065, 0.04, 0.025])
        ax_next = self.plt.axes([0.84, 0.065, 0.04, 0.025])
        self.Button(ax_prev, "Prev").on_clicked(lambda _event: self.slider.set_val(max(0, self.current_idx - 1)))
        self.Button(ax_next, "Next").on_clicked(lambda _event: self.slider.set_val(min(len(self.dataset.samples) - 1, self.current_idx + 1)))

        self.update(self.current_idx)
        self.plt.show()

    def _on_env(self, text: str) -> None:
        try:
            env_id = int(text)
            self.dataset.reload(self.dataset.episode, env_id)
            self.env_text.set_val(str(env_id))
            self.slider.valmax = max(len(self.dataset.samples) - 1, 1)
            self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
            idx = default_sample_index(self.dataset)
            self.slider.set_val(idx)
            self.update(idx)
        except Exception:
            self.env_text.set_val(str(self.dataset.env_id))

    def _on_episode(self, text: str) -> None:
        try:
            episode = int(text)
            if episode in self.episodes:
                self.dataset.reload(episode, self.dataset.env_id)
                self.ep_text.set_val(str(episode))
                self.slider.valmax = max(len(self.dataset.samples) - 1, 1)
                self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)
                idx = default_sample_index(self.dataset)
                self.slider.set_val(idx)
                self.update(idx)
        except Exception:
            self.ep_text.set_val(str(self.dataset.episode))

    def update(self, value: float) -> None:
        self.current_idx = int(value)
        draw_sample(self.fig, self.dataset, self.current_idx, axes=self._axes())
        self.fig.canvas.draw_idle()

    def _axes(self) -> dict[str, Any]:
        return {
            "world_rgb": self.ax_world_rgb,
            "world_overlay": self.ax_world_overlay,
            "wrist_rgb": self.ax_wrist_rgb,
            "wrist_overlay": self.ax_wrist_overlay,
            "wrist_depth": self.ax_wrist_depth,
            "joints": self.ax_joints,
            "velocities": self.ax_velocities,
            "distance": self.ax_distance,
            "mask_axis": self.ax_mask_axis,
            "pressure": self.ax_pressure,
            "trajectory": self.ax_trajectory,
            "stats": self.ax_stats,
        }


def draw_sample(fig: Any, dataset: FrankaMaskData, idx: int, axes: dict[str, Any]) -> None:
    sample = dataset.samples[idx]
    dataset.load_images(sample)
    for ax in axes.values():
        ax.clear()

    fig.suptitle(
        f"Franka Camera/Mask/Depth Data Viewer | {dataset.data_dir.name} | episode {dataset.episode:03d} env {dataset.env_id}",
        fontsize=14,
        fontweight="bold",
    )

    world_img = sample["world_img"]
    wrist_img = sample["wrist_img"]
    world_mask = sample["world_mask"]
    wrist_mask = sample["wrist_mask"]
    wrist_depth = sample["wrist_depth"]
    world_overlay = overlay_mask(world_img, world_mask)
    wrist_overlay = overlay_mask(wrist_img, wrist_mask)
    wrist_depth_norm = depth_display(wrist_depth)

    image_specs = [
        ("world_rgb", world_img, f"World RGB | step {sample['step_id']} | mask {sample['world_mask_px']} px"),
        ("world_overlay", world_overlay, "World RGB + mask"),
        ("wrist_rgb", wrist_img, f"Wrist RGB | dist {sample['distance']:.3f} m | mask {sample['wrist_mask_px']} px"),
        ("wrist_overlay", wrist_overlay, "Wrist RGB + mask"),
        ("wrist_depth", wrist_depth_norm, "Wrist depth"),
    ]
    for key, image, title in image_specs:
        ax = axes[key]
        if image is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        else:
            if key == "wrist_depth":
                ax.imshow(image, cmap="viridis")
                if wrist_depth is not None:
                    finite = wrist_depth[np.isfinite(wrist_depth)]
                    if finite.size:
                        title = f"{title} | {float(np.min(finite)):.3f}-{float(np.max(finite)):.3f} m"
            else:
                ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    if world_mask is not None:
        axes["world_overlay"].contour(world_mask, levels=[0.5], colors="white", linewidths=0.8)
    if wrist_mask is not None:
        axes["wrist_overlay"].contour(wrist_mask, levels=[0.5], colors="white", linewidths=0.8)

    joint_names = [f"J{i}" for i in range(len(sample["joint_pos"]))]
    axes["joints"].bar(joint_names, sample["joint_pos"], color="tab:blue")
    axes["joints"].set_title("Joint positions")
    axes["joints"].set_ylabel("rad / m")
    axes["joints"].grid(True, alpha=0.3)

    axes["velocities"].bar(joint_names, sample["joint_vel"], color="tab:purple")
    axes["velocities"].set_title("Joint velocities")
    axes["velocities"].set_ylabel("rad/s / m/s")
    axes["velocities"].grid(True, alpha=0.3)

    steps = [entry["step_id"] for entry in dataset.samples]
    distances = [entry["distance"] for entry in dataset.samples]
    mask_pixels = [entry["world_mask_px"] for entry in dataset.samples]
    ax_dist = axes["distance"]
    ax_mask = axes["mask_axis"]
    ax_dist.plot(steps, distances, color="tab:blue", label="distance")
    ax_dist.plot(sample["step_id"], sample["distance"], "ro")
    ax_dist.set_xlabel("step")
    ax_dist.set_ylabel("distance (m)", color="tab:blue")
    ax_dist.grid(True, alpha=0.3)
    ax_mask.plot(steps, mask_pixels, color="tab:orange", label="world mask px")
    ax_mask.set_ylabel("world mask pixels", color="tab:orange")
    ax_dist.set_title("Distance and mask pixels")

    pressures = [entry["pressure"] for entry in dataset.samples]
    axes["pressure"].step(steps, pressures, where="post", color="tab:red")
    axes["pressure"].plot(sample["step_id"], sample["pressure"], "ko")
    axes["pressure"].set_title("Pressure/contact")
    axes["pressure"].set_xlabel("step")
    axes["pressure"].set_ylim(-0.1, 1.1)
    axes["pressure"].set_yticks([0, 1])
    axes["pressure"].grid(True, alpha=0.3)

    ee_positions = np.asarray([entry["ee_pos"] for entry in dataset.samples])
    axes["trajectory"].plot(ee_positions[:, 0], ee_positions[:, 1], color="tab:blue")
    axes["trajectory"].plot(sample["ee_pos"][0], sample["ee_pos"][1], "ro", label="EE")
    axes["trajectory"].scatter(sample["target_pos"][0], sample["target_pos"][1], c="green", marker="*", s=180, label="target")
    axes["trajectory"].scatter(sample["cube_pos"][0], sample["cube_pos"][1], c="orange", marker="s", s=90, label="cube")
    axes["trajectory"].set_title("End effector trajectory (XY)")
    axes["trajectory"].set_xlabel("x")
    axes["trajectory"].set_ylabel("y")
    axes["trajectory"].legend(fontsize=8)
    axes["trajectory"].grid(True, alpha=0.3)
    axes["trajectory"].axis("equal")

    env_rows = dataset.env_ids == dataset.env_id
    success_rate = float(np.mean(dataset.data["success"][env_rows])) * 100.0 if "success" in dataset.data.files else 0.0
    world_coverage = sample["world_mask_px"] / world_mask.size * 100.0 if world_mask is not None else 0.0
    wrist_coverage = sample["wrist_mask_px"] / wrist_mask.size * 100.0 if wrist_mask is not None else 0.0
    field_lines = [f"{name}: {value_summary(value)}" for name, value in sample["fields"].items()]
    field_split = (len(field_lines) + 1) // 2
    sidecar_lines = [
        f"world_camera: {'yes' if world_img is not None else 'missing'}",
        f"wrist_camera: {'yes' if wrist_img is not None else 'missing'}",
        f"world_camera_mask: {'yes' if world_mask is not None else 'missing'}",
        f"wrist_camera_mask: {'yes' if wrist_mask is not None else 'missing'}",
        f"wrist_camera_depth: {'yes' if wrist_depth is not None else 'missing'}",
    ]
    left_lines = [
        f"Sample: {idx + 1}/{len(dataset.samples)}",
        f"Step: {sample['step_id']}",
        f"Env count in episode: {dataset.env_count}",
        "",
        f"EE:     {np.array2string(sample['ee_pos'], precision=3)}",
        f"Target: {np.array2string(sample['target_pos'], precision=3)}",
        f"Cube:   {np.array2string(sample['cube_pos'], precision=3)}",
        "",
        f"Distance: {sample['distance']:.4f} m",
        f"Success sample: {sample['success']}",
        f"Episode success samples: {success_rate:.1f}%",
        f"Pressure/contact: {sample['pressure']}",
        "",
        f"World mask: {sample['world_mask_px']} px ({world_coverage:.2f}%)",
        f"Wrist mask: {sample['wrist_mask_px']} px ({wrist_coverage:.2f}%)",
        "",
        "NPZ fields:",
        *field_lines[:field_split],
    ]
    right_lines = [
        "NPZ fields:",
        *field_lines[field_split:],
        "",
        "Sidecar files:",
        *sidecar_lines,
    ]
    axes["stats"].axis("off")
    axes["stats"].text(
        0.02,
        0.96,
        "\n".join(left_lines),
        transform=axes["stats"].transAxes,
        va="top",
        family="monospace",
        fontsize=7.5,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.45),
    )
    axes["stats"].text(
        0.52,
        0.96,
        "\n".join(right_lines),
        transform=axes["stats"].transAxes,
        va="top",
        family="monospace",
        fontsize=7.5,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.45),
    )


def save_snapshot(dataset: FrankaMaskData, idx: int, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.52, wspace=0.40, left=0.05, right=0.98, top=0.92, bottom=0.05)
    axes = {
        "world_rgb": fig.add_subplot(gs[0, 0]),
        "world_overlay": fig.add_subplot(gs[0, 1]),
        "wrist_rgb": fig.add_subplot(gs[0, 2]),
        "wrist_overlay": fig.add_subplot(gs[0, 3]),
        "wrist_depth": fig.add_subplot(gs[0, 4]),
        "joints": fig.add_subplot(gs[1, 0:2]),
        "velocities": fig.add_subplot(gs[1, 2:4]),
        "pressure": fig.add_subplot(gs[1, 4]),
        "distance": fig.add_subplot(gs[2, 0:3]),
        "trajectory": fig.add_subplot(gs[2, 3:5]),
        "stats": fig.add_subplot(gs[3, 0:5]),
    }
    axes["mask_axis"] = axes["distance"].twinx()
    draw_sample(fig, dataset, idx, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def default_sample_index(dataset: FrankaMaskData) -> int:
    for idx, sample in enumerate(dataset.samples):
        if (
            sample["world_img_path"].exists()
            or sample["wrist_img_path"].exists()
            or sample["world_mask_path"].exists()
            or sample["wrist_mask_path"].exists()
            or sample["wrist_depth_path"].exists()
        ):
            return idx
    return 0


def inspect_dataset(dataset: FrankaMaskData, idx: int) -> None:
    idx = min(max(idx, 0), len(dataset.samples) - 1)
    sample = dataset.samples[idx]
    dataset.load_images(sample)

    print("NPZ fields:")
    for name in dataset.data.files:
        value = dataset.data[name]
        row_value = sample["fields"].get(name)
        row_text = f", sample={value_summary(row_value)}" if row_value is not None else ""
        print(f"  {name}: {array_stats(value)}{row_text}")

    print("\nSidecar files for selected sample:")
    channels = [
        ("world_camera", sample["world_img_path"], sample["world_img"]),
        ("wrist_camera", sample["wrist_img_path"], sample["wrist_img"]),
        ("world_camera_mask", sample["world_mask_path"], sample["world_mask"]),
        ("wrist_camera_mask", sample["wrist_mask_path"], sample["wrist_mask"]),
        ("wrist_camera_depth", sample["wrist_depth_path"], sample["wrist_depth"]),
    ]
    for name, path, array in channels:
        if array is None:
            print(f"  {name}: missing ({path})")
        else:
            print(f"  {name}: {array_stats(array)} ({path})")


def main() -> int:
    args = parse_args()
    data_dir = args.dir.resolve() if args.dir else latest_dataset(args.root.resolve())
    episodes = available_episodes(data_dir)
    if not episodes:
        raise FileNotFoundError(f"No episode_###.npz files found in: {data_dir}")
    episode = args.ep if args.ep in episodes else episodes[0]

    print("=" * 72)
    print("Franka camera/mask/depth data viewer")
    print(f"Data dir: {data_dir}")
    print(f"Episodes: {episodes}")
    print(f"Selected: episode={episode}, env={args.env}")
    print("=" * 72)

    dataset = FrankaMaskData(data_dir=data_dir, episode=episode, env_id=args.env)
    sample_idx = min(max(args.sample if args.sample is not None else default_sample_index(dataset), 0), len(dataset.samples) - 1)
    if args.inspect:
        inspect_dataset(dataset, sample_idx)
        return 0
    if args.save_snapshot:
        save_snapshot(dataset, sample_idx, args.save_snapshot.resolve())
        print(f"Saved snapshot: {args.save_snapshot.resolve()}")
        return 0

    FrankaMaskViewer(dataset, episodes, initial_idx=sample_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

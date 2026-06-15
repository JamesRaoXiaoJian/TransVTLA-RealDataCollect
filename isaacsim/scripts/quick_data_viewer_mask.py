"""
掩码版数据查看器 - 交互式浏览带透明物体掩码的采集数据

功能：
  - RGB 图像 + 透明物体掩码叠加（橙色高亮）
  - 独立掩码图（二值，对比度拉伸）
  - 掩码像素数随时间变化曲线
  - 支持自动检测最新 MASK_TEST_* / MASK_* 目录
  - 与 quick_data_viewer.py 保持一致的交互界面

使用方法：
    python quick_data_viewer_mask.py                # 自动最新掩码目录
    python quick_data_viewer_mask.py --dir <path>   # 指定目录
    python quick_data_viewer_mask.py --ep 1 --env 0 # 指定 episode / env
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTED_DATA_ROOT = os.path.join(SCRIPT_DIR, "collected_data")

MASK_OVERLAY_COLOR = np.array([1.0, 0.5, 0.0])   # 橙色 overlay
MASK_OVERLAY_ALPHA = 0.55                          # 透明度


# ─────────────────────────────────────────────────────────────────────────────
def get_latest_mask_data_dir():
    """
    优先返回最新的 MASK_TEST_* 或 MASK_* 目录，
    回退到最新任意时间戳目录（兼容 quick_data_viewer）。
    """
    if not os.path.exists(COLLECTED_DATA_ROOT):
        raise FileNotFoundError(f"数据根目录不存在: {COLLECTED_DATA_ROOT}")
    all_dirs = [
        d for d in os.listdir(COLLECTED_DATA_ROOT)
        if os.path.isdir(os.path.join(COLLECTED_DATA_ROOT, d))
    ]
    if not all_dirs:
        raise FileNotFoundError(f"未找到数据目录: {COLLECTED_DATA_ROOT}")
    # 优先掩码目录
    mask_dirs = [d for d in all_dirs if d.startswith(("MASK_TEST_", "MASK_"))]
    candidates = sorted(mask_dirs if mask_dirs else all_dirs)
    return os.path.join(COLLECTED_DATA_ROOT, candidates[-1])


def _overlay_mask_on_image(rgb: np.ndarray, mask: np.ndarray,
                            color=MASK_OVERLAY_COLOR, alpha=MASK_OVERLAY_ALPHA) -> np.ndarray:
    """
    将二值掩码 (H,W) 叠加到 RGB 图像 (H,W,3) 上，返回新 uint8 图像。
    mask=1 的像素混合为橙色。
    """
    out = rgb.astype(np.float32) / 255.0
    m = (mask > 0)
    for c in range(3):
        out[:, :, c][m] = out[:, :, c][m] * (1 - alpha) + color[c] * alpha
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
class MaskDataViewer:
    """交互式掩码数据查看器"""

    def __init__(self, episode_num=1, env_id=0, data_dir=None):
        if data_dir is None:
            self.data_dir = get_latest_mask_data_dir()
            print(f"📁 Using: {os.path.basename(self.data_dir)}")
        else:
            self.data_dir = data_dir

        # 检查是否有掩码子目录
        self.has_world_mask = os.path.isdir(os.path.join(self.data_dir, "world_camera_mask"))
        self.has_wrist_mask = os.path.isdir(os.path.join(self.data_dir, "wrist_camera_mask"))
        if not (self.has_world_mask or self.has_wrist_mask):
            print("⚠️  未找到 *_mask 子目录，将只显示 RGB 图像（无掩码叠加）")

        self.episode_num = episode_num
        self.selected_env_id = env_id
        self.available_episodes = self._get_available_episodes()

        print(f"⏳ Loading episode {episode_num}, env {env_id}...")
        self.load_episode(episode_num, env_id)
        print(f"✅ Loaded {len(self.samples)} samples")

        self._build_ui()

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def _get_available_episodes(self):
        eps = []
        for f in os.listdir(self.data_dir):
            if f.startswith("episode_") and f.endswith(".npz"):
                eps.append(int(f.split("_")[1].split(".")[0]))
        return sorted(eps)

    def load_episode(self, episode_num, env_id):
        data_file = os.path.join(self.data_dir, f"episode_{episode_num:03d}.npz")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        self.data = np.load(data_file)
        self.num_envs = len(np.unique(self.data['env_ids']))

        env_mask = self.data['env_ids'] == env_id
        self.success_rate = float(np.mean(self.data['success'][env_mask]) * 100) \
            if 'success' in self.data.files else 0.0
        self.total_success = int(np.sum(self.data['success'][env_mask])) \
            if 'success' in self.data.files else 0
        self.total_samples = int(np.sum(env_mask))

        self.samples = []
        for i in range(len(self.data['env_ids'])):
            if self.data['env_ids'][i] != env_id:
                continue
            step_id = int(self.data['step_ids'][i])
            stem = f"ep{episode_num:03d}_env{env_id}_step{step_id:04d}.npy"
            sample = {
                'idx': i,
                'env_id': int(self.data['env_ids'][i]),
                'step_id': step_id,
                'joint_pos': self.data['joint_positions'][i],
                'joint_vel': self.data['joint_velocities'][i],
                'ee_pos': self.data['ee_positions'][i],
                'ee_ori': self.data['ee_orientations'][i],
                'target_pos': self.data['target_positions'][i],
                'cube_pos': self.data['cube_positions'][i],
                'success': bool(self.data['success'][i]) if 'success' in self.data.files else False,
                'distance': float(self.data['distances'][i]) if 'distances' in self.data.files
                    else float(np.linalg.norm(self.data['ee_positions'][i] -
                                              self.data['target_positions'][i])),
                # 延迟加载路径
                'world_img_path':  os.path.join(self.data_dir, "world_camera",  stem),
                'wrist_img_path':  os.path.join(self.data_dir, "wrist_camera",  stem),
                'world_mask_path': os.path.join(self.data_dir, "world_camera_mask", stem),
                'wrist_mask_path': os.path.join(self.data_dir, "wrist_camera_mask", stem),
                # 缓存槽
                'world_img': None, 'wrist_img': None,
                'world_mask': None, 'wrist_mask': None,
            }
            # 预计算掩码像素数（用于曲线，避免在 update 循环中每次重新加载）
            wmask_path = sample['world_mask_path']
            sample['world_mask_px'] = int(np.sum(np.load(wmask_path) > 0)) \
                if os.path.exists(wmask_path) else 0
            self.samples.append(sample)

        print(f"✅ {len(self.samples)} samples | "
              f"success {self.success_rate:.1f}% ({self.total_success}/{self.total_samples})")

    def _load(self, path):
        return np.load(path) if os.path.exists(path) else None

    def _get_sample_data(self, sample):
        """延迟加载图像和掩码"""
        if sample['world_img'] is None:
            sample['world_img'] = self._load(sample['world_img_path'])
        if sample['wrist_img'] is None:
            sample['wrist_img'] = self._load(sample['wrist_img_path'])
        if sample['world_mask'] is None:
            sample['world_mask'] = self._load(sample['world_mask_path'])
        if sample['wrist_mask'] is None:
            sample['wrist_mask'] = self._load(sample['wrist_mask_path'])

    # ── UI 构建 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(22, 13))
        self._update_title()

        # 布局：3行4列
        gs = gridspec.GridSpec(
            3, 4, figure=self.fig,
            hspace=0.45, wspace=0.40,
            left=0.05, right=0.98, top=0.93, bottom=0.15
        )

        # 行0：4个图像面板
        self.ax_world_rgb     = self.fig.add_subplot(gs[0, 0])   # World RGB
        self.ax_world_overlay = self.fig.add_subplot(gs[0, 1])   # World mask overlay
        self.ax_wrist_rgb     = self.fig.add_subplot(gs[0, 2])   # Wrist RGB
        self.ax_wrist_overlay = self.fig.add_subplot(gs[0, 3])   # Wrist mask overlay

        # 行1：关节 | 距离+掩码
        self.ax_joints   = self.fig.add_subplot(gs[1, 0:2])
        self.ax_distance = self.fig.add_subplot(gs[1, 2:4])

        # 行2：轨迹 | 统计
        self.ax_trajectory = self.fig.add_subplot(gs[2, 0:2])
        self.ax_stats      = self.fig.add_subplot(gs[2, 2:4])

        # ── 控件 ──────────────────────────────────────────────────────────────
        ax_env_box = plt.axes([0.05, 0.065, 0.07, 0.025])
        self.env_textbox = TextBox(ax_env_box, 'Env: ', initial=str(self.selected_env_id))
        self.env_textbox.on_submit(self._on_env_submit)

        ax_env_lbl = plt.axes([0.05, 0.040, 0.07, 0.020])
        ax_env_lbl.axis('off')
        self._env_hint = ax_env_lbl.text(0.5, 0.5, '', ha='center', va='center',
                                         fontsize=8, color='gray')

        ax_ep_box = plt.axes([0.14, 0.065, 0.08, 0.025])
        self.ep_textbox = TextBox(ax_ep_box, 'Episode: ', initial=str(self.episode_num))
        self.ep_textbox.on_submit(self._on_ep_submit)

        ax_ep_lbl = plt.axes([0.14, 0.040, 0.08, 0.020])
        ax_ep_lbl.axis('off')
        self._ep_hint = ax_ep_lbl.text(0.5, 0.5, '', ha='center', va='center',
                                       fontsize=8, color='gray')

        ax_slider = plt.axes([0.30, 0.065, 0.52, 0.025])
        self.slider = Slider(ax_slider, 'Sample', 0, max(len(self.samples)-1, 1),
                             valinit=0, valstep=1)
        self.slider.on_changed(self.update)

        ax_prev = plt.axes([0.24, 0.065, 0.04, 0.025])
        ax_next = plt.axes([0.84, 0.065, 0.04, 0.025])
        self.btn_prev = Button(ax_prev, '◀ Prev')
        self.btn_next = Button(ax_next, 'Next ▶')
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)

        self._update_hints()
        self.current_idx = 0
        self.update(0)
        plt.show()

    def _update_title(self):
        self.fig.suptitle(
            f'Episode {self.episode_num}  |  Env {self.selected_env_id}  |  '
            f'Mask Data Viewer  [{os.path.basename(self.data_dir)}]',
            fontsize=14, fontweight='bold'
        )

    def _update_hints(self):
        self._env_hint.set_text(f'(0–{self.num_envs-1})')
        ep_min, ep_max = min(self.available_episodes), max(self.available_episodes)
        self._ep_hint.set_text(f'({ep_min}–{ep_max})')

    # ── 事件处理 ──────────────────────────────────────────────────────────────

    def _on_env_submit(self, text):
        try:
            eid = int(text)
            if 0 <= eid < self.num_envs and eid != self.selected_env_id:
                self.selected_env_id = eid
                self.load_episode(self.episode_num, eid)
                self.slider.set_val(0)
                self._update_title()
        except ValueError:
            self.env_textbox.set_val(str(self.selected_env_id))

    def _on_ep_submit(self, text):
        try:
            ep = int(text)
            if ep in self.available_episodes and ep != self.episode_num:
                self.episode_num = ep
                self.load_episode(ep, self.selected_env_id)
                self.slider.set_val(0)
                self._update_hints()
                self._update_title()
        except ValueError:
            self.ep_textbox.set_val(str(self.episode_num))

    def _prev(self, _):
        if self.current_idx > 0:
            self.slider.set_val(self.current_idx - 1)

    def _next(self, _):
        if self.current_idx < len(self.samples) - 1:
            self.slider.set_val(self.current_idx + 1)

    # ── 主绘图 ────────────────────────────────────────────────────────────────

    def update(self, val):
        idx = int(self.slider.val)
        self.current_idx = idx
        sample = self.samples[idx]

        self._get_sample_data(sample)

        for ax in [self.ax_world_rgb, self.ax_world_overlay,
                   self.ax_wrist_rgb, self.ax_wrist_overlay,
                   self.ax_joints, self.ax_distance,
                   self.ax_trajectory, self.ax_stats]:
            ax.clear()

        # ── 0a. World RGB ────────────────────────────────────────────────────
        world_img = sample['world_img']
        world_mask = sample['world_mask']
        if world_img is not None:
            self.ax_world_rgb.imshow(world_img)
        else:
            self.ax_world_rgb.text(0.5, 0.5, 'No Image', ha='center', va='center',
                                   transform=self.ax_world_rgb.transAxes)
        px_w = int(np.sum(world_mask > 0)) if world_mask is not None else 0
        step = sample['step_id']
        self.ax_world_rgb.set_title(f'World RGB  step={step}  mask_px={px_w}', fontsize=9)
        self.ax_world_rgb.axis('off')

        # ── 0b. World Mask Overlay ────────────────────────────────────────────
        if world_img is not None and world_mask is not None:
            overlay = _overlay_mask_on_image(world_img, world_mask)
            self.ax_world_overlay.imshow(overlay)
            # 在图中标注 mask 轮廓（白边）
            from matplotlib.contour import QuadContourSet
            try:
                self.ax_world_overlay.contour(world_mask, levels=[0.5],
                                              colors='white', linewidths=0.8)
            except Exception:
                pass
            coverage = px_w / (world_mask.size) * 100
            self.ax_world_overlay.set_title(
                f'World RGB + Mask  coverage={coverage:.2f}%', fontsize=9)
        elif world_img is not None:
            self.ax_world_overlay.imshow(world_img)
            self.ax_world_overlay.set_title('World RGB (no mask)', fontsize=9)
        else:
            self.ax_world_overlay.text(0.5, 0.5, 'No Mask', ha='center', va='center',
                                       transform=self.ax_world_overlay.transAxes)
            self.ax_world_overlay.set_title('World Mask Overlay', fontsize=9)
        self.ax_world_overlay.axis('off')

        # ── 0c. Wrist RGB ─────────────────────────────────────────────────────
        wrist_img = sample['wrist_img']
        wrist_mask = sample['wrist_mask']
        if wrist_img is not None:
            self.ax_wrist_rgb.imshow(wrist_img)
        else:
            self.ax_wrist_rgb.text(0.5, 0.5, 'No Image', ha='center', va='center',
                                   transform=self.ax_wrist_rgb.transAxes)
        px_r = int(np.sum(wrist_mask > 0)) if wrist_mask is not None else 0
        status_icon = '✅' if sample['success'] else '❌'
        self.ax_wrist_rgb.set_title(
            f'Wrist RGB  dist={sample["distance"]:.3f}m {status_icon}  mask_px={px_r}',
            fontsize=9)
        self.ax_wrist_rgb.axis('off')

        # ── 0d. Wrist Mask Overlay ────────────────────────────────────────────
        if wrist_img is not None and wrist_mask is not None:
            wrist_overlay = _overlay_mask_on_image(wrist_img, wrist_mask)
            self.ax_wrist_overlay.imshow(wrist_overlay)
            try:
                self.ax_wrist_overlay.contour(wrist_mask, levels=[0.5],
                                              colors='white', linewidths=0.8)
            except Exception:
                pass
            coverage_r = px_r / wrist_mask.size * 100
            self.ax_wrist_overlay.set_title(
                f'Wrist RGB + Mask  coverage={coverage_r:.2f}%', fontsize=9)
        elif wrist_img is not None:
            self.ax_wrist_overlay.imshow(wrist_img)
            self.ax_wrist_overlay.set_title('Wrist RGB (no mask)', fontsize=9)
        else:
            self.ax_wrist_overlay.text(0.5, 0.5, 'No Mask', ha='center', va='center',
                                       transform=self.ax_wrist_overlay.transAxes)
            self.ax_wrist_overlay.set_title('Wrist Mask Overlay', fontsize=9)
        self.ax_wrist_overlay.axis('off')

        # ── 1. Joint Positions ────────────────────────────────────────────────
        jnames = [f'J{i}' for i in range(len(sample['joint_pos']))]
        colors = ['tab:blue'] * (len(jnames) - 1) + ['tab:gray']
        self.ax_joints.bar(jnames, sample['joint_pos'], color=colors)
        self.ax_joints.set_ylabel('Angle (rad)')
        self.ax_joints.set_title('Joint Positions')
        self.ax_joints.grid(True, alpha=0.3)
        self.ax_joints.axhline(0, color='k', linewidth=0.5)

        # ── 2. Distance + Mask Pixel Count over time ─────────────────────────
        steps    = [s['step_id'] for s in self.samples]
        dists    = [s['distance'] for s in self.samples]
        mask_pxs = [s['world_mask_px'] for s in self.samples]

        color_dist = 'tab:blue'
        color_mask = 'tab:orange'

        ax2 = self.ax_distance
        ax2_r = ax2.twinx()

        ax2.plot(steps, dists, color=color_dist, linewidth=2, label='Distance')
        ax2.plot(sample['step_id'], sample['distance'], 'o',
                 color='red', markersize=9, zorder=5)
        ax2.axhline(0.01, color='green', linestyle='--', linewidth=1, label='1cm threshold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance (m)', color=color_dist)
        ax2.tick_params(axis='y', labelcolor=color_dist)
        ax2.set_title(f'Distance & Mask Pixels  (Env {self.selected_env_id})')
        ax2.grid(True, alpha=0.3)

        ax2_r.plot(steps, mask_pxs, color=color_mask, linewidth=1.5,
                   linestyle='-.', alpha=0.8, label='World mask px')
        ax2_r.plot(sample['step_id'], sample['world_mask_px'], 's',
                   color=color_mask, markersize=8, zorder=5)
        ax2_r.set_ylabel('Mask Pixels (World)', color=color_mask)
        ax2_r.tick_params(axis='y', labelcolor=color_mask)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

        # ── 3. XY Trajectory ─────────────────────────────────────────────────
        ee_pos_arr = np.array([s['ee_pos'] for s in self.samples])
        target = sample['target_pos']
        cube   = sample['cube_pos']

        self.ax_trajectory.plot(ee_pos_arr[:, 0], ee_pos_arr[:, 1],
                                'b-', alpha=0.6, linewidth=2)
        self.ax_trajectory.plot(sample['ee_pos'][0], sample['ee_pos'][1],
                                'ro', markersize=10, label='Current EE', zorder=5)
        self.ax_trajectory.scatter(target[0], target[1], c='green', s=200,
                                   marker='*', label='Target', zorder=5)
        self.ax_trajectory.scatter(cube[0], cube[1], c='orange', s=100,
                                   marker='s', label='Cube', zorder=4)
        self.ax_trajectory.set_xlabel('X (m)')
        self.ax_trajectory.set_ylabel('Y (m)')
        self.ax_trajectory.set_title('End-Effector Trajectory (XY Plane)')
        self.ax_trajectory.legend(fontsize=8)
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.axis('equal')

        # ── 4. Stats Panel ────────────────────────────────────────────────────
        self.ax_stats.axis('off')

        valid_px_counts = [s['world_mask_px'] for s in self.samples if s['world_mask_px'] > 0]
        avg_mask_px = float(np.mean(valid_px_counts)) if valid_px_counts else 0.0
        frames_with_mask = len(valid_px_counts)
        world_res = world_img.shape[:2] if world_img is not None else (480, 640)
        total_pixels = world_res[0] * world_res[1]

        stats_text = (
            f"Episode {self.episode_num}  |  Env {self.selected_env_id}\n"
            f"{'─'*38}\n\n"
            f"Sample:   {idx+1} / {len(self.samples)}\n"
            f"Step:     {sample['step_id']}\n\n"
            f"End-Effector:\n"
            f"  ({sample['ee_pos'][0]:.3f}, {sample['ee_pos'][1]:.3f}, {sample['ee_pos'][2]:.3f})\n\n"
            f"Target:\n"
            f"  ({sample['target_pos'][0]:.3f}, {sample['target_pos'][1]:.3f}, {sample['target_pos'][2]:.3f})\n\n"
            f"Distance: {sample['distance']:.4f} m\n"
            f"Status:   {'✅ Success' if sample['success'] else '❌ Not reached'}\n\n"
            f"Mask (this frame):\n"
            f"  World:  {px_w:5d} px  ({px_w/total_pixels*100:.2f}%)\n"
            f"  Wrist:  {px_r:5d} px\n\n"
            f"Mask (episode avg):\n"
            f"  Avg px/frame: {avg_mask_px:.1f}\n"
            f"  Frames w/ mask: {frames_with_mask}/{len(self.samples)}\n\n"
            f"Episode Stats:\n"
            f"  Success rate: {self.success_rate:.1f}%\n"
            f"  ({self.total_success}/{self.total_samples} samples)"
        )

        face_color = 'lightgreen' if sample['success'] else 'lightyellow'
        self.ax_stats.text(
            0.05, 0.97, stats_text,
            fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor=face_color, alpha=0.35),
            transform=self.ax_stats.transAxes
        )

        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mask Data Viewer")
    parser.add_argument("--dir",  default=None, help="数据目录路径（默认：最新 MASK_* 目录）")
    parser.add_argument("--ep",   type=int, default=1, help="Episode 编号（默认 1）")
    parser.add_argument("--env",  type=int, default=0, help="环境 ID（默认 0）")
    args = parser.parse_args()

    print("=" * 70)
    print("Mask Data Viewer  (transparent object segmentation)")
    print("=" * 70)
    print("  拖动滑块浏览样本，输入 Env/Episode 切换")
    print("  橙色高亮 = 透明物体掩码区域\n")

    if args.dir is not None:
        data_dir = args.dir
    else:
        try:
            data_dir = get_latest_mask_data_dir()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

    print(f"📁 Data: {data_dir}")
    episodes = sorted([
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(data_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    if not episodes:
        print("❌ 未找到 episode 数据")
        return
    print(f"📊 Episodes: {episodes}")

    ep = args.ep if args.ep in episodes else episodes[0]
    MaskDataViewer(episode_num=ep, env_id=args.env, data_dir=data_dir)


if __name__ == "__main__":
    main()

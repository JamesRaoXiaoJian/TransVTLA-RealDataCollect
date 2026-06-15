"""
快速数据查看器 - 交互式浏览采集的数据

使用方法:
    python quick_data_viewer.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTED_DATA_ROOT = os.path.join("H:\\", "collected_data")


def get_latest_data_dir():
    """获取最新的数据目录（按时间戳排序）"""
    if not os.path.exists(COLLECTED_DATA_ROOT):
        raise FileNotFoundError(f"数据根目录不存在: {COLLECTED_DATA_ROOT}")
    
    # 获取所有时间戳目录
    timestamp_dirs = [d for d in os.listdir(COLLECTED_DATA_ROOT) 
                     if os.path.isdir(os.path.join(COLLECTED_DATA_ROOT, d))]
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"未找到数据目录: {COLLECTED_DATA_ROOT}")
    
    # 按时间戳排序，返回最新的
    latest_dir = sorted(timestamp_dirs)[-1]
    return os.path.join(COLLECTED_DATA_ROOT, latest_dir)


class InteractiveDataViewer:
    """交互式数据查看器 - 优化支持大规模数据集"""
    
    def __init__(self, episode_num=1, env_id=0, data_dir=None, lazy_load_images=True):
        """
        Args:
            episode_num: Episode编号
            env_id: 环境ID
            data_dir: 数据目录（None则自动使用最新）
            lazy_load_images: 是否延迟加载图像（推荐True，减少内存占用）
        """
        # 如果未指定目录，使用最新的数据目录
        if data_dir is None:
            self.data_dir = get_latest_data_dir()
            print(f"📁 Using latest data directory: {os.path.basename(self.data_dir)}")
        else:
            self.data_dir = data_dir
        
        self.episode_num = episode_num
        self.selected_env_id = env_id
        self.lazy_load_images = lazy_load_images  # 延迟加载图像以节省内存
        self.available_episodes = self.get_available_episodes()
        
        # 加载数据
        print(f"⏳ Loading episode {episode_num}, environment {env_id}...")
        self.load_episode(episode_num, env_id)
        print(f"✅ Loaded {len(self.samples)} samples")
        
        # 创建图形
        self.fig = plt.figure(figsize=(20, 12))
        self.update_title()
        
        # 布局 - 优化为3行4列，增加间距避免重叠
        gs = gridspec.GridSpec(3, 4, figure=self.fig, hspace=0.45, wspace=0.4,
                              left=0.05, right=0.98, top=0.93, bottom=0.15)
        
        # 子图
        self.ax_world = self.fig.add_subplot(gs[0, 0:2])  # World camera
        self.ax_wrist = self.fig.add_subplot(gs[0, 2:4])  # Wrist camera
        self.ax_joints = self.fig.add_subplot(gs[1, 0:2])   # Joint positions
        self.ax_distance = self.fig.add_subplot(gs[1, 2:4]) # Distance curve
        self.ax_trajectory = self.fig.add_subplot(gs[2, 0:2]) # XY trajectory
        self.ax_stats = self.fig.add_subplot(gs[2, 2:4])    # Statistics
        
        # 控制面板区域 - 移到底部
        # Env输入框
        ax_env_box = plt.axes([0.05, 0.065, 0.08, 0.025])
        self.env_textbox = TextBox(ax_env_box, 'Env ID: ', initial=str(env_id))
        self.env_textbox.on_submit(self.on_env_submit)
        
        # Env范围提示标签
        ax_env_label = plt.axes([0.05, 0.04, 0.08, 0.02])
        ax_env_label.axis('off')
        self.env_range_text = ax_env_label.text(0.5, 0.5, '', ha='center', va='center', 
                                                fontsize=8, color='gray')
        
        # Episode输入框
        ax_ep_box = plt.axes([0.15, 0.065, 0.10, 0.025])
        self.ep_textbox = TextBox(ax_ep_box, 'Episode: ', initial=str(episode_num))
        self.ep_textbox.on_submit(self.on_episode_submit)
        
        # Episode范围提示标签
        ax_ep_label = plt.axes([0.15, 0.04, 0.10, 0.02])
        ax_ep_label.axis('off')
        self.ep_range_text = ax_ep_label.text(0.5, 0.5, '', ha='center', va='center',
                                              fontsize=8, color='gray')
        
        # 样本滑块
        ax_slider = plt.axes([0.34, 0.065, 0.52, 0.025])
        self.slider = Slider(
            ax_slider, 'Sample', 
            0, len(self.samples)-1, 
            valinit=0, 
            valstep=1
        )
        self.slider.on_changed(self.update)
        
        # 按钮
        ax_prev = plt.axes([0.27, 0.065, 0.04, 0.025])
        ax_next = plt.axes([0.88, 0.065, 0.04, 0.025])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        
        # 更新范围提示
        self.update_range_hints()
        
        # 初始显示
        self.current_idx = 0
        self.update(0)
        
        plt.show()
    
    def update_range_hints(self):
        """更新输入范围提示"""
        self.env_range_text.set_text(f'(0-{self.num_envs-1})')
        ep_min = min(self.available_episodes)
        ep_max = max(self.available_episodes)
        self.ep_range_text.set_text(f'({ep_min}-{ep_max})')
    
    def get_available_episodes(self):
        """获取所有可用的episode"""
        episodes = []
        for file in os.listdir(self.data_dir):
            if file.startswith("episode_") and file.endswith(".npz"):
                ep_num = int(file.split("_")[1].split(".")[0])
                episodes.append(ep_num)
        return sorted(episodes)
    
    def on_env_submit(self, text):
        """环境ID输入提交"""
        try:
            env_id = int(text)
            if 0 <= env_id < self.num_envs:
                if env_id != self.selected_env_id:
                    self.selected_env_id = env_id
                    self.load_episode(self.episode_num, env_id)
                    self.slider.valmax = len(self.samples) - 1
                    self.slider.ax.set_xlim(0, len(self.samples) - 1)
                    self.slider.set_val(0)
                    self.update_title()
            else:
                print(f"⚠️ Invalid Env ID: {env_id}. Must be 0-{self.num_envs-1}")
                self.env_textbox.set_val(str(self.selected_env_id))
        except ValueError:
            print(f"⚠️ Invalid input. Please enter a number.")
            self.env_textbox.set_val(str(self.selected_env_id))
    
    def on_episode_submit(self, text):
        """Episode输入提交"""
        try:
            ep_num = int(text)
            if ep_num in self.available_episodes:
                if ep_num != self.episode_num:
                    self.episode_num = ep_num
                    self.load_episode(ep_num, self.selected_env_id)
                    self.slider.valmax = len(self.samples) - 1
                    self.slider.ax.set_xlim(0, len(self.samples) - 1)
                    self.slider.set_val(0)
                    self.update_range_hints()
                    self.update_title()
            else:
                ep_min = min(self.available_episodes)
                ep_max = max(self.available_episodes)
                print(f"⚠️ Episode {ep_num} not found. Available: {ep_min}-{ep_max}")
                self.ep_textbox.set_val(str(self.episode_num))
        except ValueError:
            print(f"⚠️ Invalid input. Please enter a number.")
            self.ep_textbox.set_val(str(self.episode_num))
    
    def update_title(self):
        """更新标题"""
        self.fig.suptitle(
            f'Episode {self.episode_num} - Environment {self.selected_env_id} - Interactive Data Viewer',
            fontsize=16, fontweight='bold'
        )
    
    def load_episode(self, episode_num, env_id):
        """
        加载指定 episode 和环境的数据（优化内存使用）
        """
        print(f"⏳ Loading Episode {episode_num}, Environment {env_id}...")
        
        # 加载数值数据
        data_file = os.path.join(self.data_dir, f"episode_{episode_num:03d}.npz")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        self.data = np.load(data_file)
        
        # 获取该episode中的环境数量
        self.num_envs = len(np.unique(self.data['env_ids']))
        
        # 筛选指定环境的数据
        env_mask = self.data['env_ids'] == env_id
        
        # 统计成功率
        if 'success' in self.data.files:
            env_success = self.data['success'][env_mask]
            self.success_rate = np.mean(env_success) * 100
            self.total_success = np.sum(env_success)
            self.total_samples = len(env_success)
        else:
            self.success_rate = 0.0
            self.total_success = 0
            self.total_samples = 0
        
        # 获取样本信息（只包含选定环境的数据）
        self.samples = []
        
        for i in range(len(self.data['env_ids'])):
            if self.data['env_ids'][i] != env_id:
                continue
            
            sample = {
                'idx': i,
                'env_id': self.data['env_ids'][i],
                'step_id': self.data['step_ids'][i],
                'joint_pos': self.data['joint_positions'][i],
                'joint_vel': self.data['joint_velocities'][i],
                'ee_pos': self.data['ee_positions'][i],
                'ee_ori': self.data['ee_orientations'][i],
                'target_pos': self.data['target_positions'][i],
                'cube_pos': self.data['cube_positions'][i],
            }
            
            # 加载成功标记和距离（如果有）
            if 'success' in self.data.files:
                sample['success'] = self.data['success'][i]
            if 'distances' in self.data.files:
                sample['distance'] = self.data['distances'][i]
            else:
                # 计算距离
                sample['distance'] = np.linalg.norm(
                    sample['ee_pos'] - sample['target_pos']
                )
            
            # 图像路径（延迟加载）
            step_id = sample['step_id']
            sample['world_img_path'] = os.path.join(
                self.data_dir, "world_camera",
                f"ep{episode_num:03d}_env{env_id}_step{step_id:04d}.npy"
            )
            sample['wrist_img_path'] = os.path.join(
                self.data_dir, "wrist_camera",
                f"ep{episode_num:03d}_env{env_id}_step{step_id:04d}.npy"
            )
            
            # 如果不是延迟加载，立即加载图像
            if not self.lazy_load_images:
                sample['world_img'] = self._load_image(sample['world_img_path'])
                sample['wrist_img'] = self._load_image(sample['wrist_img_path'])
            else:
                sample['world_img'] = None
                sample['wrist_img'] = None
            
            self.samples.append(sample)
        
        print(f"✅ Loaded {len(self.samples)} samples for Environment {env_id}")
        print(f"   Success rate: {self.success_rate:.1f}% ({self.total_success}/{self.total_samples})")
    
    def _load_image(self, path):
        """延迟加载图像"""
        if os.path.exists(path):
            return np.load(path)
        return None
    
    def update(self, val):
        """更新显示（优化内存使用）"""
        idx = int(self.slider.val)
        self.current_idx = idx
        sample = self.samples[idx]
        
        # 延迟加载图像（如果还没有加载）
        if self.lazy_load_images:
            if sample['world_img'] is None:
                sample['world_img'] = self._load_image(sample['world_img_path'])
            if sample['wrist_img'] is None:
                sample['wrist_img'] = self._load_image(sample['wrist_img_path'])
        
        # 清空所有子图
        self.ax_world.clear()
        self.ax_wrist.clear()
        self.ax_joints.clear()
        self.ax_distance.clear()
        self.ax_trajectory.clear()
        self.ax_stats.clear()
        
        # 1. World Camera
        if sample['world_img'] is not None:
            self.ax_world.imshow(sample['world_img'])
            self.ax_world.set_title(
                f"World Camera - Step {sample['step_id']}"
            )
        else:
            self.ax_world.text(0.5, 0.5, 'No Image', ha='center', va='center')
        self.ax_world.axis('off')
        
        # 2. Wrist Camera
        if sample['wrist_img'] is not None:
            self.ax_wrist.imshow(sample['wrist_img'])
            success_marker = "✅" if sample.get('success', False) else "❌"
            self.ax_wrist.set_title(
                f"Wrist Camera - Distance: {sample['distance']:.4f}m {success_marker}"
            )
        else:
            self.ax_wrist.text(0.5, 0.5, 'No Image', ha='center', va='center')
        self.ax_wrist.axis('off')
        
        # 3. Joint Positions
        joint_names = [f'J{i}' for i in range(8)]
        colors = ['tab:blue'] * 7 + ['tab:gray']  # 最后一个是gripper
        self.ax_joints.bar(joint_names, sample['joint_pos'], color=colors)
        self.ax_joints.set_ylabel('Angle (rad)')
        self.ax_joints.set_title('Joint Positions')
        self.ax_joints.grid(True, alpha=0.3)
        self.ax_joints.axhline(0, color='k', linewidth=0.5)
        
        # 4. Distance Curve (该环境的完整轨迹)
        steps = [s['step_id'] for s in self.samples]
        distances = [s['distance'] for s in self.samples]
        
        self.ax_distance.plot(steps, distances, 'b-', linewidth=2)
        
        # 标记当前位置
        current_step = sample['step_id']
        current_dist = sample['distance']
        self.ax_distance.plot(current_step, current_dist, 'ro', markersize=10)
        
        # 标记成功的点
        if 'success' in sample:
            success_steps = [s['step_id'] for s in self.samples if s.get('success', False)]
            success_dists = [s['distance'] for s in self.samples if s.get('success', False)]
            if success_steps:
                self.ax_distance.scatter(success_steps, success_dists, c='green', s=20, alpha=0.5, label='Success')
        
        # 阈值线
        self.ax_distance.axhline(0.01, color='g', linestyle='--', label='Target (1cm)')
        
        self.ax_distance.set_xlabel('Step')
        self.ax_distance.set_ylabel('Distance (m)')
        self.ax_distance.set_title(f'Distance Trajectory - Env {self.selected_env_id}')
        self.ax_distance.legend()
        self.ax_distance.grid(True, alpha=0.3)
        
        # 5. XY轨迹
        ee_positions = np.array([s['ee_pos'] for s in self.samples])
        target_pos = sample['target_pos']
        
        self.ax_trajectory.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', alpha=0.6, linewidth=2)
        self.ax_trajectory.plot(sample['ee_pos'][0], sample['ee_pos'][1], 'ro', markersize=10, label='Current')
        self.ax_trajectory.scatter(target_pos[0], target_pos[1], c='green', s=200, marker='*', label='Target')
        
        self.ax_trajectory.set_xlabel('X (m)')
        self.ax_trajectory.set_ylabel('Y (m)')
        self.ax_trajectory.set_title('End-Effector Trajectory (XY Plane)')
        self.ax_trajectory.legend()
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.axis('equal')
        
        # 6. 统计信息
        self.ax_stats.axis('off')
        
        stats_text = (
            f"Episode {self.episode_num} - Environment {self.selected_env_id}\n"
            f"{'='*40}\n\n"
            f"Sample: {idx+1}/{len(self.samples)}\n"
            f"Step: {sample['step_id']}\n\n"
            f"End-Effector:\n"
            f"  Position: ({sample['ee_pos'][0]:.3f}, {sample['ee_pos'][1]:.3f}, {sample['ee_pos'][2]:.3f})\n\n"
            f"Target:\n"
            f"  Position: ({sample['target_pos'][0]:.3f}, {sample['target_pos'][1]:.3f}, {sample['target_pos'][2]:.3f})\n\n"
            f"Distance: {sample['distance']:.4f}m\n"
        )
        
        if 'success' in sample:
            status = "✅ Success" if sample['success'] else "❌ Failed"
            stats_text += f"Status: {status}\n\n"
        
        stats_text += (
            f"Environment Statistics:\n"
            f"  Success Rate: {self.success_rate:.1f}%\n"
            f"  Successful Samples: {self.total_success}/{self.total_samples}\n"
        )
        
        color = 'lightgreen' if sample.get('success', False) else 'lightcoral'
        self.ax_stats.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
                          family='monospace',
                          bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        self.fig.canvas.draw_idle()
    
    def prev_sample(self, event):
        """上一个样本"""
        if self.current_idx > 0:
            self.slider.set_val(self.current_idx - 1)
    
    def next_sample(self, event):
        """下一个样本"""
        if self.current_idx < len(self.samples) - 1:
            self.slider.set_val(self.current_idx + 1)


def main():
    print("=" * 70)
    print("Interactive Data Viewer")
    print("=" * 70)
    print("\n使用说明:")
    print("  - 拖动滑块浏览样本")
    print("  - 点击 Prev/Next 按钮切换")
    print("  - 关闭窗口退出\n")
    
    # 获取最新的数据目录
    try:
        data_dir = get_latest_data_dir()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"\n请先运行数据采集程序生成数据")
        return
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 查找可用的 episodes
    episodes = sorted([
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(data_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ])
    
    if not episodes:
        print("❌ 未找到任何 episode 数据")
        return
    
    print(f"📁 找到 {len(episodes)} 个 episodes: {episodes}")
    
    # 选择要查看的 episode
    episode_num = episodes[0]  # 默认第一个
    print(f"\n📊 查看 Episode {episode_num}")
    
    # 启动查看器
    viewer = InteractiveDataViewer(episode_num, data_dir=data_dir)


if __name__ == "__main__":
    main()

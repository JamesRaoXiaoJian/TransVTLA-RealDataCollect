"""
RM75 多环境相机数据分析工具

功能:
1. 加载和查看采集的数据
2. 可视化相机图像
3. 分析机器人轨迹
4. 生成统计报告
5. 导出视频
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path

# =============================================================================
# 配置
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTED_DATA_ROOT = os.path.join("H:\\", "collected_data")

def get_latest_data_dir():
    """获取最新的数据目录（按时间戳排序）"""
    if not os.path.exists(COLLECTED_DATA_ROOT):
        return None
    
    timestamp_dirs = [d for d in os.listdir(COLLECTED_DATA_ROOT) 
                     if os.path.isdir(os.path.join(COLLECTED_DATA_ROOT, d))]
    
    if not timestamp_dirs:
        return None
    
    latest_dir = sorted(timestamp_dirs)[-1]
    return os.path.join(COLLECTED_DATA_ROOT, latest_dir)

def list_available_datasets():
    """列出所有可用的数据集"""
    if not os.path.exists(COLLECTED_DATA_ROOT):
        return []
    
    return sorted([d for d in os.listdir(COLLECTED_DATA_ROOT) 
                  if os.path.isdir(os.path.join(COLLECTED_DATA_ROOT, d))])

# 如果命令行指定了时间戳，使用指定的；否则使用最新的
if len(sys.argv) > 1:
    timestamp = sys.argv[1]
    DATA_DIR = os.path.join(COLLECTED_DATA_ROOT, timestamp)
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 指定的数据目录不存在: {DATA_DIR}")
        print(f"\n可用的数据集:")
        for dataset in list_available_datasets():
            print(f"  - {dataset}")
        sys.exit(1)
    print(f"📁 使用指定的数据目录: {timestamp}")
else:
    DATA_DIR = get_latest_data_dir()
    if DATA_DIR is None:
        print(f"❌ 错误: 未找到数据目录: {COLLECTED_DATA_ROOT}")
        sys.exit(1)
    print(f"📁 使用最新的数据目录: {os.path.basename(DATA_DIR)}")

OUTPUT_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📊 数据目录: {DATA_DIR}")
print(f"📁 输出目录: {OUTPUT_DIR}")
print("=" * 70)


# =============================================================================
# 数据加载
# =============================================================================

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.world_cam_dir = os.path.join(data_dir, "world_camera")
        self.wrist_cam_dir = os.path.join(data_dir, "wrist_camera")
        
        # 扫描所有 episode 文件
        self.episode_files = sorted([
            f for f in os.listdir(data_dir) 
            if f.startswith("episode_") and f.endswith(".npz")
        ])
        
        print(f"📁 Found {len(self.episode_files)} episodes")
    
    def load_episode(self, episode_num: int) -> dict:
        """
        加载指定 episode 的数据
        
        Returns:
            {
                'joint_positions': (N, 8),      # 关节角度
                'joint_velocities': (N, 8),     # 关节速度
                'ee_positions': (N, 3),         # 末端位置
                'ee_orientations': (N, 4),      # 末端四元数
                'success': (N,),                # 成功标记
                'distances': (N,),              # 距离数据
                'target_positions': (N, 3),     # 目标位置
                'cube_positions': (N, 3),       # Cube 位置
                'env_ids': (N,),                # 环境ID
                'step_ids': (N,),               # 步骤ID
                'episode_ids': (N,),            # Episode ID
            }
        """
        filename = f"episode_{episode_num:03d}.npz"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Episode {episode_num} not found")
        
        data = np.load(filepath)
        return {key: data[key] for key in data.files}
    
    def load_image(self, camera_type: str, episode_num: int, env_id: int, step: int):
        """
        加载指定图像
        
        Args:
            camera_type: 'world' 或 'wrist'
            episode_num: Episode 编号
            env_id: 环境ID
            step: 步骤编号
        """
        cam_dir = self.world_cam_dir if camera_type == 'world' else self.wrist_cam_dir
        filename = f"ep{episode_num:03d}_env{env_id}_step{step:04d}.npy"
        filepath = os.path.join(cam_dir, filename)
        
        if os.path.exists(filepath):
            return np.load(filepath)
        return None
    
    def get_episode_images(self, camera_type: str, episode_num: int, env_id: int):
        """获取某个 episode 中某个环境的所有图像"""
        cam_dir = self.world_cam_dir if camera_type == 'world' else self.wrist_cam_dir
        pattern = f"ep{episode_num:03d}_env{env_id}_step*.npy"
        
        images = []
        step_ids = []
        
        for file in sorted(os.listdir(cam_dir)):
            if file.startswith(f"ep{episode_num:03d}_env{env_id}_"):
                step = int(file.split("_step")[1].split(".")[0])
                img = np.load(os.path.join(cam_dir, file))
                images.append(img)
                step_ids.append(step)
        
        return images, step_ids


# =============================================================================
# 数据分析
# =============================================================================

class TrajectoryAnalyzer:
    """轨迹分析器"""
    
    @staticmethod
    def compute_distances(ee_positions: np.ndarray, target_positions: np.ndarray):
        """计算末端到目标的距离"""
        return np.linalg.norm(ee_positions - target_positions, axis=1)
    
    @staticmethod
    def compute_velocities(positions: np.ndarray, dt: float = 0.016):
        """计算速度 (数值微分)"""
        velocities = np.diff(positions, axis=0) / dt
        # 添加第一个点 (假设初始速度为0)
        velocities = np.vstack([np.zeros((1, positions.shape[1])), velocities])
        return velocities
    
    @staticmethod
    def compute_accelerations(velocities: np.ndarray, dt: float = 0.016):
        """计算加速度"""
        accelerations = np.diff(velocities, axis=0) / dt
        accelerations = np.vstack([np.zeros((1, velocities.shape[1])), accelerations])
        return accelerations
    
    @staticmethod
    def analyze_episode(data: dict):
        """分析一个 episode 的统计信息"""
        # 计算距离（如果没有distances字段）
        if 'distances' in data:
            distances = data['distances']
        else:
            distances = TrajectoryAnalyzer.compute_distances(
                data['ee_positions'], 
                data['target_positions']
            )
        
        ee_velocities = TrajectoryAnalyzer.compute_velocities(data['ee_positions'])
        ee_speeds = np.linalg.norm(ee_velocities, axis=1)
        
        joint_speeds = np.linalg.norm(data['joint_velocities'], axis=1)
        
        stats = {
            'total_samples': len(distances),
            'num_envs': len(np.unique(data['env_ids'])),
            'distance_mean': np.mean(distances),
            'distance_std': np.std(distances),
            'distance_min': np.min(distances),
            'distance_max': np.max(distances),
            'final_distance_mean': np.mean(distances[-10:]),  # 最后10步
            'ee_speed_mean': np.mean(ee_speeds),
            'ee_speed_max': np.max(ee_speeds),
            'joint_speed_mean': np.mean(joint_speeds),
            'joint_speed_max': np.max(joint_speeds),
        }
        
        # 添加成功率统计
        if 'success' in data:
            success_array = data['success']
            stats['success_rate'] = np.mean(success_array) * 100
            stats['total_success'] = np.sum(success_array)
            stats['total_failure'] = len(success_array) - np.sum(success_array)
            
            # 每个环境的成功率
            env_success_rates = {}
            for env_id in np.unique(data['env_ids']):
                mask = data['env_ids'] == env_id
                env_success_rate = np.mean(success_array[mask]) * 100
                env_success_rates[env_id] = env_success_rate
            stats['env_success_rates'] = env_success_rates
        
        return stats, distances


# =============================================================================
# 可视化
# =============================================================================

class Visualizer:
    """数据可视化器"""
    
    @staticmethod
    def plot_episode_overview(data: dict, distances: np.ndarray, save_path: str = None):
        """绘制 episode 概览 - 简化版，只显示最重要的数据"""
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        num_envs = len(np.unique(data['env_ids']))
        has_success = 'success' in data
        
        # 1. 距离曲线（所有环境，带成功标记）
        ax1 = fig.add_subplot(gs[0, :])
        for env_id in range(num_envs):
            mask = data['env_ids'] == env_id
            steps = data['step_ids'][mask]
            dist = distances[mask]
            ax1.plot(steps, dist, label=f'Env {env_id}', alpha=0.6, linewidth=1.2)
            
            # 标记成功点
            if has_success:
                success_mask = mask & data['success']
                if np.any(success_mask):
                    success_steps = data['step_ids'][success_mask]
                    success_dist = distances[success_mask]
                    ax1.scatter(success_steps, success_dist, c='green', s=8, alpha=0.3)
        
        ax1.axhline(y=0.01, color='r', linestyle='--', label='Success Threshold (1cm)', alpha=0.5)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Distance to Target (m)', fontsize=12)
        ax1.set_title('End-Effector Distance to Target', fontsize=14, fontweight='bold')
        # 如果环境数太多，不显示图例
        if num_envs <= 10:
            ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # 2. 成功率统计（如果有）
        if has_success:
            ax2 = fig.add_subplot(gs[1, 0])
            env_ids = sorted(np.unique(data['env_ids']))
            success_rates = []
            for env_id in env_ids:
                mask = data['env_ids'] == env_id
                rate = np.mean(data['success'][mask]) * 100
                success_rates.append(rate)
            
            colors = ['green' if r >= 80 else 'orange' if r >= 50 else 'red' for r in success_rates]
            ax2.bar(env_ids, success_rates, color=colors, alpha=0.7)
            ax2.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
            ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Fair (50%)')
            ax2.set_xlabel('Environment ID', fontsize=11)
            ax2.set_ylabel('Success Rate (%)', fontsize=11)
            ax2.set_title('Success Rate by Environment', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 105])
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 距离分布直方图
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(x=np.mean(distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances):.4f}m', linewidth=2)
        ax3.axvline(x=0.01, color='green', linestyle='--', 
                   label='Success: 0.01m', alpha=0.7, linewidth=2)
        ax3.set_xlabel('Distance (m)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Distance Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def show_camera_samples(loader: DatasetLoader, episode_num: int, env_id: int, 
                           num_samples: int = 6, save_path: str = None):
        """显示单个环境的相机图像样本"""
        world_images, world_steps = loader.get_episode_images('world', episode_num, env_id)
        wrist_images, wrist_steps = loader.get_episode_images('wrist', episode_num, env_id)
        
        if not world_images:
            print(f"⚠️ No images found for episode {episode_num}, env {env_id}")
            return
        
        # 均匀采样
        indices = np.linspace(0, len(world_images)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 7))
        fig.suptitle(f'Episode {episode_num} - Environment {env_id}', fontsize=16)
        
        for i, idx in enumerate(indices):
            # World camera
            axes[0, i].imshow(world_images[idx])
            axes[0, i].set_title(f'Step {world_steps[idx]}')
            axes[0, i].axis('off')
            
            # Wrist camera
            if idx < len(wrist_images):
                axes[1, i].imshow(wrist_images[idx])
                axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('World Camera', fontsize=12)
        axes[1, 0].set_ylabel('Wrist Camera', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📷 Saved camera samples: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_video(images: list, output_path: str, fps: int = 30):
        """从图像列表创建视频"""
        if not images:
            print("❌ No images to create video")
            return
        
        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img in images:
            # Convert RGB to BGR for OpenCV
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(bgr_img)
        
        video.release()
        print(f"🎬 Created video: {output_path}")


# =============================================================================
# 主分析程序
# =============================================================================

def main():
    print("=" * 70)
    print("RM75 Multi-Environment Data Analysis")
    print("=" * 70)
    
    # 加载数据
    loader = DatasetLoader(DATA_DIR)
    
    if len(loader.episode_files) == 0:
        print("❌ No episode data found!")
        return
    
    # 分析所有 episodes
    print(f"\n📊 Analyzing {len(loader.episode_files)} episodes...")
    all_stats = []
    
    for i, filename in enumerate(loader.episode_files):
        episode_num = int(filename.split("_")[1].split(".")[0])
        data = loader.load_episode(episode_num)
        stats, distances = TrajectoryAnalyzer.analyze_episode(data)
        all_stats.append(stats)
        
        print(f"\n{'='*70}")
        print(f"Episode {episode_num}")
        print(f"{'='*70}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Environments: {stats['num_envs']}")
        print(f"  Distance to target:")
        print(f"    Mean: {stats['distance_mean']:.4f} m")
        print(f"    Std:  {stats['distance_std']:.4f} m")
        print(f"    Min:  {stats['distance_min']:.4f} m")
        print(f"    Max:  {stats['distance_max']:.4f} m")
        print(f"    Final (last 10 steps): {stats['final_distance_mean']:.4f} m")
        print(f"  End-effector speed:")
        print(f"    Mean: {stats['ee_speed_mean']:.4f} m/s")
        print(f"    Max:  {stats['ee_speed_max']:.4f} m/s")
        print(f"  Joint speed:")
        print(f"    Mean: {stats['joint_speed_mean']:.4f} rad/s")
        print(f"    Max:  {stats['joint_speed_max']:.4f} rad/s")
        
        # 添加成功率统计
        if 'success_rate' in stats:
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"    Successful samples: {stats['total_success']}")
            print(f"    Failed samples: {stats['total_failure']}")
            if 'env_success_rates' in stats:
                print(f"  Per-environment success rates:")
                for env_id, rate in stats['env_success_rates'].items():
                    status = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
                    print(f"    Env {env_id}: {rate:.1f}% {status}")
        
        # 生成可视化
        plot_path = os.path.join(OUTPUT_DIR, f"episode_{episode_num:03d}_overview.png")
        Visualizer.plot_episode_overview(data, distances, plot_path)
        
        # 随机选择一个环境显示相机图像样本
        random_env = np.random.randint(0, stats['num_envs'])
        cam_plot_path = os.path.join(OUTPUT_DIR, f"episode_{episode_num:03d}_env{random_env}_cameras.png")
        try:
            Visualizer.show_camera_samples(loader, episode_num, env_id=random_env, 
                                          num_samples=6, save_path=cam_plot_path)
            print(f"📷 Camera samples from randomly selected Env {random_env}")
        except Exception as e:
            print(f"⚠️ Could not create camera visualization: {e}")
    
    # 汇总统计
    print(f"\n{'='*70}")
    print("Overall Statistics")
    print(f"{'='*70}")
    print(f"  Total episodes: {len(all_stats)}")
    print(f"  Total samples: {sum([s['total_samples'] for s in all_stats])}")
    
    if all_stats and 'success_rate' in all_stats[0]:
        avg_success = np.mean([s['success_rate'] for s in all_stats])
        total_success = sum([s['total_success'] for s in all_stats])
        total_samples = sum([s['total_samples'] for s in all_stats])
        print(f"  Overall success rate: {avg_success:.1f}%")
        print(f"  Total successful samples: {total_success}/{total_samples}")
    
    print(f"  Average final distance: {np.mean([s['final_distance_mean'] for s in all_stats]):.4f} m")
    print(f"  Best final distance: {np.min([s['final_distance_mean'] for s in all_stats]):.4f} m")
    print(f"  Worst final distance: {np.max([s['final_distance_mean'] for s in all_stats]):.4f} m")
    
    # 保存汇总报告
    report_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RM75 数据集分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"数据目录: {DATA_DIR}\n")
        f.write(f"总Episodes: {len(all_stats)}\n")
        f.write(f"总样本数: {sum([s['total_samples'] for s in all_stats])}\n\n")
        
        if all_stats and 'success_rate' in all_stats[0]:
            f.write(f"总体成功率: {avg_success:.1f}%\n")
            f.write(f"成功样本数: {total_success}/{total_samples}\n\n")
        
        f.write(f"平均最终距离: {np.mean([s['final_distance_mean'] for s in all_stats]):.4f}m\n")
        f.write(f"最佳最终距离: {np.min([s['final_distance_mean'] for s in all_stats]):.4f}m\n")
        f.write(f"最差最终距离: {np.max([s['final_distance_mean'] for s in all_stats]):.4f}m\n")
    
    print(f"\n📄 分析报告已保存: {report_path}")
    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

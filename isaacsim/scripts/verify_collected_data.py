"""
验证采集的多环境相机数据
"""
import numpy as np
import os
import sys
from glob import glob

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

print("=" * 70)
print("验证多环境相机数据采集结果")
print("=" * 70)
print(f"数据路径: {DATA_DIR}")
print("=" * 70)

# 1. 检查 episode 数据文件
print("\n📁 Episode 数据文件:")
episode_files = sorted(glob(os.path.join(DATA_DIR, "episode_*.npz")))
total_samples = 0
total_success = 0

for f in episode_files:
    data = np.load(f)
    ep_name = os.path.basename(f)
    print(f"\n  {ep_name}:")
    print(f"    📊 数据维度:")
    print(f"       - joint_positions: {data['joint_positions'].shape}")
    print(f"       - joint_velocities: {data['joint_velocities'].shape}")
    print(f"       - ee_positions: {data['ee_positions'].shape}")
    print(f"       - ee_orientations: {data['ee_orientations'].shape}")
    print(f"       - env_ids: {data['env_ids'].shape}")
    
    num_samples = len(data['joint_positions'])
    num_envs = len(np.unique(data['env_ids']))
    total_samples += num_samples
    
    print(f"    📈 统计信息:")
    print(f"       - 总样本数: {num_samples}")
    print(f"       - 环境数: {num_envs}")
    print(f"       - 每环境样本数: {num_samples // num_envs if num_envs > 0 else 0}")
    
    # 检查新字段
    if 'success' in data.files and 'distances' in data.files:
        success_array = data['success']
        distances_array = data['distances']
        
        success_rate = np.mean(success_array) * 100
        total_success += np.sum(success_array)
        
        print(f"    ✅ 成功率统计:")
        print(f"       - 总体成功率: {success_rate:.1f}%")
        print(f"       - 成功样本数: {np.sum(success_array)}/{num_samples}")
        print(f"       - 平均距离: {np.mean(distances_array):.4f}m")
        print(f"       - 最小距离: {np.min(distances_array):.4f}m")
        print(f"       - 最大距离: {np.max(distances_array):.4f}m")
        
        # 每个环境的成功率
        env_ids = data['env_ids']
        print(f"    🎯 各环境成功率:")
        for env_id in sorted(np.unique(env_ids)):
            mask = env_ids == env_id
            env_success_rate = np.mean(success_array[mask]) * 100
            print(f"       - Env {env_id}: {env_success_rate:.1f}%")
    else:
        print(f"    ⚠️  缺少 success/distances 字段")

print(f"\n📊 总体统计:")
print(f"   - 总 Episodes: {len(episode_files)}")
print(f"   - 总样本数: {total_samples}")
if total_samples > 0:
    print(f"   - 总体成功率: {(total_success / total_samples * 100):.1f}%")

# 2. 检查 World Camera 图像
print("\n📷 World Camera 图像:")
world_images = glob(os.path.join(DATA_DIR, "world_camera", "*.npy"))
print(f"  - 总图像数: {len(world_images)}")
if world_images:
    sample_img = np.load(world_images[0])
    print(f"  - 图像形状: {sample_img.shape}")
    print(f"  - 图像类型: {sample_img.dtype}")
    print(f"  - 数值范围: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    
    # 统计每个环境的图像数
    env_counts = {}
    for img_file in world_images:
        filename = os.path.basename(img_file)
        # ep001_env0_step0005.npy
        env_id = filename.split('_')[1]  # env0
        env_counts[env_id] = env_counts.get(env_id, 0) + 1
    
    print(f"  - 每个环境的图像数:")
    for env_id in sorted(env_counts.keys()):
        print(f"      {env_id}: {env_counts[env_id]}")

# 3. 检查 Wrist Camera 图像
print("\n🤖 Wrist Camera 图像:")
wrist_images = glob(os.path.join(DATA_DIR, "wrist_camera", "*.npy"))
print(f"  - 总图像数: {len(wrist_images)}")
if wrist_images:
    sample_img = np.load(wrist_images[0])
    print(f"  - 图像形状: {sample_img.shape}")
    print(f"  - 图像类型: {sample_img.dtype}")
    print(f"  - 数值范围: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    
    # 统计每个环境的图像数
    env_counts = {}
    for img_file in wrist_images:
        filename = os.path.basename(img_file)
        env_id = filename.split('_')[1]
        env_counts[env_id] = env_counts.get(env_id, 0) + 1
    
    print(f"  - 每个环境的图像数:")
    for env_id in sorted(env_counts.keys()):
        print(f"      {env_id}: {env_counts[env_id]}")

print("\n" + "=" * 70)
print("✅ 数据验证完成!")
print("=" * 70)
print("\n💡 数据说明:")
print("  - episode_*.npz: 关节角度、速度、末端位姿等数值数据")
print("  - world_camera/: 第三视角 RGB 图像 (640x480)")
print("  - wrist_camera/: 手腕视角 RGB 图像 (320x240)")
print("  - 每 5 步保存一次图像")
print("  - 每个环境独立采集，有隔离墙防止相机交叉污染")

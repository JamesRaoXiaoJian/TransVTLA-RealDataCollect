# RM75 RMPFlow 配置生成器
# 使用 Lula 工具链从 URDF 自动生成 RMPFlow 所需的配置文件
#
# 运行方法:
#   cd c:\isaac-sim
#   python scripts\generate_rm75_rmpflow_config.py
#
# 生成的文件:
#   - scripts/rmpflow_rm75/robot_descriptor.yaml  (机器人描述)
#   - scripts/rmpflow_rm75/rm75_rmpflow_config.yaml (RMPFlow参数)

from isaacsim import SimulationApp

# 必须先启动 SimulationApp，Lula 模块才能导入
print("=" * 60)
print("RM75 RMPFlow 配置生成器")
print("=" * 60)
print("\n🚀 启动 Isaac Sim (headless 模式)...")

simulation_app = SimulationApp({"headless": True})

import os
import yaml

# 尝试导入 Lula 工具
print("📦 导入 Lula 模块...")

try:
    from isaacsim.robot_motion.lula import LulaInterfaceHelper
    HAS_LULA_HELPER = True
    print("✅ LulaInterfaceHelper 可用")
except ImportError:
    HAS_LULA_HELPER = False
    print("⚠️ LulaInterfaceHelper 不可用")

try:
    import isaacsim.robot_motion.motion_generation as mg
    HAS_MG = True
    print("✅ motion_generation 可用")
except ImportError:
    HAS_MG = False
    print("⚠️ motion_generation 不可用")

# =========================
# 配置路径
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "rmpflow_rm75")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用本地修改过的 URDF（已修复 camera_rojoint 的 velocity=0 问题）
URDF_PATH = os.path.join(OUTPUT_DIR, "RM75-B-V.urdf")

# 如果本地没有，使用原始路径
if not os.path.exists(URDF_PATH):
    URDF_PATH = r"C:/isaac-sim/URDFFiles/RM75-B-V/urdf/RM75-B-V.urdf"

print(f"\n📂 URDF 路径: {URDF_PATH}")
print(f"📂 输出目录: {OUTPUT_DIR}")

# =========================
# RM75 机器人参数（根据你的 URDF 结构）
# =========================
RM75_CONFIG = {
    "root_link": "base_link",
    "tip_link": "camera_link",  # 末端执行器
    "joints": [
        "joint_1",
        "joint_2", 
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ],
    "fixed_joints": {
        "camera_rojoint": 0.0,  # 固定相机关节
    },
    # 关节限制（弧度）
    "joint_limits": {
        "joint_1": {"lower": -3.106, "upper": 3.106},
        "joint_2": {"lower": -2.2689, "upper": 2.2689},
        "joint_3": {"lower": -3.106, "upper": 3.106},
        "joint_4": {"lower": -2.356, "upper": 2.356},
        "joint_5": {"lower": -3.106, "upper": 3.106},
        "joint_6": {"lower": -2.234, "upper": 2.234},
        "joint_7": {"lower": -6.28, "upper": 6.28},
    },
    # 默认关节位置
    "default_q": [0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0],
}

def generate_robot_descriptor():
    """生成 robot_descriptor.yaml"""
    print("\n📝 生成 robot_descriptor.yaml...")
    
    descriptor = {
        "api_version": 1.0,
        
        # 控制空间（可控关节）
        "cspace": RM75_CONFIG["joints"],
        
        # 根链接
        "root_link": RM75_CONFIG["root_link"],
        
        # 默认关节位置
        "default_q": RM75_CONFIG["default_q"],
        
        # 加速度限制
        "acceleration_limits": [6.0] * 7,
        
        # 加加速度限制
        "jerk_limits": [1200.0] * 7,
        
        # 固定关节映射规则
        "cspace_to_urdf_rules": [
            {"name": name, "rule": "fixed", "value": value}
            for name, value in RM75_CONFIG["fixed_joints"].items()
        ],
        
        # 碰撞球体（用于避障）
        # 注意：Lula 要求 collision_spheres 是一个序列（list），
        # 其中每个元素是一个字典，键为 frame name，值为球体列表
        "collision_spheres": [
            {"base_link": [
                {"center": [0.0, 0.0, 0.05], "radius": 0.12},
                {"center": [0.0, 0.0, 0.12], "radius": 0.10},
            ]},
            {"link_1": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.08},
                {"center": [0.0, -0.02, -0.05], "radius": 0.07},
            ]},
            {"link_2": [
                {"center": [0.0, -0.05, 0.0], "radius": 0.07},
                {"center": [0.0, -0.13, 0.0], "radius": 0.06},
                {"center": [0.0, -0.20, 0.0], "radius": 0.06},
            ]},
            {"link_3": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.07},
                {"center": [0.0, -0.02, -0.05], "radius": 0.06},
            ]},
            {"link_4": [
                {"center": [0.0, -0.05, 0.0], "radius": 0.06},
                {"center": [0.0, -0.12, 0.0], "radius": 0.05},
                {"center": [0.0, -0.18, 0.0], "radius": 0.05},
            ]},
            {"link_5": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.06},
                {"center": [0.0, -0.02, -0.04], "radius": 0.05},
            ]},
            {"link_6": [
                {"center": [0.0, -0.04, 0.0], "radius": 0.05},
                {"center": [0.0, -0.10, 0.0], "radius": 0.05},
            ]},
            {"link_7": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.05},
                {"center": [0.0, 0.0, 0.04], "radius": 0.04},
            ]},
            {"camera_link": [
                {"center": [0.0, 0.0, 0.0], "radius": 0.04},
            ]},
        ],
    }
    
    output_path = os.path.join(OUTPUT_DIR, "robot_descriptor.yaml")
    with open(output_path, "w") as f:
        yaml.dump(descriptor, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ 已保存: {output_path}")
    return output_path


def generate_rmpflow_config():
    """生成 rm75_rmpflow_config.yaml"""
    print("\n📝 生成 rm75_rmpflow_config.yaml...")
    
    num_joints = len(RM75_CONFIG["joints"])
    
    config = {
        # 关节限制缓冲（防止到达极限）
        "joint_limit_buffers": [0.02] * num_joints,
        
        # RMP 参数
        "rmp_params": {
            "cspace_target_rmp": {
                "metric_scalar": 50.0,
                "position_gain": 100.0,
                "damping_gain": 50.0,
                "robust_position_term_thresh": 0.5,
                "inertia": 1.0,
            },
            "cspace_trajectory_rmp": {
                "p_gain": 100.0,
                "d_gain": 10.0,
                "ff_gain": 0.25,
                "weight": 50.0,
            },
            "cspace_affine_rmp": {
                "final_handover_time_std_dev": 0.25,
                "weight": 2000.0,
            },
            "joint_limit_rmp": {
                "metric_scalar": 1000.0,
                "metric_length_scale": 0.01,
                "metric_exploder_eps": 1e-3,
                "metric_velocity_gate_length_scale": 0.01,
                "accel_damper_gain": 200.0,
                "accel_potential_gain": 1.0,
                "accel_potential_exploder_length_scale": 0.1,
                "accel_potential_exploder_eps": 1e-2,
            },
            "joint_velocity_cap_rmp": {
                "max_velocity": 1.5,
                "velocity_damping_region": 0.3,
                "damping_gain": 1000.0,
                "metric_weight": 100.0,
            },
            "target_rmp": {
                "accel_p_gain": 30.0,
                "accel_d_gain": 85.0,
                "accel_norm_eps": 0.075,
                "metric_alpha_length_scale": 0.05,
                "min_metric_alpha": 0.01,
                "max_metric_scalar": 10000.0,
                "min_metric_scalar": 2500.0,
                "proximity_metric_boost_scalar": 20.0,
                "proximity_metric_boost_length_scale": 0.02,
                "xi_estimator_gate_std_dev": 20000.0,
                "accept_user_weights": False,
            },
            "axis_target_rmp": {
                "accel_p_gain": 210.0,
                "accel_d_gain": 60.0,
                "metric_scalar": 10.0,
                "proximity_metric_boost_scalar": 3000.0,
                "proximity_metric_boost_length_scale": 0.08,
                "xi_estimator_gate_std_dev": 20000.0,
                "accept_user_weights": False,
            },
            "collision_rmp": {
                "damping_gain": 50.0,
                "damping_std_dev": 0.04,
                "damping_robustness_eps": 1e-2,
                "damping_velocity_gate_length_scale": 0.01,
                "repulsion_gain": 800.0,
                "repulsion_std_dev": 0.01,
                "metric_modulation_radius": 0.5,
                "metric_scalar": 10000.0,
                "metric_exploder_std_dev": 0.02,
                "metric_exploder_eps": 0.001,
            },
            "damping_rmp": {
                "accel_d_gain": 30.0,
                "metric_scalar": 50.0,
                "inertia": 100.0,
            },
        },
        
        # 求解器参数
        "canonical_resolve": {
            "max_acceleration_norm": 50.0,
            "projection_tolerance": 0.01,
            "verbose": False,
        },
        
        # 身体碰撞圆柱体（用于自碰撞检测）
        "body_cylinders": [
            {"name": "base_link", "pt1": [0.0, 0.0, 0.0], "pt2": [0.0, 0.0, 0.15], "radius": 0.12},
            {"name": "link_2", "pt1": [0.0, 0.0, 0.0], "pt2": [0.0, -0.256, 0.0], "radius": 0.07},
            {"name": "link_4", "pt1": [0.0, 0.0, 0.0], "pt2": [0.0, -0.21, 0.0], "radius": 0.06},
            {"name": "link_6", "pt1": [0.0, 0.0, 0.0], "pt2": [0.0, -0.168, 0.0], "radius": 0.05},
        ],
        
        # 末端执行器碰撞控制器
        "body_collision_controllers": [
            {"name": "camera_link", "radius": 0.05},
        ],
    }
    
    output_path = os.path.join(OUTPUT_DIR, "rm75_rmpflow_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ 已保存: {output_path}")
    return output_path


def try_lula_generation():
    """尝试使用 Lula 工具自动生成配置"""
    print("\n🔧 尝试使用 Lula 工具自动生成...")
    
    if not HAS_MG:
        print("❌ motion_generation 模块不可用，使用手动配置")
        return False
    
    try:
        # 尝试加载已支持的机器人配置作为参考
        supported = mg.interface_config_loader.get_supported_robot_list()
        print(f"📋 已支持的机器人: {supported}")
        
        # 尝试为自定义机器人生成配置
        # 注意：这需要 URDF 路径正确且格式兼容
        if os.path.exists(URDF_PATH):
            print(f"📂 找到 URDF: {URDF_PATH}")
            
            # 这里可以尝试使用 Lula 的自动生成功能
            # 但由于 RM75 不在预配置列表中，可能需要手动配置
            
        return False
        
    except Exception as e:
        print(f"⚠️ Lula 自动生成失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始生成 RM75 RMPFlow 配置文件")
    print("=" * 60)
    
    # 检查 URDF 是否存在
    if not os.path.exists(URDF_PATH):
        print(f"\n❌ 错误: URDF 文件不存在: {URDF_PATH}")
        print("请确保 URDF 文件路径正确")
        simulation_app.close()
        return
    
    # 尝试 Lula 自动生成（通常对自定义机器人不可用）
    lula_success = try_lula_generation()
    
    if not lula_success:
        print("\n📝 使用手动配置生成...")
        
    # 生成配置文件
    descriptor_path = generate_robot_descriptor()
    config_path = generate_rmpflow_config()
    
    print("\n" + "=" * 60)
    print("✅ 配置生成完成!")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  1. {descriptor_path}")
    print(f"  2. {config_path}")
    print(f"\n下一步:")
    print(f"  1. 检查生成的配置文件参数是否合理")
    print(f"  2. 运行: python scripts\\rm75_follow_target_rmpflow.py")
    print(f"\n⚠️ 注意:")
    print(f"  - 碰撞球体参数是估算值，可能需要根据实际模型调整")
    print(f"  - 如果机器人运动不稳定，尝试降低 max_velocity 和 acceleration_limits")
    
    print("\n👋 关闭 Isaac Sim...")
    simulation_app.close()


if __name__ == "__main__":
    main()

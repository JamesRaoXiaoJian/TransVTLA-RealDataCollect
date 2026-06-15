# RM75 Follow Target with RMPFlow
# 基于Franka官方示例改编，使用RMPFlow控制器跟踪目标

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.robot.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation as mg
from pxr import UsdGeom, Usd

# =========================
# 配置
# =========================
USD_PATH = r"C:/isaac-sim/USDFiles/simple_scene.usd"
ROBOT_PATH = "/World/RM75_B_V"
EE_LINK_PATH = "/World/RM75_B_V/link_7"  # 真正的机械臂末端，不是相机
CUBE_PATH = "/World/Cube"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RMPFLOW_DIR = os.path.join(SCRIPT_DIR, "rmpflow_rm75")
RM75_URDF_PATH = os.path.join(RMPFLOW_DIR, "RM75-B-V.urdf")
ROBOT_DESCRIPTOR_PATH = os.path.join(RMPFLOW_DIR, "robot_descriptor.yaml")
RMPFLOW_CONFIG_PATH = os.path.join(RMPFLOW_DIR, "rm75_rmpflow_config.yaml")  # 新文件名

print("=" * 60)
print("RM75 Follow Target with RMPFlow")
print("=" * 60)

# =========================
# 加载场景
# =========================
print("\n📂 加载USD场景...")
open_stage(USD_PATH)

# =========================
# 创建World
# =========================
print("🌍 创建World...")
my_world = World(stage_units_in_meters=1.0)

# =========================
# 添加机器人
# =========================
print("🤖 初始化RM75机械臂...")

rm75 = my_world.scene.add(
    SingleManipulator(
        prim_path=ROBOT_PATH, 
        name="rm75_robot", 
        end_effector_prim_path=EE_LINK_PATH
    )
)

# 设置初始关节位置 (7 个可控关节 + camera_rojoint 固定为 0)
initial_joints = np.array([0.0, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
rm75.set_joints_default_state(positions=initial_joints)

# =========================
# 获取Cube对象
# =========================
def get_cube_position():
    """获取Cube的世界坐标"""
    stage = my_world.stage
    prim = stage.GetPrimAtPath(CUBE_PATH)
    if not prim.IsValid():
        print("⚠️ Cube不存在，使用默认位置")
        return np.array([-0.3, -0.2, 0.9])

    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])

print("🎯 使用透明Cube作为目标...")

# =========================
# 目标追踪配置
# =========================
CUBE_SIZE = np.array([0.1, 0.1, 0.1])  # Cube尺寸 (10cm)
TARGET_HEIGHT_OFFSET = 0.15  # 目标位置在Cube上方的高度偏移
ARRIVAL_THRESHOLD = 0.01  # 到达阈值 (1cm)
PAUSE_THRESHOLD = 0.005  # 暂停阈值 (5mm) - 非常接近时暂停
RESUME_THRESHOLD = 0.05  # 恢复阈值 (5cm) - 目标移动超过此距离时恢复运动
STABLE_FRAMES = 30  # 需要稳定的帧数才认为真正到达

# 末端朝向：让末端朝下指向Cube
# 绕X轴旋转180度，使末端Z轴朝下
# 四元数 (w, x, y, z) 格式
TARGET_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0])  # 绕X轴旋转180度，末端朝下

# 获取初始Cube位置
initial_cube_pos = get_cube_position()
print(f"📍 初始Cube位置: {initial_cube_pos}")
print(f"📍 Cube尺寸: {CUBE_SIZE}")
print(f"📍 目标高度偏移: {TARGET_HEIGHT_OFFSET}m")
print(f"📍 目标朝向: {TARGET_ORIENTATION} (末端朝下)")
print(f"📍 到达阈值: {ARRIVAL_THRESHOLD}m, 暂停阈值: {PAUSE_THRESHOLD}m")


def build_rm75_motion_policy_config():
    """Assemble the config dictionary consumed by Lula RMPFlow."""
    required_paths = {
        "robot_descriptor": ROBOT_DESCRIPTOR_PATH,
        "rmpflow_config": RMPFLOW_CONFIG_PATH,
        "urdf": RM75_URDF_PATH,
    }
    missing = [label for label, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"缺少RMPFlow配置文件: {missing}.\n"
            f"请先运行: python scripts\\generate_rm75_rmpflow_config.py\n"
            f"然后再运行本脚本。"
        )

    def _normalize(path):
        return path.replace("\\", "/")

    return {
        "end_effector_frame_name": "link_7",  # 机械臂末端，不是相机
        "maximum_substep_size": 0.00334,
        "ignore_robot_state_updates": False,
        "robot_description_path": _normalize(ROBOT_DESCRIPTOR_PATH),
        "urdf_path": _normalize(RM75_URDF_PATH),
        "rmpflow_config_path": _normalize(RMPFLOW_CONFIG_PATH),
    }


class RM75RMPFlowController(mg.MotionPolicyController):
    def __init__(self, robot_articulation, physics_dt, config):
        self._config = config
        self._rmp_flow = mg.lula.motion_policies.RmpFlow(**config)
        self._articulation_motion_policy = mg.ArticulationMotionPolicy(robot_articulation, self._rmp_flow, physics_dt)
        super().__init__(name="rm75_rmpflow_controller", articulation_motion_policy=self._articulation_motion_policy)
        self._default_position, self._default_orientation = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def add_obstacle(self, obstacle):
        """添加障碍物到碰撞世界"""
        self._rmp_flow.add_obstacle(obstacle)
    
    def remove_obstacle(self, obstacle):
        """移除障碍物"""
        self._rmp_flow.remove_obstacle(obstacle)
    
    def add_ground_plane(self, height=0.0):
        """添加地面平面作为障碍物"""
        from lula import create_obstacle
        ground = create_obstacle("ground_plane", {"plane_normal": [0, 0, 1], "plane_origin": [0, 0, height]})
        self._rmp_flow.add_obstacle(ground)
        return ground

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

# =========================
# 创建RMPFlow控制器
# =========================
print("🎮 创建RMPFlow控制器...")

try:
    rm75_motion_policy_config = build_rm75_motion_policy_config()
    physics_dt = getattr(my_world, "get_physics_dt", lambda: 1.0 / 60.0)()
    rmp_controller = RM75RMPFlowController(
        robot_articulation=rm75,
        physics_dt=physics_dt,
        config=rm75_motion_policy_config,
    )
    print("✅ RMPFlow控制器已创建")
    print("   - Robot descriptor:", rm75_motion_policy_config["robot_description_path"])
    print("   - RMPFlow config:", rm75_motion_policy_config["rmpflow_config_path"])
    print("   - URDF:", rm75_motion_policy_config["urdf_path"])
    
    # 添加障碍物到碰撞世界
    print("🛡️ 添加碰撞障碍物...")
    try:
        import lula
        
        # 创建 Lula World 并添加障碍物
        collision_world = lula.create_world()
        
        # 添加地面平面 (作为大的扁平盒子)
        ground_obstacle = lula.create_obstacle(lula.Obstacle.Type.CUBE)
        ground_obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, (3.0, 3.0, 0.02))
        ground_pose = lula.Pose3(lula.Rotation3.identity(), (-0.0, 0.0, -0.01))
        collision_world.add_obstacle(ground_obstacle, ground_pose)
        print("   ✅ 已添加地面障碍物")
        
        # 添加Cube作为障碍物 (目标物体)
        cube_obstacle = lula.create_obstacle(lula.Obstacle.Type.CUBE)
        cube_obstacle.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, (0.1, 0.1, 0.1))
        cube_pose = lula.Pose3(lula.Rotation3.identity(), tuple(cube_position))
        collision_world.add_obstacle(cube_obstacle, cube_pose)
        print(f"   ✅ 已添加Cube障碍物 @ {cube_position}")
        
        # 将碰撞世界设置到 RmpFlow
        world_view = collision_world.add_world_view()
        rmp_controller._rmp_flow.set_world_view(world_view)
        print("   ✅ 已设置碰撞世界")
        
    except Exception as obs_err:
        import traceback
        print(f"   ⚠️ 添加障碍物时出错: {obs_err}")
        traceback.print_exc()
        print("   继续运行（无碰撞避免）")
        
except Exception as e:
    print(f"❌ 无法创建RMPFlow控制器: {e}")
    print("   请检查 rmpflow_rm75 目录中的配置文件以及URDF路径是否正确。")
    simulation_app.close()
    raise

# 获取关节控制器
articulation_controller = rm75.get_articulation_controller()

# =========================
# 重置World
# =========================
print("🔄 重置场景...", flush=True)
my_world.reset()

# 打印调试信息
robot_base_pos, robot_base_ori = rm75.get_world_pose()
ee_pos, ee_ori = rm75.end_effector.get_world_pose()
initial_target = initial_cube_pos + np.array([0.0, 0.0, TARGET_HEIGHT_OFFSET])
print(f"\n📐 调试信息:", flush=True)
print(f"   - 机器人底座位置: {robot_base_pos}", flush=True)
print(f"   - 底座朝向 (wxyz): {robot_base_ori}", flush=True)
print(f"   - 末端执行器位置: {ee_pos}", flush=True)
print(f"   - 初始目标位置: {initial_target}", flush=True)
print(f"   - 初始距离: {np.linalg.norm(initial_target - ee_pos):.4f}m", flush=True)
print(f"   - RMPFlow 末端帧: link_7 (机械臂真正末端)", flush=True)

# 打印关节状态
joint_positions = rm75.get_joint_positions()
print(f"   - 当前关节位置: {joint_positions}", flush=True)

# =========================
# 主循环
# =========================
print("\n" + "=" * 60)
print("🚀 开始执行 - 使用RMPFlow跟踪透明Cube")
print("=" * 60)
print("\n💡 提示:")
print("  1. 机器人会自动移动到透明Cube上方")
print("  2. 移动Cube，机器人会实时跟踪")
print("  3. 到达目标后会暂停，移动Cube后自动恢复")
print("  4. 按 Ctrl+C 或关闭窗口退出\n")

# 状态变量
reset_needed = False
step_count = 0
is_paused = False  # 是否暂停（到达目标后）
stable_count = 0  # 稳定帧计数
last_target_pos = None  # 上一个目标位置
last_cube_pos = None  # 上一个Cube位置

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
        target_reached = False
        step_count = 0

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            rmp_controller.reset()
            reset_needed = False
            target_reached = False
            step_count = 0
            print("🔄 场景已重置")

        step_count += 1

        # 实时读取Cube位置
        current_cube_pos = get_cube_position()
        
        # 计算目标位置（Cube上方）
        target_position = current_cube_pos + np.array([0.0, 0.0, TARGET_HEIGHT_OFFSET])
        target_orientation = TARGET_ORIENTATION  # 末端朝下

        # 获取当前末端位置
        current_ee_pos, _ = rm75.end_effector.get_world_pose()
        distance = np.linalg.norm(target_position - current_ee_pos)
        
        # 检测Cube是否移动
        cube_moved = False
        if last_cube_pos is not None:
            cube_movement = np.linalg.norm(current_cube_pos - last_cube_pos)
            if cube_movement > 0.01:  # Cube移动超过1cm
                cube_moved = True
        last_cube_pos = current_cube_pos.copy()

        # 状态机逻辑
        if is_paused:
            # 暂停状态：检查是否需要恢复
            if distance > RESUME_THRESHOLD or cube_moved:
                is_paused = False
                stable_count = 0
                print(f"\n🔄 目标移动，恢复追踪... Cube位置: {current_cube_pos}")
            else:
                # 保持当前位置，不发送新动作
                if step_count % 100 == 0:
                    print(f"⏸️  暂停中 | 距离: {distance:.4f}m | Cube: {current_cube_pos}")
                continue  # 跳过动作计算
        
        # 运动状态：计算并应用动作
        actions = rmp_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation
        )
        articulation_controller.apply_action(actions)

        # 每50步打印一次状态
        if step_count % 50 == 0:
            status = "追踪中" if not is_paused else "暂停中"
            print(f"📊 步数: {step_count} | 距离: {distance:.4f}m | Cube: [{current_cube_pos[0]:.3f}, {current_cube_pos[1]:.3f}, {current_cube_pos[2]:.3f}] | {status}")

        # 检测是否到达目标
        if distance < ARRIVAL_THRESHOLD:
            stable_count += 1
            if stable_count >= STABLE_FRAMES and not is_paused:
                is_paused = True
                print(f"\n✅ 到达目标并稳定! 距离: {distance:.4f}m")
                print(f"   Cube位置: {current_cube_pos}")
                print(f"   末端位置: {current_ee_pos}")
                print("⏸️  已暂停 - 移动Cube可继续追踪")
        else:
            stable_count = 0  # 距离增大，重置稳定计数
        
        # 更新上一个目标位置
        last_target_pos = target_position.copy()

# =========================
# 清理
# =========================
print("\n👋 关闭仿真...")
simulation_app.close()

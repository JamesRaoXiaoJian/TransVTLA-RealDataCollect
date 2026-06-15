# RM75 Multi-Environment Data Collection with Camera Images
# 使用隔离环境 + 双相机图像采集（World Camera + Wrist Camera）

from isaacsim import SimulationApp

# 仿真配置
CONFIG = {
    "headless": True,  # True = 无头模式
    "width": 1920,
    "height": 1080,
}

simulation_app = SimulationApp(CONFIG)

import os
import numpy as np
from typing import List, Tuple, Dict
from pxr import Usd, UsdGeom, Gf
from datetime import datetime

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.prims import XFormPrim
from isaacsim.robot.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.sensors.camera import Camera

# =============================================================================
# 配置参数
# =============================================================================

# 多环境配置
NUM_ENVS = 20  # 环境数量 - 推荐：8-16 (50个会导致GPU内存不足)
ENV_SPACING = 3.5  # 环境间距 (米) - 需要足够大以容纳隔离墙

# ⚠️ 警告：环境数量过多会导致性能问题
# - 每个环境有2个相机（World + Wrist）
# - 50个环境 = 100个相机同时渲染，会卡死
# - 建议：双RTX 4090最多16-20个环境

# 场景路径 - 使用带隔离墙的环境
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "USDFiles")
ENV_USD_PATH = os.path.join(USD_DIR, "env.usd")

# RMPFlow 配置
RMPFLOW_DIR = os.path.join(SCRIPT_DIR, "rmpflow_rm75")
RM75_URDF_PATH = os.path.join(RMPFLOW_DIR, "RM75-B-V.urdf")
ROBOT_DESCRIPTOR_PATH = os.path.join(RMPFLOW_DIR, "robot_descriptor.yaml")
RMPFLOW_CONFIG_PATH = os.path.join(RMPFLOW_DIR, "rm75_rmpflow_config.yaml")

# 控制参数
# 控制参数
TARGET_HEIGHT_OFFSET = 0.25  # 目标位置：Cube中心正上方25cm
ARRIVAL_XY_THRESHOLD = 0.02  # XY平面距离阈值 (2cm)
ARRIVAL_Z_MIN = 0.23         # 成功判定高度下限 (23cm)
ARRIVAL_Z_MAX = 0.28         # 成功判定高度上限 (28cm)
STABLE_FRAMES = 30
# 目标姿态：先绕X轴旋转180度(0100)，再绕Z轴旋转180度(0001)
# Result = q_z * q_x = [0, 0, 0, 1] * [0, 1, 0, 0] = [0, 0, 1, 0]
TARGET_ORIENTATION = np.array([0.0, 0.0, 1.0, 0.0])

# 相机配置
WORLD_CAMERA_RESOLUTION = (640, 480)  # 第三视角相机分辨率
WRIST_CAMERA_RESOLUTION = (320, 240)  # 手腕相机分辨率

# 数据采集
SAVE_DATA = True
# 使用时间戳创建数据目录
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = os.path.join(SCRIPT_DIR, "collected_data", TIMESTAMP)
STEPS_PER_EPISODE = 250
# 注意：randomize_cubes() 会把 current_episode 从 0 加到 1，
# 所以实际运行 episode 数 = MAX_EPISODES - 1
MAX_EPISODES = 501
SAVE_IMAGE_INTERVAL = 10  # 每10步保存一次图像（避免数据量过大）

print("=" * 70)
print("RM75 Multi-Environment Data Collection with Cameras")
print("=" * 70)
print(f"  Environments: {NUM_ENVS} (with isolation walls)")
print(f"  Spacing: {ENV_SPACING}m")
print(f"  World Camera: {WORLD_CAMERA_RESOLUTION}")
print(f"  Wrist Camera: {WRIST_CAMERA_RESOLUTION}")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")
print(f"  Max episodes: {MAX_EPISODES}")
print(f"  Success criteria: XY < {ARRIVAL_XY_THRESHOLD*100:.1f}cm, Z in [{ARRIVAL_Z_MIN*100:.1f}, {ARRIVAL_Z_MAX*100:.1f}]cm")
print(f"  Timestamp: {TIMESTAMP}")
print("=" * 70)


# 机器人复位姿态生成策略
def get_random_ready_pose():
    """
    生成一个随机的合理初始姿态
    基础姿态：位于上方，弯曲，腕部相机朝向工作区
    叠加随机噪声以增加多样性，但保持在合理范围内
    """
    # 基础姿态 [J1, J2, J3, J4, J5, J6, J7, Gripper]
    # J1(Base): 0 (正前方)
    # J2(Shoulder): -0.5 (后仰/抬起)
    # J3(Elbow): 0.8 (向前弯曲)
    # J5(Wrist): 0.8 (手腕向下弯)
    base_pose = np.array([0.0, 0.5, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0])
    
    # 随机范围 (弧度)
    # J1: +/- 0.4 (左右摆动，约23度) - 覆盖不同入社角度
    # J2: +/- 0.1 (大臂俯仰) - 改变高度
    # J3: +/- 0.1 (小臂俯仰) - 改变远近
    # J5: +/- 0.1 (手腕俯仰) - 改变相机视角微调
    # 其他关节: 较小噪声
    noise = np.random.uniform(
        low=[-0.4, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.0],
        high=[0.4, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.0]
    )
    
    return base_pose + noise



# 创建数据目录
if SAVE_DATA:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "world_camera"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "wrist_camera"), exist_ok=True)
    print(f"📁 Data directory: {DATA_DIR}")


# =============================================================================
# 辅助函数
# =============================================================================

def build_rmpflow_config():
    """构建 RMPFlow 配置"""
    def _normalize(path):
        return path.replace("\\", "/")
    
    return {
        "end_effector_frame_name": "link_7",
        "maximum_substep_size": 0.00334,
        "ignore_robot_state_updates": False,
        "robot_description_path": _normalize(ROBOT_DESCRIPTOR_PATH),
        "urdf_path": _normalize(RM75_URDF_PATH),
        "rmpflow_config_path": _normalize(RMPFLOW_CONFIG_PATH),
    }


class RM75RMPFlowController(mg.MotionPolicyController):
    """单个 RM75 的 RMPFlow 控制器"""
    
    def __init__(self, robot_articulation, physics_dt, config):
        self._config = config
        self._rmp_flow = mg.lula.motion_policies.RmpFlow(**config)
        self._articulation_motion_policy = mg.ArticulationMotionPolicy(
            robot_articulation, self._rmp_flow, physics_dt
        )
        super().__init__(
            name="rm75_rmpflow_controller",
            articulation_motion_policy=self._articulation_motion_policy
        )
        self._default_position, self._default_orientation = \
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation
        )
    
    def update_base_pose(self, position, orientation):
        """更新机器人基座位姿 (克隆后需要)"""
        self._default_position = position
        self._default_orientation = orientation
        self._motion_policy.set_robot_base_pose(
            robot_position=position,
            robot_orientation=orientation
        )


# =============================================================================
# 多环境管理器 (带相机)
# =============================================================================

class MultiEnvWithCameraManager:
    """多环境管理器 - 带相机图像采集"""
    
    def __init__(self, world: World, num_envs: int, env_spacing: float):
        self.world = world
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.stage = world.stage
        
        # 环境数据
        self.env_origins: np.ndarray = None
        self.robots: List[SingleManipulator] = []
        self.controllers: List[RM75RMPFlowController] = []
        self.articulation_controllers: List = []
        
        # 相机
        self.world_cameras: List[Camera] = []  # 第三视角相机
        self.wrist_cameras: List[Camera] = []  # 手腕相机
        
        # 状态跟踪
        self.is_paused = np.zeros(num_envs, dtype=bool)
        self.stable_counts = np.zeros(num_envs, dtype=int)
        self.last_cube_positions = np.zeros((num_envs, 3))
        
        # 数据缓冲
        self.data_buffer = {
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_orientations': [],
            'target_positions': [],
            'cube_positions': [],
            'world_camera_images': [],  # 第三视角图像
            'wrist_camera_images': [],  # 手腕相机图像
            'env_ids': [],
            'step_ids': [],
            'episode_ids': [],
            'success': [],  # 是否成功到达目标
            'distances': [],  # 到目标的距离
        }
        self.current_episode = 0
        self.current_step = 0
    
    def setup_environments(self):
        """设置多个隔离环境"""
        print("\n🔧 Setting up isolated environments...")
        
        # 计算网格布局
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        
        # 计算每个环境的原点位置
        self.env_origins = np.zeros((self.num_envs, 3))
        for i in range(self.num_envs):
            row = i // num_per_row
            col = i % num_per_row
            self.env_origins[i] = [
                col * self.env_spacing,
                row * self.env_spacing,
                0.0
            ]
        
        # 创建环境根节点
        define_prim("/World/envs", "Xform")
        
        # 为每个环境加载隔离的env场景
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            self._create_single_env(i, env_path)
            print(f"   ✅ Created isolated env_{i} at {self.env_origins[i]}")
        
        print(f"\n✅ All {self.num_envs} isolated environments created!")
        return self.env_origins
    
    def _create_single_env(self, env_idx: int, env_path: str):
        """创建单个隔离环境"""
        origin = self.env_origins[env_idx]
        
        # 创建环境 Xform
        env_prim = define_prim(env_path, "Xform")
        env_xform = UsdGeom.Xformable(env_prim)
        env_xform.AddTranslateOp().Set(Gf.Vec3d(origin[0], origin[1], origin[2]))
        
        # 加载完整的隔离环境 USD
        # env.usd 包含: env/table_instanceable/RM75_B_V, env/table_instanceable/Cube,
        #               env/BlackGrid (隔离墙), env/World_Camera
        add_reference_to_stage(ENV_USD_PATH, env_path)
    
    def initialize_robots(self):
        """初始化所有机器人"""
        print("\n🤖 Initializing robots...")
        
        physics_dt = self.world.get_physics_dt()
        rmpflow_config = build_rmpflow_config()
        
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            # 根据 env.usd 结构，机器人在 env/table_instanceable/RM75_B_V
            robot_path = f"{env_path}/env/table_instanceable/RM75_B_V"
            ee_path = f"{robot_path}/link_7"
            
            try:
                # 创建机器人实例
                robot = self.world.scene.add(
                    SingleManipulator(
                        prim_path=robot_path,
                        name=f"rm75_robot_{i}",
                        end_effector_prim_path=ee_path
                    )
                )
                
                # 设置初始关节位置 - 使用随机 Ready Pose
                initial_joints = get_random_ready_pose()
                robot.set_joints_default_state(positions=initial_joints)
                
                # 创建控制器
                controller = RM75RMPFlowController(
                    robot_articulation=robot,
                    physics_dt=physics_dt,
                    config=rmpflow_config
                )
                
                # 更新控制器的基座位姿
                robot_pos, robot_ori = robot.get_world_pose()
                controller.update_base_pose(robot_pos, robot_ori)
                
                # 获取 articulation controller
                art_controller = robot.get_articulation_controller()
                
                self.robots.append(robot)
                self.controllers.append(controller)
                self.articulation_controllers.append(art_controller)
                
                print(f"   ✅ Robot {i} initialized")
                
            except Exception as e:
                print(f"   ❌ Failed to initialize robot {i}: {e}")
                raise
        
        print(f"\n✅ All {len(self.robots)} robots initialized!")
    
    def initialize_cameras(self):
        """初始化所有相机"""
        print("\n📷 Initializing cameras...")
        
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            
            # World Camera (第三视角) - env.usd 中已定义
            world_cam_path = f"{env_path}/env/World_Camera"
            try:
                # 使用 add 方法将相机添加到场景
                world_cam = self.world.scene.add(
                    Camera(
                        prim_path=world_cam_path,
                        name=f"world_camera_{i}",
                        resolution=WORLD_CAMERA_RESOLUTION
                    )
                )
                self.world_cameras.append(world_cam)
                print(f"   ✅ World camera {i} initialized")
            except Exception as e:
                print(f"   ⚠️ Failed to init world camera {i}: {e}")
                self.world_cameras.append(None)
            
            # Wrist Camera (手腕相机) - env.usd 中已定义
            wrist_cam_path = f"{env_path}/env/table_instanceable/RM75_B_V/camera_link/Wrist_Camera"
            try:
                # 使用 add 方法将相机添加到场景
                wrist_cam = self.world.scene.add(
                    Camera(
                        prim_path=wrist_cam_path,
                        name=f"wrist_camera_{i}",
                        resolution=WRIST_CAMERA_RESOLUTION
                    )
                )
                self.wrist_cameras.append(wrist_cam)
                print(f"   ✅ Wrist camera {i} initialized")
            except Exception as e:
                print(f"   ⚠️ Failed to init wrist camera {i}: {e}")
                self.wrist_cameras.append(None)
        
        print(f"\n✅ All cameras initialized!")
    
    def get_cube_positions(self) -> np.ndarray:
        """获取所有环境中 Cube 的世界坐标"""
        positions = np.zeros((self.num_envs, 3))
        
        for i in range(self.num_envs):
            # 根据 env.usd，Cube 在 env/table_instanceable/Cube
            cube_path = f"/World/envs/env_{i}/env/table_instanceable/Cube"
            prim = self.stage.GetPrimAtPath(cube_path)
            
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = mat.ExtractTranslation()
                positions[i] = [t[0], t[1], t[2]]
            else:
                # 默认位置
                positions[i] = self.env_origins[i] + np.array([0.3, 0.0, 0.05])
        
        return positions
    
    def randomize_cubes(self, x_range=(0.25, 0.40), y_range=(-0.20, 0.20)):
        """
        随机化所有 Cube 的位置
        
        优化后的范围：
        - X轴 (前后): 0.25~0.40m (避开基座碰撞体半径 0.12m)
        - Y轴 (左右): -0.20~0.20m (左右各约1/3桌宽，总范围40cm)
        - 确保80%以上位置处于机械臂活动范围内
        """
        for i in range(self.num_envs):
            cube_path = f"/World/envs/env_{i}/env/table_instanceable/Cube"
            prim = self.stage.GetPrimAtPath(cube_path)
            
            if prim.IsValid():
                # 随机位置 - 优化后的范围
                # 80%概率在核心工作区，20%在外围
                if np.random.random() < 0.8:
                    # 核心工作区：X: 0.28~0.35m, Y: -0.10~0.10m
                    x = np.random.uniform(0.28, 0.35)
                    y = np.random.uniform(-0.10, 0.10)
                else:
                    # 外围区域：使用完整范围
                    x = np.random.uniform(*x_range)
                    y = np.random.uniform(*y_range)
                
                z = 0.035  # 桌面高度
                
                # 设置位置
                xform = UsdGeom.Xformable(prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(x, y, z))
                        break
        
        # 重置机器人位置
        print("   🤖 Resetting robots to random ready poses...")
        for i, robot in enumerate(self.robots):
            random_pose = get_random_ready_pose()
            robot.set_joint_positions(random_pose)
            robot.set_joint_velocities(np.zeros_like(random_pose))
            
            # 重置控制器
            # 获取当前位姿并更新控制器，防止跳变
            # robot_pos, robot_ori = robot.end_effector.get_world_pose()
            # self.controllers[i].update_base_pose(robot_pos, robot_ori) # 这可能不是必须的，取决于RmpFlow实现
            # 这里我们重置控制器内部状态
            self.controllers[i].reset()

        # 重置状态
        self.is_paused[:] = False
        self.stable_counts[:] = 0
        self.current_step = 0
        self.current_episode += 1
        
        print(f"🎲 Episode {self.current_episode}: Randomized cube positions & Reset robots")
    
    def capture_camera_images(self) -> Tuple[List, List]:
        """采集所有相机图像"""
        world_images = []
        wrist_images = []
        
        for i in range(self.num_envs):
            # World camera
            if self.world_cameras[i] is not None:
                try:
                    world_img = self.world_cameras[i].get_rgba()
                    world_images.append(world_img[:, :, :3])  # RGB only
                except:
                    world_images.append(None)
            else:
                world_images.append(None)
            
            # Wrist camera
            if self.wrist_cameras[i] is not None:
                try:
                    wrist_img = self.wrist_cameras[i].get_rgba()
                    wrist_images.append(wrist_img[:, :, :3])  # RGB only
                except:
                    wrist_images.append(None)
            else:
                wrist_images.append(None)
        
        return world_images, wrist_images
    
    def step(self) -> Dict:
        """执行一步控制并返回数据"""
        self.current_step += 1
        
        # 获取 Cube 位置
        cube_positions = self.get_cube_positions()
        target_positions = cube_positions + np.array([0.0, 0.0, TARGET_HEIGHT_OFFSET])
        
        # 采集图像（每隔一定步数）
        save_images = (self.current_step % SAVE_IMAGE_INTERVAL == 0)
        if save_images:
            world_images, wrist_images = self.capture_camera_images()
        else:
            world_images, wrist_images = None, None
        
        # 收集数据
        step_data = {
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_orientations': [],
            'distances': [],
            'world_images': world_images,
            'wrist_images': wrist_images,
        }
        
        for i, (robot, controller, art_ctrl) in enumerate(
            zip(self.robots, self.controllers, self.articulation_controllers)
        ):
            # 获取状态
            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            ee_pos, ee_ori = robot.end_effector.get_world_pose()
            
            step_data['joint_positions'].append(joint_pos)
            step_data['joint_velocities'].append(joint_vel)
            step_data['ee_positions'].append(ee_pos)
            step_data['ee_orientations'].append(ee_ori)
            
            # 计算距离
            distance = np.linalg.norm(target_positions[i] - ee_pos)
            step_data['distances'].append(distance)
            
            # 检测 Cube 移动
            cube_moved = np.linalg.norm(cube_positions[i] - self.last_cube_positions[i]) > 0.01
            self.last_cube_positions[i] = cube_positions[i].copy()
            
            # 状态机
            if self.is_paused[i]:
                if distance > 0.05 or cube_moved:
                    self.is_paused[i] = False
                    self.stable_counts[i] = 0
                else:
                    continue
            
            # 计算并应用动作
            actions = controller.forward(
                target_end_effector_position=target_positions[i],
                target_end_effector_orientation=TARGET_ORIENTATION
            )
            art_ctrl.apply_action(actions)
            
            # 检测到达 - 使用XY平面距离判断 + Z轴高度范围
            ee_pos = step_data['ee_positions'][i]
            target_pos = target_positions[i]
            cube_pos = cube_positions[i]
            
            # XY距离（相对于目标中心）
            xy_distance = np.sqrt((ee_pos[0] - target_pos[0])**2 + (ee_pos[1] - target_pos[1])**2)
            
            # Z高度（相对于Cube表面）
            # target_pos[2] 是 cube_pos[2] + TARGET_HEIGHT_OFFSET (0.25)
            # 我们需要 ee_pos[2] - cube_pos[2] 在 [ARRIVAL_Z_MIN, ARRIVAL_Z_MAX] 之间
            rel_z = ee_pos[2] - cube_pos[2]
            is_height_ok = ARRIVAL_Z_MIN <= rel_z <= ARRIVAL_Z_MAX
            
            is_above_target = xy_distance < ARRIVAL_XY_THRESHOLD and is_height_ok
            
            if is_above_target:
                self.stable_counts[i] += 1
                if self.stable_counts[i] >= STABLE_FRAMES:
                    self.is_paused[i] = True
            else:
                self.stable_counts[i] = 0
        
        # 保存数据到缓冲区
        if SAVE_DATA and save_images:
            for i in range(self.num_envs):
                distance = step_data['distances'][i]
                ee_pos = step_data['ee_positions'][i]
                target_pos = target_positions[i]
                
                # 成功判定：XY平面距离 + Z高度都在阈值内
                cube_pos = cube_positions[i]
                xy_distance = np.sqrt((ee_pos[0] - target_pos[0])**2 + (ee_pos[1] - target_pos[1])**2)
                rel_z = ee_pos[2] - cube_pos[2]
                is_success = (xy_distance < ARRIVAL_XY_THRESHOLD) and (ARRIVAL_Z_MIN <= rel_z <= ARRIVAL_Z_MAX)
                
                self.data_buffer['joint_positions'].append(step_data['joint_positions'][i])
                self.data_buffer['joint_velocities'].append(step_data['joint_velocities'][i])
                self.data_buffer['ee_positions'].append(step_data['ee_positions'][i])
                self.data_buffer['ee_orientations'].append(step_data['ee_orientations'][i])
                self.data_buffer['target_positions'].append(target_positions[i])
                self.data_buffer['cube_positions'].append(cube_positions[i])
                self.data_buffer['env_ids'].append(i)
                self.data_buffer['step_ids'].append(self.current_step)
                self.data_buffer['episode_ids'].append(self.current_episode)
                self.data_buffer['success'].append(is_success)
                self.data_buffer['distances'].append(distance)
                
                # 保存图像
                if world_images and world_images[i] is not None:
                    img_path = os.path.join(
                        DATA_DIR, "world_camera",
                        f"ep{self.current_episode:03d}_env{i}_step{self.current_step:04d}.npy"
                    )
                    np.save(img_path, world_images[i])
                
                if wrist_images and wrist_images[i] is not None:
                    img_path = os.path.join(
                        DATA_DIR, "wrist_camera",
                        f"ep{self.current_episode:03d}_env{i}_step{self.current_step:04d}.npy"
                    )
                    np.save(img_path, wrist_images[i])
        
        return step_data
    
    def save_episode_data(self):
        """保存当前 episode 的数据"""
        if not SAVE_DATA or len(self.data_buffer['joint_positions']) == 0:
            return
        
        # 计算每个环境的成功率
        success_array = np.array(self.data_buffer['success'])
        env_ids_array = np.array(self.data_buffer['env_ids'])
        distances_array = np.array(self.data_buffer['distances'])
        
        # 重新计算最终成功状态（基于XY+Z判定）
        ee_positions_array = np.array(self.data_buffer['ee_positions'])
        target_positions_array = np.array(self.data_buffer['target_positions'])
        
        success_stats = []
        for env_id in range(self.num_envs):
            mask = env_ids_array == env_id
            env_success = success_array[mask]
            env_distances = distances_array[mask]
            env_ee_positions = ee_positions_array[mask]
            env_target_positions = target_positions_array[mask]
            
            if len(env_success) > 0:
                success_rate = np.mean(env_success) * 100
                final_distance = env_distances[-1] if len(env_distances) > 0 else 999
                
                # 计算最终位置的XY和Z距离
                final_ee = env_ee_positions[-1]
                final_target = env_target_positions[-1]
                
                # 注意：这里我们重新获取这个环境对应的Cube位置可能比较麻烦，因为数据只存了每一步的
                # 但我们可以利用 final_target 计算，因为 final_target = cube + offset
                # 所以 cube = final_target - offset
                # rel_z = final_ee - cube = final_ee - (final_target - offset) = final_ee - final_target + offset
                
                final_xy_dist = np.sqrt((final_ee[0] - final_target[0])**2 + (final_ee[1] - final_target[1])**2)
                
                # 反推相对高度
                final_rel_z = final_ee[2] - (final_target[2] - TARGET_HEIGHT_OFFSET)
                
                is_final_success = (final_xy_dist < ARRIVAL_XY_THRESHOLD) and (ARRIVAL_Z_MIN <= final_rel_z <= ARRIVAL_Z_MAX)
                
                success_stats.append({
                    'env_id': env_id,
                    'success_rate': success_rate,
                    'final_distance': final_distance,
                    'final_xy_distance': final_xy_dist,
                    'final_z_distance': final_rel_z,
                    'is_final_success': is_final_success
                })
        
        # 保存关节和末端数据
        data_file = os.path.join(DATA_DIR, f"episode_{self.current_episode:03d}.npz")
        np.savez(
            data_file,
            joint_positions=np.array(self.data_buffer['joint_positions']),
            joint_velocities=np.array(self.data_buffer['joint_velocities']),
            ee_positions=np.array(self.data_buffer['ee_positions']),
            ee_orientations=np.array(self.data_buffer['ee_orientations']),
            target_positions=np.array(self.data_buffer['target_positions']),
            cube_positions=np.array(self.data_buffer['cube_positions']),
            env_ids=np.array(self.data_buffer['env_ids']),
            step_ids=np.array(self.data_buffer['step_ids']),
            episode_ids=np.array(self.data_buffer['episode_ids']),
            success=success_array,
            distances=distances_array,
        )
        
        # 统计总体成功率
        overall_success_rate = np.mean(success_array) * 100
        final_success_count = sum([s['is_final_success'] for s in success_stats])
        
        print(f"💾 Saved episode {self.current_episode} data: {len(self.data_buffer['joint_positions'])} samples")
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
        print(f"   Final success: {final_success_count}/{self.num_envs} environments")
        
        # 显示每个环境的详细信息
        for stats in success_stats:
            status = "✅" if stats['is_final_success'] else "❌"
            print(f"   Env {stats['env_id']}: {status} "
                  f"XY={stats['final_xy_distance']*100:.2f}cm, H={stats['final_z_distance']*100:.2f}cm (Target 24-28)")
        
        # 清空缓冲区
        for key in self.data_buffer:
            self.data_buffer[key] = []


# =============================================================================
# 主程序
# =============================================================================

# 创建World
print("\n🌍 Creating World...")
my_world = World(stage_units_in_meters=1.0)

# 创建多环境管理器
manager = MultiEnvWithCameraManager(my_world, NUM_ENVS, ENV_SPACING)

# 设置环境
manager.setup_environments()

# 初始化机器人 (添加到场景)
manager.initialize_robots()

# 初始化相机 (添加到场景)
manager.initialize_cameras()

# 重置World (初始化所有对象)
print("\n🔄 Resetting world...")
my_world.reset()

# 必须在 reset() 之后显式调用 play()，否则主循环中 is_playing() 将始终为 False
my_world.play()

# 随机化 Cube 位置
# 注意：此函数会将 current_episode 从 0 递增到 1
manager.randomize_cubes()

# =============================================================================
# 主循环
# =============================================================================
print("\n" + "=" * 70)
print("🚀 Starting Multi-Environment Data Collection with Cameras")
print("=" * 70)
print("\n💡 Tips:")
print("  - Each environment is isolated with walls")
print("  - Collecting: joints, EE poses, World Camera, Wrist Camera")
print("  - Press Ctrl+C to stop\n")

try:
    while simulation_app.is_running() and manager.current_episode < MAX_EPISODES:
        my_world.step(render=True)
        
        if my_world.is_playing():
            # 执行一步
            step_data = manager.step()
            
            # 打印状态
            if manager.current_step % 50 == 0:
                avg_dist = np.mean(step_data['distances'])
                reached = np.sum(manager.is_paused)
                print(f"📊 Ep {manager.current_episode} | Step {manager.current_step:4d} | "
                      f"Avg dist: {avg_dist:.4f}m | Reached: {reached}/{NUM_ENVS}")
            
            # Episode 结束
            if manager.current_step >= STEPS_PER_EPISODE:
                print(f"✅ Episode {manager.current_episode} completed")
                manager.save_episode_data()
                
                if manager.current_episode < MAX_EPISODES:
                    manager.randomize_cubes()

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")

# 保存最后的数据
manager.save_episode_data()

print("\n✅ Data collection completed!")
print(f"📁 Data saved to: {DATA_DIR}")
print(f"   - Joint/EE data: episode_*.npz")
print(f"   - World camera images: world_camera/")
print(f"   - Wrist camera images: wrist_camera/")

# 清理
simulation_app.close()

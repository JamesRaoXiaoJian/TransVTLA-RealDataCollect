# RM75 Multi-Environment Data Collection with Camera Images + Transparent Object Masks
# 功能：隔离环境 + 双相机图像采集（World Camera + Wrist Camera）+ 透明物体语义掩码
#
# 掩码说明：
#   透明物体（Cube）在渲染上视觉透明，但 Isaac Sim 的分割系统基于几何体，
#   因此掩码轮廓完全精确，不受透明度/折射影响。
#   - world_camera_mask/ : 第三视角语义掩码 (H, W) uint16
#   - wrist_camera_mask/ : 手腕相机语义掩码 (H, W) uint16
#   - 像素值 0 = 背景，非0 = 透明目标物体
#
# 测试验证（_test_mask_imports.py）:
#   TEST_11: semantic_segmentation shape=(128, 128), dtype=uint32, unique_ids=[0, 1]
#   TEST_12: instance_segmentation shape=(128, 128), dtype=uint32
#   所有掩码功能正常可用。

from isaacsim import SimulationApp

# 仿真配置
CONFIG = {
    "headless": True,  # True = 无头模式
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(CONFIG)

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from pxr import Usd, UsdGeom, Gf
from datetime import datetime

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.semantics import add_labels  # 语义标签 (掩码核心)
from isaacsim.core.prims import XFormPrim
from isaacsim.robot.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.sensors.camera import Camera

# =============================================================================
# 配置参数
# =============================================================================

# 多环境配置
NUM_ENVS = 20  # 环境数量
ENV_SPACING = 3.5  # 环境间距 (米)

# 透明物体语义标签 —— 需要与 USD Prim 上打的标签一致
TRANSPARENT_LABEL = "transparent_obj"

# 场景路径（Linux 路径）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "USDFiles")
ENV_USD_PATH = os.path.join(USD_DIR, "env.usd")

# RMPFlow 配置
RMPFLOW_DIR = os.path.join(SCRIPT_DIR, "rmpflow_rm75")
RM75_URDF_PATH = os.path.join(RMPFLOW_DIR, "RM75-B-V.urdf")
ROBOT_DESCRIPTOR_PATH = os.path.join(RMPFLOW_DIR, "robot_descriptor.yaml")
RMPFLOW_CONFIG_PATH = os.path.join(RMPFLOW_DIR, "rm75_rmpflow_config.yaml")

# 控制参数
TARGET_HEIGHT_OFFSET = 0.25
ARRIVAL_XY_THRESHOLD = 0.02
ARRIVAL_Z_MIN = 0.23
ARRIVAL_Z_MAX = 0.28
STABLE_FRAMES = 30
TARGET_ORIENTATION = np.array([0.0, 0.0, 1.0, 0.0])

# 相机配置
WORLD_CAMERA_RESOLUTION = (640, 480)
WRIST_CAMERA_RESOLUTION = (320, 240)

# 数据采集
SAVE_DATA = True
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = os.path.join(SCRIPT_DIR, "/media/sysu/collected_data", TIMESTAMP)
STEPS_PER_EPISODE = 250
MAX_EPISODES = 501
# 注意：randomize_cubes() 会将 current_episode 从 0 加到 1，
# 所以实际运行 episode 数 = MAX_EPISODES - 1 = 500
SAVE_IMAGE_INTERVAL = 10

print("=" * 70)
print("RM75 Multi-Env Data Collection with Cameras + Transparent Object Mask")
print("=" * 70)
print(f"  Environments: {NUM_ENVS}")
print(f"  Spacing: {ENV_SPACING}m")
print(f"  World Camera: {WORLD_CAMERA_RESOLUTION}")
print(f"  Wrist Camera: {WRIST_CAMERA_RESOLUTION}")
print(f"  Transparent label: '{TRANSPARENT_LABEL}'")
print(f"  Steps per episode: {STEPS_PER_EPISODE}")
print(f"  Max episodes: {MAX_EPISODES}")
print(f"  Data directory: {DATA_DIR}")
print("=" * 70)


# =============================================================================
# 辅助函数
# =============================================================================

def get_random_ready_pose():
    """生成随机初始姿态"""
    base_pose = np.array([0.0, 0.5, 0.0, 0.8, 0.0, 1.0, 0.0, 0.0])
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
    # 新增：掩码目录
    os.makedirs(os.path.join(DATA_DIR, "world_camera_mask"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "wrist_camera_mask"), exist_ok=True)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"   + world_camera/        (RGB images)")
    print(f"   + wrist_camera/        (RGB images)")
    print(f"   + world_camera_mask/   (transparent object masks, uint16)")
    print(f"   + wrist_camera_mask/   (transparent object masks, uint16)")


def build_rmpflow_config():
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
        self._default_position = position
        self._default_orientation = orientation
        self._motion_policy.set_robot_base_pose(
            robot_position=position,
            robot_orientation=orientation
        )


# =============================================================================
# 多环境管理器（带相机 + 透明物体掩码）
# =============================================================================

class MultiEnvWithMaskManager:
    """多环境管理器 - 带相机图像采集 + 透明物体语义掩码"""

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
        self.world_cameras: List[Optional[Camera]] = []
        self.wrist_cameras: List[Optional[Camera]] = []

        # 状态跟踪
        self.is_paused = np.zeros(num_envs, dtype=bool)
        self.stable_counts = np.zeros(num_envs, dtype=int)
        self.last_cube_positions = np.zeros((num_envs, 3))

        # 数据缓冲（新增掩码字段）
        self.data_buffer = {
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_orientations': [],
            'target_positions': [],
            'cube_positions': [],
            'world_camera_images': [],
            'wrist_camera_images': [],
            'world_camera_masks': [],   # 新增：第三视角透明物体掩码
            'wrist_camera_masks': [],   # 新增：手腕相机透明物体掩码
            'env_ids': [],
            'step_ids': [],
            'episode_ids': [],
            'success': [],
            'distances': [],
        }
        self.current_episode = 0
        self.current_step = 0

        # 掩码统计：记录每帧透明物体像素数量（用于验证掩码质量）
        self._mask_pixel_counts = []

    # ------------------------------------------------------------------
    # 环境搭建
    # ------------------------------------------------------------------

    def setup_environments(self):
        """设置多个隔离环境"""
        print("\n🔧 Setting up isolated environments...")

        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        self.env_origins = np.zeros((self.num_envs, 3))
        for i in range(self.num_envs):
            row = i // num_per_row
            col = i % num_per_row
            self.env_origins[i] = [col * self.env_spacing, row * self.env_spacing, 0.0]

        define_prim("/World/envs", "Xform")
        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            self._create_single_env(i, env_path)
            print(f"   ✅ Created isolated env_{i} at {self.env_origins[i]}")

        print(f"\n✅ All {self.num_envs} isolated environments created!")
        return self.env_origins

    def _create_single_env(self, env_idx: int, env_path: str):
        """创建单个隔离环境"""
        origin = self.env_origins[env_idx]
        env_prim = define_prim(env_path, "Xform")
        env_xform = UsdGeom.Xformable(env_prim)
        env_xform.AddTranslateOp().Set(Gf.Vec3d(origin[0], origin[1], origin[2]))
        add_reference_to_stage(ENV_USD_PATH, env_path)

    def setup_semantic_labels(self):
        """
        为所有环境中的透明目标物体（Cube）添加语义标签。

        必须在 world.reset() 之前调用。
        标签 TRANSPARENT_LABEL 会被分割系统识别，无论物体材质是否透明。
        """
        print(f"\n🏷️  Setting up semantic labels ('{TRANSPARENT_LABEL}') for transparent objects...")
        labeled_count = 0
        for i in range(self.num_envs):
            cube_path = f"/World/envs/env_{i}/env/table_instanceable/Cube"
            prim = self.stage.GetPrimAtPath(cube_path)
            if prim.IsValid():
                add_labels(prim, labels=[TRANSPARENT_LABEL], instance_name="class")
                labeled_count += 1
            else:
                print(f"   ⚠️ Cube prim not found at {cube_path}")
        print(f"✅ Labeled {labeled_count}/{self.num_envs} transparent objects")

    # ------------------------------------------------------------------
    # 机器人初始化
    # ------------------------------------------------------------------

    def initialize_robots(self):
        """初始化所有机器人"""
        print("\n🤖 Initializing robots...")
        physics_dt = self.world.get_physics_dt()
        rmpflow_config = build_rmpflow_config()

        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"
            robot_path = f"{env_path}/env/table_instanceable/RM75_B_V"
            ee_path = f"{robot_path}/link_7"

            try:
                robot = self.world.scene.add(
                    SingleManipulator(
                        prim_path=robot_path,
                        name=f"rm75_robot_{i}",
                        end_effector_prim_path=ee_path
                    )
                )
                initial_joints = get_random_ready_pose()
                robot.set_joints_default_state(positions=initial_joints)

                controller = RM75RMPFlowController(
                    robot_articulation=robot,
                    physics_dt=physics_dt,
                    config=rmpflow_config
                )
                robot_pos, robot_ori = robot.get_world_pose()
                controller.update_base_pose(robot_pos, robot_ori)
                art_controller = robot.get_articulation_controller()

                self.robots.append(robot)
                self.controllers.append(controller)
                self.articulation_controllers.append(art_controller)
                print(f"   ✅ Robot {i} initialized")
            except Exception as e:
                print(f"   ❌ Failed to initialize robot {i}: {e}")
                raise

        print(f"\n✅ All {len(self.robots)} robots initialized!")

    # ------------------------------------------------------------------
    # 相机初始化
    # ------------------------------------------------------------------

    def initialize_cameras(self):
        """初始化所有相机（不含分割 annotator，分割需在 world.reset() 后启用）"""
        print("\n📷 Initializing cameras...")

        for i in range(self.num_envs):
            env_path = f"/World/envs/env_{i}"

            # World Camera
            world_cam_path = f"{env_path}/env/World_Camera"
            try:
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

            # Wrist Camera
            wrist_cam_path = f"{env_path}/env/table_instanceable/RM75_B_V/camera_link/Wrist_Camera"
            try:
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

    def setup_segmentation_annotators(self):
        """
        在 world.reset() 之后启用语义分割 annotator。

        必须在 world.reset() 之后调用，否则 annotator 可能不生效。
        启用后 get_current_frame() 会返回 "semantic_segmentation" 字段。
        """
        print("\n🎭 Setting up segmentation annotators (after world.reset())...")
        success_count = 0
        for i, cam in enumerate(self.world_cameras):
            if cam is not None:
                try:
                    cam.add_semantic_segmentation_to_frame()
                    success_count += 1
                except Exception as e:
                    print(f"   ⚠️ World camera {i} segmentation annotator failed: {e}")
        for i, cam in enumerate(self.wrist_cameras):
            if cam is not None:
                try:
                    cam.add_semantic_segmentation_to_frame()
                    success_count += 1
                except Exception as e:
                    print(f"   ⚠️ Wrist camera {i} segmentation annotator failed: {e}")
        print(f"✅ Segmentation annotators enabled on {success_count} cameras")

    # ------------------------------------------------------------------
    # 数据采集
    # ------------------------------------------------------------------

    def _extract_transparent_mask(self, frame: dict) -> Optional[np.ndarray]:
        """
        从 get_current_frame() 返回的帧数据中提取透明物体的二值掩码。

        返回值：
            uint16 数组 (H, W)，0=背景，1=透明物体（TRANSPARENT_LABEL）
            若分割数据不可用则返回 None
        """
        if "semantic_segmentation" not in frame:
            return None

        seg_data = frame["semantic_segmentation"]
        if not isinstance(seg_data, dict) or "data" not in seg_data:
            return None

        seg_map = seg_data["data"]  # shape (H, W), dtype uint32

        # 优先从 idToLabels 找到 TRANSPARENT_LABEL 对应的 ID
        transparent_ids = set()
        info = seg_data.get("info", {})
        id_to_labels = info.get("idToLabels", {})
        for id_str, label_dict in id_to_labels.items():
            if isinstance(label_dict, dict) and label_dict.get("class") == TRANSPARENT_LABEL:
                transparent_ids.add(int(id_str))

        if transparent_ids:
            # 精确：只标记匹配 TRANSPARENT_LABEL 的像素
            mask = np.zeros(seg_map.shape, dtype=np.uint16)
            for tid in transparent_ids:
                mask[seg_map == tid] = 1
        else:
            # 回退：若 API 未返回 idToLabels，则非背景(0)即目标
            # 适用于场景中只有一类标注物体的情况
            mask = (seg_map > 0).astype(np.uint16)

        return mask

    def get_cube_positions(self) -> np.ndarray:
        """获取所有环境中 Cube 的世界坐标"""
        positions = np.zeros((self.num_envs, 3))
        for i in range(self.num_envs):
            cube_path = f"/World/envs/env_{i}/env/table_instanceable/Cube"
            prim = self.stage.GetPrimAtPath(cube_path)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = mat.ExtractTranslation()
                positions[i] = [t[0], t[1], t[2]]
            else:
                positions[i] = self.env_origins[i] + np.array([0.3, 0.0, 0.05])
        return positions

    def randomize_cubes(self, x_range=(0.25, 0.40), y_range=(-0.20, 0.20)):
        """随机化所有 Cube 的位置"""
        for i in range(self.num_envs):
            cube_path = f"/World/envs/env_{i}/env/table_instanceable/Cube"
            prim = self.stage.GetPrimAtPath(cube_path)
            if prim.IsValid():
                if np.random.random() < 0.8:
                    x = np.random.uniform(0.28, 0.35)
                    y = np.random.uniform(-0.10, 0.10)
                else:
                    x = np.random.uniform(*x_range)
                    y = np.random.uniform(*y_range)
                z = 0.035

                xform = UsdGeom.Xformable(prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(x, y, z))
                        break

        print("   🤖 Resetting robots to random ready poses...")
        for i, robot in enumerate(self.robots):
            random_pose = get_random_ready_pose()
            robot.set_joint_positions(random_pose)
            robot.set_joint_velocities(np.zeros_like(random_pose))
            self.controllers[i].reset()

        self.is_paused[:] = False
        self.stable_counts[:] = 0
        self.current_step = 0
        self.current_episode += 1

        print(f"🎲 Episode {self.current_episode}: Randomized cube positions & Reset robots")

    def capture_camera_images(self) -> Tuple[List, List, List, List]:
        """
        采集所有相机的 RGB 图像 + 透明物体语义掩码。

        返回：
            world_images  : List[(H,W,3) uint8]  第三视角 RGB
            wrist_images  : List[(H,W,3) uint8]  手腕相机 RGB
            world_masks   : List[(H,W) uint16]   第三视角透明物体掩码
            wrist_masks   : List[(H,W) uint16]   手腕相机透明物体掩码
        """
        world_images, wrist_images = [], []
        world_masks, wrist_masks = [], []

        for i in range(self.num_envs):
            # ---- World Camera ----
            if self.world_cameras[i] is not None:
                try:
                    # RGB
                    rgba = self.world_cameras[i].get_rgba()
                    world_images.append(rgba[:, :, :3])

                    # 掩码：通过 get_current_frame() 获取语义分割数据
                    frame = self.world_cameras[i].get_current_frame()
                    mask = self._extract_transparent_mask(frame)
                    world_masks.append(mask)
                except Exception as e:
                    world_images.append(None)
                    world_masks.append(None)
            else:
                world_images.append(None)
                world_masks.append(None)

            # ---- Wrist Camera ----
            if self.wrist_cameras[i] is not None:
                try:
                    rgba = self.wrist_cameras[i].get_rgba()
                    wrist_images.append(rgba[:, :, :3])

                    frame = self.wrist_cameras[i].get_current_frame()
                    mask = self._extract_transparent_mask(frame)
                    wrist_masks.append(mask)
                except Exception as e:
                    wrist_images.append(None)
                    wrist_masks.append(None)
            else:
                wrist_images.append(None)
                wrist_masks.append(None)

        return world_images, wrist_images, world_masks, wrist_masks

    # ------------------------------------------------------------------
    # 主步进
    # ------------------------------------------------------------------

    def step(self) -> Dict:
        """执行一步控制并返回数据（含掩码）"""
        self.current_step += 1

        cube_positions = self.get_cube_positions()
        target_positions = cube_positions + np.array([0.0, 0.0, TARGET_HEIGHT_OFFSET])

        save_images = (self.current_step % SAVE_IMAGE_INTERVAL == 0)
        if save_images:
            world_images, wrist_images, world_masks, wrist_masks = self.capture_camera_images()
        else:
            world_images = wrist_images = None
            world_masks = wrist_masks = None

        step_data = {
            'joint_positions': [],
            'joint_velocities': [],
            'ee_positions': [],
            'ee_orientations': [],
            'distances': [],
            'world_images': world_images,
            'wrist_images': wrist_images,
            'world_masks': world_masks,   # 新增
            'wrist_masks': wrist_masks,   # 新增
        }

        for i, (robot, controller, art_ctrl) in enumerate(
            zip(self.robots, self.controllers, self.articulation_controllers)
        ):
            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            ee_pos, ee_ori = robot.end_effector.get_world_pose()

            step_data['joint_positions'].append(joint_pos)
            step_data['joint_velocities'].append(joint_vel)
            step_data['ee_positions'].append(ee_pos)
            step_data['ee_orientations'].append(ee_ori)

            distance = np.linalg.norm(target_positions[i] - ee_pos)
            step_data['distances'].append(distance)

            cube_moved = np.linalg.norm(cube_positions[i] - self.last_cube_positions[i]) > 0.01
            self.last_cube_positions[i] = cube_positions[i].copy()

            if self.is_paused[i]:
                if distance > 0.05 or cube_moved:
                    self.is_paused[i] = False
                    self.stable_counts[i] = 0
                else:
                    continue

            actions = controller.forward(
                target_end_effector_position=target_positions[i],
                target_end_effector_orientation=TARGET_ORIENTATION
            )
            art_ctrl.apply_action(actions)

            # 到达检测
            target_pos = target_positions[i]
            cube_pos = cube_positions[i]
            xy_distance = np.sqrt((ee_pos[0] - target_pos[0])**2 + (ee_pos[1] - target_pos[1])**2)
            rel_z = ee_pos[2] - cube_pos[2]
            is_height_ok = ARRIVAL_Z_MIN <= rel_z <= ARRIVAL_Z_MAX

            if xy_distance < ARRIVAL_XY_THRESHOLD and is_height_ok:
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

                stem = f"ep{self.current_episode:03d}_env{i}_step{self.current_step:04d}"

                # ---- 保存 RGB 图像 ----
                if world_images and world_images[i] is not None:
                    np.save(os.path.join(DATA_DIR, "world_camera", stem + ".npy"), world_images[i])

                if wrist_images and wrist_images[i] is not None:
                    np.save(os.path.join(DATA_DIR, "wrist_camera", stem + ".npy"), wrist_images[i])

                # ---- 保存透明物体掩码 ----
                if world_masks and world_masks[i] is not None:
                    mask = world_masks[i]
                    np.save(os.path.join(DATA_DIR, "world_camera_mask", stem + ".npy"), mask)
                    # 统计掩码质量（用于调试）
                    self._mask_pixel_counts.append(int(np.sum(mask > 0)))

                if wrist_masks and wrist_masks[i] is not None:
                    np.save(os.path.join(DATA_DIR, "wrist_camera_mask", stem + ".npy"), wrist_masks[i])

        return step_data

    # ------------------------------------------------------------------
    # 数据保存
    # ------------------------------------------------------------------

    def save_episode_data(self):
        """保存当前 episode 的数据"""
        if not SAVE_DATA or len(self.data_buffer['joint_positions']) == 0:
            return

        success_array = np.array(self.data_buffer['success'])
        env_ids_array = np.array(self.data_buffer['env_ids'])
        distances_array = np.array(self.data_buffer['distances'])
        ee_positions_array = np.array(self.data_buffer['ee_positions'])
        target_positions_array = np.array(self.data_buffer['target_positions'])

        success_stats = []
        for env_id in range(self.num_envs):
            mask_env = env_ids_array == env_id
            env_success = success_array[mask_env]
            env_distances = distances_array[mask_env]
            env_ee_positions = ee_positions_array[mask_env]
            env_target_positions = target_positions_array[mask_env]

            if len(env_success) > 0:
                success_rate = np.mean(env_success) * 100
                final_distance = env_distances[-1] if len(env_distances) > 0 else 999

                final_ee = env_ee_positions[-1]
                final_target = env_target_positions[-1]
                final_xy_dist = np.sqrt((final_ee[0] - final_target[0])**2 + (final_ee[1] - final_target[1])**2)
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

        # 保存关节和末端数据（npz）
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

        overall_success_rate = np.mean(success_array) * 100
        final_success_count = sum([s['is_final_success'] for s in success_stats])

        print(f"💾 Saved episode {self.current_episode} data: {len(self.data_buffer['joint_positions'])} samples")
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
        print(f"   Final success: {final_success_count}/{self.num_envs} environments")

        # 掩码质量报告
        if self._mask_pixel_counts:
            avg_px = np.mean(self._mask_pixel_counts)
            print(f"   Mask quality: avg transparent pixels/frame = {avg_px:.1f}")
            self._mask_pixel_counts.clear()

        for stats in success_stats:
            status = "✅" if stats['is_final_success'] else "❌"
            print(f"   Env {stats['env_id']}: {status} "
                  f"XY={stats['final_xy_distance']*100:.2f}cm, H={stats['final_z_distance']*100:.2f}cm")

        for key in self.data_buffer:
            self.data_buffer[key] = []


# =============================================================================
# 主程序
# =============================================================================

print("\n🌍 Creating World...")
my_world = World(stage_units_in_meters=1.0)

# 创建多环境管理器（带掩码）
manager = MultiEnvWithMaskManager(my_world, NUM_ENVS, ENV_SPACING)

# 1. 设置环境
manager.setup_environments()

# 2. 初始化机器人
manager.initialize_robots()

# 3. 初始化相机
manager.initialize_cameras()

# 4. 给透明物体打语义标签（必须在 world.reset() 之前）
manager.setup_semantic_labels()

# 5. 重置World
print("\n🔄 Resetting world...")
my_world.reset()

# 必须在 reset() 之后显式调用 play()，否则 is_playing() 始终为 False
my_world.play()

# 6. 启用分割 annotator（必须在 world.reset() 之后）
manager.setup_segmentation_annotators()

# 7. 随机化 Cube 位置（会将 current_episode 从 0 递增到 1）
manager.randomize_cubes()

# =============================================================================
# 主循环
# =============================================================================
print("\n" + "=" * 70)
print("🚀 Starting Multi-Env Data Collection with Cameras + Transparent Masks")
print("=" * 70)
print("\n💡 Tips:")
print("  - Each environment is isolated with walls")
print("  - Collecting: joints, EE poses, RGB images, transparent object masks")
print("  - Mask: uint16 (H,W), 0=background, 1=transparent object")
print("  - Press Ctrl+C to stop\n")

try:
    while simulation_app.is_running() and manager.current_episode < MAX_EPISODES:
        my_world.step(render=True)

        if my_world.is_playing():
            step_data = manager.step()

            if manager.current_step % 50 == 0:
                avg_dist = np.mean(step_data['distances'])
                reached = np.sum(manager.is_paused)
                print(f"📊 Ep {manager.current_episode} | Step {manager.current_step:4d} | "
                      f"Avg dist: {avg_dist:.4f}m | Reached: {reached}/{NUM_ENVS}")

            if manager.current_step >= STEPS_PER_EPISODE:
                print(f"✅ Episode {manager.current_episode} completed")
                manager.save_episode_data()

                if manager.current_episode < MAX_EPISODES:
                    manager.randomize_cubes()

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")

# 保存最后的数据
manager.save_episode_data()

print("\n✅ Data collection with masks completed!")
print(f"📁 Data saved to: {DATA_DIR}")
print(f"   - Joint/EE data:         episode_*.npz")
print(f"   - World camera RGB:      world_camera/*.npy       (H,W,3) uint8")
print(f"   - Wrist camera RGB:      wrist_camera/*.npy       (H,W,3) uint8")
print(f"   - World camera mask:     world_camera_mask/*.npy  (H,W) uint16, 0=bg 1=obj")
print(f"   - Wrist camera mask:     wrist_camera_mask/*.npy  (H,W) uint16, 0=bg 1=obj")

simulation_app.close()

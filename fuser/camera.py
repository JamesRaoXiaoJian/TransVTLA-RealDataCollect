import torch
from dataclasses import dataclass
from .contrastive import project_3d_to_2d_672_rlbench, project_3d_to_2d_672_franka_right, project_3d_to_2d_672_franka_front, project_3d_to_2d_672_world_camera, project_3d_to_2d_672_wrist_camera

@dataclass
class CameraParams:
    K: torch.Tensor 
    R: torch.Tensor  
    t: torch.Tensor  

# used camera parameters
CAMERA_CONFIGS = {
    "rlbench_front": CameraParams(
        K=torch.tensor([
            [-307.7174807,    0.0,         112.0],
            [   0.0,        -307.7174807,  112.0],
            [   0.0,           0.0,          1.0]
        ], dtype=torch.float32),
        R=torch.tensor([
            [ 1.19209290e-07, -4.22617942e-01, -9.06307936e-01],
            [-1.00000000e+00, -5.96046448e-07,  1.49011612e-07],
            [-5.66244125e-07,  9.06307936e-01, -4.22617912e-01]
        ], dtype=torch.float32),
        t=torch.tensor([1.34999919e+00, 3.71546562e-08, 1.57999933e+00], dtype=torch.float32)
    ),
    "franka_right": CameraParams(
        K=torch.tensor([
            [387.414794921875, 0.0, 319.47052001953125],   
            [0.0, 386.8714904785156, 241.13287353515625],  
            [0.0, 0.0, 1.0]                             
        ],dtype=torch.float32),
        R=torch.tensor([
            [ 0.91300858,  0.26157042, -0.31304353],
            [ 0.39730357, -0.7442472,   0.53688545],
            [-0.09254842, -0.61455433, -0.78342694]
        ], dtype=torch.float32),
        t=torch.tensor([0.8591219242556176, -0.5851783639922448, 0.7535876808722389], dtype=torch.float32)
    ),
    "franka_front": CameraParams(
        K=torch.tensor([
            [388.2638244628906, 0.0, 328.3757019042969],
            [0.0, 387.84130859375, 240.24295043945312],
            [0.0, 0.0, 1.0]
        ],dtype=torch.float32),
        R=torch.tensor([
            [-0.01750229,  0.95018522, -0.31119403],
            [ 0.99984609,  0.01625676, -0.00659609],
            [-0.0012085,  -0.31126158, -0.95032351],
        ], dtype=torch.float32),
        t=torch.tensor([0.8545415959817313, 0.5748472977587156, 1.0411478820663598], dtype=torch.float32)
    ),
    # Wrist RealSense 腕部相机（eye-on-hand）。
    # 注意：下面是旧 1280x720 SDK 内参和旧手眼结果。标准双 RealSense 采集使用
    # 848x480@30，必须用 calibration/hand_eye_calibration.py 重新导出的 K/R/t 替换。
    # R/t: 手眼标定结果（TSAI 方法，平均旋转误差 1.78°，平移误差 6.1mm）
    # 注意：eye-on-hand 的 R/t 随机械臂运动变化，此处为标定时刻的 T_base_cam
    # 实际使用时需实时计算：T_base_cam = T_base_ee_current @ T_ee_cam
    "wrist_camera": CameraParams(
        K=torch.tensor([
            [912.02, 0.0, 642.71],
            [0.0, 911.63, 370.44],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32),
        R=torch.tensor([
            [-0.07276065, -0.98413571, -0.16181059],
            [-0.99724014, 0.07419038, -0.00280307],
            [0.01476339, 0.16116007, -0.98681784],
        ], dtype=torch.float32),
        t=torch.tensor([0.42140274, 0.04006448, 0.38033282], dtype=torch.float32)
    ),
    # World RealSense D435 固定第三视角深度相机（eye-to-hand）。
    # 标定日期：2026-06-23。内参来自 D435 848x480 profile，畸变系数为全 0。
    # 标定质量：mean rot=3.78 deg, max rot=6.46 deg,
    # mean trans=36.5 mm, max trans=70.3 mm, reprojection=0.0526 px。
    # T_ee_target 显示棋盘格相对末端约 98.5 deg / 220 mm 偏移。
    # R/t 约定：T_base_cam；R 为相机坐标轴在机器人基座坐标系中的方向，t 为相机位置。
    "world_camera": CameraParams(
        K=torch.tensor([
            [602.716614, 0.0, 432.202942],
            [0.0, 603.465698, 249.251740],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32),
        R=torch.tensor([
            [-0.96943095,  0.09244450, -0.22728317],
            [ 0.20452160, -0.20729519, -0.95666066],
            [-0.13555273, -0.97390077,  0.18205148]
        ], dtype=torch.float32),
        t=torch.tensor([0.33983827, 0.83416543, 0.11930434], dtype=torch.float32)
    ),
}

def get_camera_params(config_name="default", device=None):
    # ==========================================================
    # [静默修复] 无论外部传什么(None, "default", ""), 只要无效
    # 就强制指定为 "rlbench_front"，确保第一阶段模仿学习不崩溃。
    # ==========================================================
    if config_name not in CAMERA_CONFIGS:
        config_name = "rlbench_front" 
    # ==========================================================
    
    # 原有的逻辑继续执行，不会再报错了
    params = CAMERA_CONFIGS[config_name]
    if device is not None:
        params.K = params.K.to(device)
        params.R = params.R.to(device)
        params.t = params.t.to(device)
    
    return params

# --- 找到 get_projection_func 函数 (约第 69 行) ---



PROJECT_FUNCS = {
    "rlbench_front": project_3d_to_2d_672_rlbench,
    "franka_right": project_3d_to_2d_672_franka_right,
    "franka_front": project_3d_to_2d_672_franka_front,
    "world_camera": project_3d_to_2d_672_world_camera,
    "wrist_camera": project_3d_to_2d_672_wrist_camera,
}

def get_projection_func(camera_name: str):
    # [同步修复] 同样的静默降级处理
    if camera_name not in PROJECT_FUNCS:
        camera_name = "rlbench_front"
        
    return PROJECT_FUNCS[camera_name]

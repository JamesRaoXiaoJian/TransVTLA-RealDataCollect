#!/usr/bin/env python3
"""
手眼标定脚本 (eye-on-hand)
标定板：11×8 棋盘格，方格尺寸 25mm
相机：RealSense D455（机械臂腕部，eye-on-hand）
机器人：锐曼 7-DOF 机械臂

标定的是 T_ee_cam（相机在末端执行器坐标系下的位姿）

使用方法：
    python fuser/hand_eye_calibration.py
    python fuser/hand_eye_calibration.py --mode load --data fuser/hand_eye_data.json

操作：
    SPACE : 采集一帧（同时记录机器人位姿 + 棋盘格检测）
    S     : 保存数据并执行手眼标定
    Q     : 退出（不保存）
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保能导入 Robotic_Arm 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import pyrealsense2 as rs
import json
import time
from realsense_standard import STANDARD_RS_FPS, STANDARD_RS_HEIGHT, STANDARD_RS_WIDTH

# ================================================================
# 标定板参数
# ================================================================
CHECKERBOARD = (10, 7)       # 内角点数（11×8 棋盘格 → 10×7 内角点）
SQUARE_SIZE = 0.025          # 方格尺寸 25mm = 0.025m
CHECKERBOARD_3D = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
CHECKERBOARD_3D[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
CHECKERBOARD_3D *= SQUARE_SIZE

# ================================================================
# 采集分辨率（与 collect_data.py 一致）
# ================================================================
RS_WIDTH = STANDARD_RS_WIDTH
RS_HEIGHT = STANDARD_RS_HEIGHT
RS_FPS = STANDARD_RS_FPS

# ================================================================
# 机器人连接
# ================================================================
ROBOT_IP = "192.168.31.92"
ROBOT_PORT = 8080

# ================================================================
# OpenCV 手眼标定方法
# ================================================================
HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

# 所有可用方法
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# ================================================================
# 保存路径
# ================================================================
OUTPUT_DIR = Path(__file__).parent
RECORD_FILE = OUTPUT_DIR / "hand_eye_data.json"


# ================================================================
# 机器人 API 封装
# ================================================================
class RobotArmAPI:
    """封装锐曼机械臂 API，提供 get_ee_pose() 接口"""

    def __init__(self, ip=ROBOT_IP, port=ROBOT_PORT):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(ip, port)
        print(f"已连接机械臂: {ip}:{port}")

    def get_ee_pose(self) -> np.ndarray:
        """获取末端执行器位姿，返回 4x4 齐次变换矩阵 T_base_ee
        使用 SDK 的 rm_algo_pos2matrix 正确处理欧拉角（非 cv2.Rodrigues）"""
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            print(f"警告: 获取机械臂状态失败 (code={code})")
            return np.eye(4)

        pose = state["pose"]  # [x, y, z, rx, ry, rz]

        # 使用 SDK 的 rm_algo_pos2matrix 正确转换（处理欧拉角约定）
        rm_matrix = self.robot.rm_algo_pos2matrix(pose)

        # rm_matrix_t 有 data[16]，按行优先排列
        T = np.array(rm_matrix.data, dtype=np.float64).reshape(4, 4)
        return T

    def disconnect(self):
        self.robot.rm_delete_robot_arm()
        print("机械臂已断开")


# ================================================================
# 棋盘格检测
# ================================================================
def detect_checkerboard(image, camera_matrix, dist_coeffs):
    """检测棋盘格角点，返回 (success, corners, rvec, tvec)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ret, rvec, tvec = cv2.solvePnP(
        CHECKERBOARD_3D, corners, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )
    return ret, corners, rvec, tvec


def pose_to_matrix(rvec, tvec):
    """旋转向量 + 平移向量 → 4x4 齐次变换矩阵"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


# ================================================================
# 实时采集
# ================================================================
def collect_data(robot_api, save_path=RECORD_FILE, serial: str | None = None):
    """实时采集手眼标定数据（RealSense + 机械臂）"""

    # ── 图片保存目录 ──
    img_dir = OUTPUT_DIR / "wrist_calib_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── 启动 RealSense ──
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    config.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16, RS_FPS)
    profile = pipeline.start(config)

    # 读取内参
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    print(f"RealSense 内参 ({RS_WIDTH}x{RS_HEIGHT}):")
    print(f"  fx={intr.fx:.2f}  fy={intr.fy:.2f}  cx={intr.ppx:.2f}  cy={intr.ppy:.2f}")
    print(f"  畸变: k1={intr.coeffs[0]:.4f} k2={intr.coeffs[1]:.4f}")
    print(f"  图片保存目录: {img_dir}")

    T_base_ee_list = []
    T_cam_target_list = []

    # 加载已有图片（如果有）
    existing_imgs = sorted(img_dir.glob("*.png"))
    if existing_imgs:
        print(f"\n发现已有 {len(existing_imgs)} 张标定图片，加载中...")
        for img_path in existing_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ok, corners, rvec, tvec = detect_checkerboard(img, camera_matrix, dist_coeffs)
            if ok:
                # 从文件名解析对应的 T_base_ee（保存在 JSON 中）
                pass  # 已有数据需要从 JSON 加载，这里仅提示
        print(f"提示：如需复用已有数据，请用 --mode load 加载 JSON 文件")

    print(f"\n按 SPACE 采集 | 按 S 保存并标定 | 按 Q/ESC 退出")
    print(f"手动移动机械臂到不同姿态，每次按 SPACE 采集\n")

    count = 0
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        success, corners, rvec, tvec = detect_checkerboard(image, camera_matrix, dist_coeffs)

        display = image.copy()
        if success:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, success)

        # 状态信息
        color = (0, 255, 0) if success else (0, 0, 255)
        status = "DETECTED" if success else "SEARCHING..."
        cv2.putText(display, f"Board: {status} | Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, "SPACE:capture  S:save+calibrate  Q/ESC:quit", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
        cv2.imshow("Hand-Eye Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            cv2.destroyAllWindows()
            pipeline.stop()
            return None, None, None, None

        elif key == ord(' '):  # 采集
            if not success:
                print(f"[{count}] 棋盘格未检测到，跳过")
                continue

            # 获取机器人位姿
            T_base_ee = robot_api.get_ee_pose()
            T_cam_target = pose_to_matrix(rvec, tvec)

            T_base_ee_list.append(T_base_ee)
            T_cam_target_list.append(T_cam_target)
            count += 1

            # 保存原图
            img_path = img_dir / f"hand_eye_{count:03d}.png"
            cv2.imwrite(str(img_path), image)

            ee_pos = T_base_ee[:3, 3]
            board_pos = tvec.ravel()
            print(f"[{count}] 末端: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) | "
                  f"棋盘格: ({board_pos[0]:.3f}, {board_pos[1]:.3f}, {board_pos[2]:.3f}) | "
                  f"saved→{img_path.name}")

        elif key in (ord('s'), ord('S')):  # 保存并标定
            if len(T_base_ee_list) < 5:
                print(f"数据不足（{len(T_base_ee_list)} 组，至少需要 5 组）")
                continue

            # 保存数据（含图片路径列表）
            data = {
                "resolution": {"width": RS_WIDTH, "height": RS_HEIGHT},
                "serial": serial,
                "T_base_ee_list": [m.tolist() for m in T_base_ee_list],
                "T_cam_target_list": [m.tolist() for m in T_cam_target_list],
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.tolist(),
                "image_files": [f"hand_eye_{i:03d}.png" for i in range(1, count + 1)],
            }
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n数据已保存: {save_path}（{len(T_base_ee_list)} 组）")
            print(f"图片已保存: {img_dir}")

            cv2.destroyAllWindows()
            pipeline.stop()
            return T_base_ee_list, T_cam_target_list, camera_matrix, dist_coeffs


# ================================================================
# 加载已保存数据
# ================================================================
def load_data(save_path=RECORD_FILE, recompute_targets=True):
    """
    加载已保存数据。recompute_targets=True 时从保存的图片重新检测棋盘格位姿（用最新 solvePnP 方法）。
    """
    with open(save_path, 'r') as f:
        data = json.load(f)

    T_base_ee_list = [np.array(m) for m in data["T_base_ee_list"]]
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])

    if recompute_targets and "image_files" in data:
        # 从保存的图片重新检测棋盘格
        img_dir = OUTPUT_DIR / "wrist_calib_images"
        T_cam_target_list = []
        valid_indices = []
        for i, fname in enumerate(data["image_files"]):
            img_path = img_dir / fname
            if not img_path.exists():
                print(f"  跳过 {fname}（文件不存在）")
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ok, corners, rvec, tvec = detect_checkerboard(img, camera_matrix, dist_coeffs)
            if ok:
                T_cam_target_list.append(pose_to_matrix(rvec, tvec))
                valid_indices.append(i)
            else:
                print(f"  {fname}: 棋盘格未检测到，跳过")

        # 只保留有效帧的 T_base_ee
        T_base_ee_list = [T_base_ee_list[i] for i in valid_indices]
        print(f"重新检测完成: {len(T_cam_target_list)}/{len(data['image_files'])} 帧有效")
    else:
        T_cam_target_list = [np.array(m) for m in data["T_cam_target_list"]]

    return T_base_ee_list, T_cam_target_list, camera_matrix, dist_coeffs


# ================================================================
# 手眼标定
# ================================================================
def calibrate_hand_eye(T_base_ee_list, T_cam_target_list):
    """尝试所有方法，返回 T_base_target 一致性最好的那个"""

    R_gripper2base = [m[:3, :3] for m in T_base_ee_list]
    t_gripper2base = [m[:3, 3] for m in T_base_ee_list]
    R_target2cam = [m[:3, :3] for m in T_cam_target_list]
    t_target2cam = [m[:3, 3] for m in T_cam_target_list]

    best_method = None
    best_T_ee_cam = None
    best_error = float('inf')

    print(f"\n尝试所有手眼标定方法（{len(T_base_ee_list)} 组数据）...")

    for name, method in HAND_EYE_METHODS.items():
        try:
            R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=method
            )
        except cv2.error as e:
            print(f"  {name:12s}: 失败 ({e})")
            continue

        T_ee_cam = np.eye(4)
        T_ee_cam[:3, :3] = R_cam2ee
        T_ee_cam[:3, 3] = t_cam2ee.ravel()

        # 计算 T_base_target 一致性误差
        T_base_targets = [T_base_ee_list[i] @ T_ee_cam @ T_cam_target_list[i]
                          for i in range(len(T_base_ee_list))]
        T_ref = T_base_targets[0]
        errors = []
        for T_bt in T_base_targets:
            R_err = T_ref[:3, :3] @ T_bt[:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)) * 180 / np.pi
            t_err = np.linalg.norm(T_ref[:3, 3] - T_bt[:3, 3]) * 1000  # mm
            errors.append(angle + t_err)  # 综合误差

        avg_err = np.mean(errors)
        t_dist = np.linalg.norm(t_cam2ee) * 1000  # mm
        print(f"  {name:12s}: 平均综合误差={avg_err:.1f}  t_cam2ee距离={t_dist:.1f}mm")

        if avg_err < best_error:
            best_error = avg_err
            best_method = name
            best_T_ee_cam = T_ee_cam

    if best_T_ee_cam is None:
        print("\n所有方法都失败了！")
        return None, None, None

    print(f"\n最佳方法: {best_method}（平均综合误差={best_error:.1f}）")

    R_cam2ee = best_T_ee_cam[:3, :3]
    t_cam2ee = best_T_ee_cam[:3, 3]

    print(f"\n{'='*60}")
    print(f"T_ee_cam（相机在末端坐标系下的位姿）:")
    print(f"R_cam2ee:\n{R_cam2ee}")
    print(f"t_cam2ee: {t_cam2ee.ravel()}")
    print(f"t距离: {np.linalg.norm(t_cam2ee)*1000:.1f}mm")

    return R_cam2ee, t_cam2ee, best_T_ee_cam


# ================================================================
# 重投影误差
# ================================================================
def compute_reprojection_error(T_base_ee_list, T_cam_target_list, T_ee_cam):
    """
    验证标定精度：棋盘格固定不动时，所有帧的 T_base_target 应该一致。
    T_base_target = T_base_ee @ T_ee_cam @ T_cam_target
    """
    T_base_targets = []
    for i in range(len(T_base_ee_list)):
        T_bt = T_base_ee_list[i] @ T_ee_cam @ T_cam_target_list[i]
        T_base_targets.append(T_bt)

    # 用第一帧作为参考
    T_ref = T_base_targets[0]
    errors_r = []
    errors_t = []

    for i, T_bt in enumerate(T_base_targets):
        R_err = T_ref[:3, :3] @ T_bt[:3, :3].T
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)) * 180 / np.pi
        t_err = np.linalg.norm(T_ref[:3, 3] - T_bt[:3, 3])
        errors_r.append(angle_err)
        errors_t.append(t_err)
        print(f"  Frame {i+1}: 旋转={angle_err:.2f}°  平移={t_err*1000:.1f}mm")

    avg_r = np.mean(errors_r)
    avg_t = np.mean(errors_t) * 1000
    print(f"\n平均: 旋转={avg_r:.2f}°  平移={avg_t:.1f}mm")
    if avg_r < 2.0 and avg_t < 10:
        print("✅  标定质量良好")
    elif avg_r < 5.0 and avg_t < 30:
        print("⚠️  可用，建议补充更多姿态")
    else:
        print("❌  误差偏大，建议重新采集")


# ================================================================
# 导出为 camera.py 格式
# ================================================================
def export_to_camera_config(T_ee_cam, T_base_ee_sample, camera_matrix=None):
    """
    导出 camera.py 的 R/t 格式
    R = R_cam_to_world（相机旋转到世界/基座坐标系）
    t = 相机在世界/基座坐标系中的位置
    """
    # T_base_cam = T_base_ee @ T_ee_cam
    T_base_cam = T_base_ee_sample @ T_ee_cam

    R_base_cam = T_base_cam[:3, :3]        # 相机坐标系轴在基座坐标系中的方向
    R_cam_to_world = R_base_cam             # = R_cam_to_base（世界=基座）
    t_cam_position = T_base_cam[:3, 3]      # 相机在基座坐标系中的位置

    print(f"\n{'='*60}")
    print(f"camera.py wrist_camera 格式（复制粘贴）")
    print(f"{'='*60}")
    print(f'    "wrist_camera": CameraParams(')
    print(f"        K=torch.tensor([  # {RS_WIDTH}x{RS_HEIGHT}")
    if camera_matrix is None:
        print(f"            [fx, 0.0, cx],")
        print(f"            [0.0, fy, cy],")
    else:
        print(f"            [{camera_matrix[0,0]:.8f}, 0.0, {camera_matrix[0,2]:.8f}],")
        print(f"            [0.0, {camera_matrix[1,1]:.8f}, {camera_matrix[1,2]:.8f}],")
    print(f"            [0.0, 0.0, 1.0]")
    print(f"        ], dtype=torch.float32),")
    print(f"        R=torch.tensor([")
    for row in R_cam_to_world:
        print(f"            [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
    print(f"        ], dtype=torch.float32),")
    print(f"        t=torch.tensor([{t_cam_position[0]:.8f}, {t_cam_position[1]:.8f}, {t_cam_position[2]:.8f}], dtype=torch.float32)")
    print(f"    ),")

    return R_cam_to_world, t_cam_position


# ================================================================
# 主函数
# ================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="手眼标定 (eye-on-hand)")
    parser.add_argument('--mode', type=str, default='live',
                        choices=['live', 'load'],
                        help='live: 实时采集 | load: 加载已有数据')
    parser.add_argument('--data', type=str, default=str(RECORD_FILE),
                        help='数据文件路径')
    parser.add_argument('--ip', type=str, default=ROBOT_IP, help='机械臂 IP')
    parser.add_argument('--port', type=int, default=ROBOT_PORT, help='机械臂端口')
    parser.add_argument('--serial', type=str, default=None, help='wrist RealSense serial number')
    args = parser.parse_args()

    if args.mode == 'live':
        # 连接机械臂
        robot_api = RobotArmAPI(ip=args.ip, port=args.port)

        T_base_ee_list, T_cam_target_list, camera_matrix, dist_coeffs = collect_data(
            robot_api=robot_api, save_path=args.data, serial=args.serial
        )
        robot_api.disconnect()

        if T_base_ee_list is None:
            print("用户取消")
            return
    else:
        T_base_ee_list, T_cam_target_list, camera_matrix, dist_coeffs = load_data(args.data)
        print(f"加载了 {len(T_base_ee_list)} 组数据")

    # 手眼标定
    R_cam2ee, t_cam2ee, T_ee_cam = calibrate_hand_eye(T_base_ee_list, T_cam_target_list)

    # 验证
    print(f"\n{'='*60}")
    print(f"重投影误差验证")
    print(f"{'='*60}")
    compute_reprojection_error(T_base_ee_list, T_cam_target_list, T_ee_cam)

    # 导出（用任意一帧的 T_base_ee 来计算 T_base_cam）
    # 注意：eye-on-hand 的 T_base_cam 随机械臂运动变化
    # camera.py 中存的是某一姿态下的值，实际使用时需要实时计算
    print(f"\n注意：eye-on-hand 配置下，相机位姿随机械臂运动变化。")
    print(f"camera.py 中存储的是标定时刻的 T_base_cam，实际使用时需要：")
    print(f"  T_base_cam = T_base_ee_current @ T_ee_cam")
    print(f"  其中 T_ee_cam 是标定结果（固定不变）")

    export_to_camera_config(T_ee_cam, T_base_ee_list[0], camera_matrix)


if __name__ == "__main__":
    main()

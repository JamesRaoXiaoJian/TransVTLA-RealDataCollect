#!/usr/bin/env python3
"""诊断：检查 T_base_target 在各帧之间的一致性"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import json

CHECKERBOARD = (10, 7)
SQUARE_SIZE = 0.025
CHECKERBOARD_3D = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
CHECKERBOARD_3D[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
CHECKERBOARD_3D *= SQUARE_SIZE

def main():
    data_path = Path(__file__).parent / "hand_eye_data.json"
    with open(data_path) as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    img_dir = Path(__file__).parent / "wrist_calib_images"

    # 加载 T_base_ee
    T_base_ee_list = [np.array(m) for m in data["T_base_ee_list"]]

    # 重新检测 T_cam_target
    T_cam_target_list = []
    valid_idx = []
    for i, fname in enumerate(data["image_files"]):
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not ret:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ok, rvec, tvec = cv2.solvePnP(CHECKERBOARD_3D, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        T_cam_target_list.append(T)
        valid_idx.append(i)

    T_base_ee_list = [T_base_ee_list[i] for i in valid_idx]
    print(f"有效帧数: {len(T_base_ee_list)}")

    # 用所有 5 种方法标定
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    R_g2b = [m[:3, :3] for m in T_base_ee_list]
    t_g2b = [m[:3, 3] for m in T_base_ee_list]
    R_t2c = [m[:3, :3] for m in T_cam_target_list]
    t_t2c = [m[:3, 3] for m in T_cam_target_list]

    for name, method in methods.items():
        try:
            R_c2e, t_c2e = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
        except:
            print(f"\n{name}: 失败")
            continue

        T_ee_cam = np.eye(4)
        T_ee_cam[:3, :3] = R_c2e
        T_ee_cam[:3, 3] = t_c2e.ravel()

        print(f"\n{'='*50}")
        print(f"方法: {name}")
        print(f"T_ee_cam t = {t_c2e.ravel()} ({np.linalg.norm(t_c2e)*1000:.1f}mm)")

        # 计算每帧的 T_base_target
        T_bt_list = []
        for i in range(len(T_base_ee_list)):
            T_bt = T_base_ee_list[i] @ T_ee_cam @ T_cam_target_list[i]
            T_bt_list.append(T_bt)

        # 打印前5帧的 T_base_target 位置
        print(f"\nT_base_target 位置（前5帧）:")
        for i in range(min(5, len(T_bt_list))):
            pos = T_bt_list[i][:3, 3]
            print(f"  Frame {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        # 检查一致性
        T_ref = T_bt_list[0]
        print(f"\n与第1帧对比:")
        for i in range(1, len(T_bt_list)):
            R_err = T_ref[:3, :3] @ T_bt_list[i][:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)) * 180 / np.pi
            t_err = np.linalg.norm(T_ref[:3, 3] - T_bt_list[i][:3, 3]) * 1000
            if i < 5 or angle > 10:
                print(f"  Frame {i+1}: 旋转={angle:.1f}°  平移={t_err:.1f}mm")

if __name__ == "__main__":
    main()

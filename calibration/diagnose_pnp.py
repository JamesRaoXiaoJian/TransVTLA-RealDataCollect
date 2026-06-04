#!/usr/bin/env python3
"""诊断：检查 solvePnP 在不同帧之间的 T_cam_target 一致性"""

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

def detect(image, K, dist, flag):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvec, tvec = cv2.solvePnP(CHECKERBOARD_3D, corners, K, dist, flags=flag)
    if not ret:
        return False, None, None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return True, T, corners

def main():
    data_path = Path(__file__).parent / "hand_eye_data.json"
    with open(data_path) as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    img_dir = Path(__file__).parent / "wrist_calib_images"
    img_files = data["image_files"]

    # 用两种方法检测前 5 帧
    flags = {
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "IPPE": cv2.SOLVEPNP_IPPE,
        "SQPNP": cv2.SOLVEPNP_SQPNP,
    }

    for flag_name, flag in flags.items():
        print(f"\n{'='*50}")
        print(f"方法: {flag_name}")
        print(f"{'='*50}")

        T_list = []
        for i, fname in enumerate(img_files[:5]):
            img = cv2.imread(str(img_dir / fname))
            if img is None:
                continue
            ok, T, corners = detect(img, K, dist, flag)
            if not ok:
                print(f"  {fname}: 检测失败")
                continue
            T_list.append(T)
            pos = T[:3, 3]
            print(f"  {fname}: t=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

        if len(T_list) >= 2:
            # 检查帧间差异
            T_ref = T_list[0]
            print(f"\n  与第1帧对比:")
            for j in range(1, len(T_list)):
                R_err = T_ref[:3, :3] @ T_list[j][:3, :3].T
                angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)) * 180 / np.pi
                t_err = np.linalg.norm(T_ref[:3, 3] - T_list[j][:3, 3]) * 1000
                print(f"    Frame {j+1}: 旋转差={angle:.2f}°  平移差={t_err:.1f}mm")

    # 检查棋盘格角点顺序一致性
    print(f"\n{'='*50}")
    print(f"角点顺序检查（第1帧 vs 第2帧）")
    print(f"{'='*50}")
    img1 = cv2.imread(str(img_dir / img_files[0]))
    img2 = cv2.imread(str(img_dir / img_files[1]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, c1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    _, c2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)
    if c1 is not None and c2 is not None:
        c1 = c1.reshape(-1, 2)
        c2 = c2.reshape(-1, 2)
        print(f"  Frame 1 前5角点: {c1[:5].round(1)}")
        print(f"  Frame 2 前5角点: {c2[:5].round(1)}")
        print(f"  Frame 1 最后5角点: {c1[-5:].round(1)}")
        print(f"  Frame 2 最后5角点: {c2[-5:].round(1)}")

if __name__ == "__main__":
    main()

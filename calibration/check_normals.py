#!/usr/bin/env python3
"""检查 T_cam_target 中棋盘格法向量的一致性"""

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

    print("帧 | 棋盘格法向量(z_cam方向) | 距离(m) | x范围 | y范围")
    print("-" * 80)

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
        # 棋盘格法向量在相机坐标系中的方向（z轴）
        normal = R[:, 2]
        dist_cam = np.linalg.norm(tvec)

        # 角点范围
        pts = corners.reshape(-1, 2)
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()

        print(f"{i+1:2d} | ({normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}) | {dist_cam:.3f} | {x_range:.0f}px | {y_range:.0f}px")

if __name__ == "__main__":
    main()

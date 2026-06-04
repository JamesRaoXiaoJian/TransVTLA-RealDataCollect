#!/usr/bin/env python3
"""验证 RealSense 内参：用 solvePnP 重投影检查"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

CHECKERBOARD = (10, 7)
SQUARE_SIZE = 0.025
CHECKERBOARD_3D = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
CHECKERBOARD_3D[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
CHECKERBOARD_3D *= SQUARE_SIZE

# 直接从 RealSense 读取内参
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

stream = profile.get_stream(rs.stream.color)
intr = stream.as_video_stream_profile().get_intrinsics()

K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
dist = np.array(intr.coeffs, dtype=np.float64)

print(f"RealSense 内参 ({intr.width}x{intr.height}):")
print(f"  fx={intr.fx:.4f}  fy={intr.fy:.4f}")
print(f"  cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
print(f"  畸变模型: {intr.model}")
print(f"  畸变系数: {intr.coeffs}")
print(f"  K:\n{K}")
print(f"  dist: {dist}")

# 拍一帧，检测棋盘格，检查重投影误差
frames = pipeline.wait_for_frames(timeout_ms=5000)
color_frame = frames.get_color_frame()
image = np.asanyarray(color_frame.get_data())
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ok, rvec, tvec = cv2.solvePnP(CHECKERBOARD_3D, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        # 重投影
        imgpoints, _ = cv2.projectPoints(CHECKERBOARD_3D, rvec, tvec, K, dist)
        errors = np.linalg.norm(corners.reshape(-1, 2) - imgpoints.reshape(-1, 2), axis=1)
        print(f"\n棋盘格检测成功，{len(corners)} 个角点")
        print(f"重投影误差: 平均={errors.mean():.4f}px  最大={errors.max():.4f}px  std={errors.std():.4f}px")

        R, _ = cv2.Rodrigues(rvec)
        print(f"\n棋盘格在相机坐标系中的位置:")
        print(f"  t = {tvec.ravel()}")
        print(f"  距离 = {np.linalg.norm(tvec):.4f}m ({np.linalg.norm(tvec)*1000:.1f}mm)")
        print(f"  R =\n{R}")
else:
    print("棋盘格未检测到！")

pipeline.stop()

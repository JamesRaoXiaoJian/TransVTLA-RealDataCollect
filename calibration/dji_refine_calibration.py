#!/usr/bin/env python3
"""
DJI 内参标定 - 筛选已有图片并重新标定
从 dji_calib_images/ 加载所有图片，过滤低质量帧，重新计算内参

使用方法：
    python fuser/dji_refine_calibration.py
    python fuser/dji_refine_calibration.py --min-coverage 0.10
"""

import json
from pathlib import Path

import cv2
import numpy as np

CHECKERBOARD = (10, 7)


def make_checkerboard_3d(square_size: float) -> np.ndarray:
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not ret:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def compute_coverage(corners, img_shape):
    h, w = img_shape[:2]
    pts = corners.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return (x_max - x_min) * (y_max - y_min) / (w * h)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DJI 内参标定 - 筛选重算")
    parser.add_argument("--square-size", type=float, default=0.025)
    parser.add_argument("--min-coverage", type=float, default=0.10, help="最小覆盖率阈值")
    parser.add_argument("--save", action="store_true", help="保存结果覆盖原有文件")
    args = parser.parse_args()

    img_dir = Path(__file__).parent / "dji_calib_images"
    if not img_dir.exists():
        print(f"图片目录不存在: {img_dir}")
        return

    checkerboard_3d = make_checkerboard_3d(args.square_size)
    img_paths = sorted(img_dir.glob("*.png"))

    if not img_paths:
        print("没有找到图片")
        return

    print(f"加载 {len(img_paths)} 张图片，筛选 coverage > {args.min_coverage:.0%}...\n")

    # 逐张分析
    records = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ok, corners = detect_corners(img)
        if not ok:
            records.append({"path": p, "ok": False, "coverage": 0})
            continue
        coverage = compute_coverage(corners, img.shape)
        records.append({
            "path": p, "ok": True, "coverage": coverage,
            "corners": corners, "shape": img.shape
        })

    # 统计
    detected = [r for r in records if r["ok"]]
    filtered = [r for r in detected if r["coverage"] >= args.min_coverage]
    rejected = [r for r in detected if r["coverage"] < args.min_coverage]

    print(f"检测到棋盘格: {len(detected)} 张")
    print(f"被过滤 (coverage < {args.min_coverage:.0%}): {len(rejected)} 张")
    for r in rejected:
        print(f"  ✗ {r['path'].name}  coverage={r['coverage']:.1%}")
    print(f"保留: {len(filtered)} 张")
    for r in filtered:
        print(f"  ✓ {r['path'].name}  coverage={r['coverage']:.1%}")

    if len(filtered) < 3:
        print(f"\n有效图片不足 3 张，无法标定")
        return

    # 标定
    obj_points = [checkerboard_3d.copy() for _ in filtered]
    img_points = [r["corners"] for r in filtered]
    h, w = filtered[0]["shape"][:2]

    print(f"\n使用 {len(filtered)} 张图片标定...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    # 逐帧误差
    errors = []
    for i in range(len(filtered)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
        errors.append(err)

    print(f"\n{'='*50}")
    print(f"标定结果")
    print(f"{'='*50}")
    print(f"RMS = {rms:.4f}")
    print(f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
    print(f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    if dist is not None:
        print(f"k1={dist[0,0]:.4f}  k2={dist[0,1]:.4f}  k3={dist[0,4]:.4f}")
        print(f"p1={dist[0,2]:.6f}  p2={dist[0,3]:.6f}")

    print(f"\n逐帧误差:")
    for i, (r, err) in enumerate(zip(filtered, errors)):
        print(f"  {r['path'].name}: {err:.4f}")
    print(f"  平均={np.mean(errors):.4f}  最大={np.max(errors):.4f}")

    if rms < 0.5:
        print("\n✅  标定质量良好")
    elif rms < 1.0:
        print("\n⚠️  可用，建议补充更多角度的图片")
    else:
        print("\n❌  RMS 偏高，建议重新拍摄")

    # 保存
    if args.save:
        save_dir = Path(__file__).parent
        out = {
            "camera": "dji_action5",
            "resolution": {"width": w, "height": h},
            "rms_reprojection_error": float(rms),
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "num_images": len(filtered),
            "min_coverage_threshold": args.min_coverage,
        }
        json_path = save_dir / "dji_action5_intrinsics.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        npz_path = save_dir / "dji_action5_intrinsics.npz"
        np.savez(npz_path, camera_matrix=K, dist_coeffs=dist)
        print(f"\n已覆盖保存: {json_path}")
        print(f"已覆盖保存: {npz_path}")

        print(f"\n{'='*60}")
        print('    "world_camera": CameraParams(')
        print("        K=torch.tensor([")
        print(f"            [{K[0,0]:.6f}, 0.0, {K[0,2]:.6f}],")
        print(f"            [0.0, {K[1,1]:.6f}, {K[1,2]:.6f}],")
        print("            [0.0, 0.0, 1.0]")
        print("        ], dtype=torch.float32),")
        print("        R=torch.tensor([  # TODO: 外参标定")
        print("            [1.0, 0.0, 0.0],")
        print("            [0.0, 1.0, 0.0],")
        print("            [0.0, 0.0, 1.0]")
        print("        ], dtype=torch.float32),")
        print("        t=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)")
        print("    ),")
        print("=" * 60)
    else:
        print("\n结果未保存，加 --save 覆盖保存")


if __name__ == "__main__":
    main()

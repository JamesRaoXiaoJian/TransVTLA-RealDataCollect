#!/usr/bin/env python3
"""
DJI Action 5 Pro 内参标定脚本 v2
标定板：11×8 棋盘格（10×7 内角点），方格尺寸 25mm
相机：DJI Action 5 Pro（USB 模式，OpenCV DirectShow）

改进：
- 关闭自动采集，全部手动
- 独立采集线程，画面不卡
- 标定异步执行，界面实时显示结果
- 每张采集图保存到本地，方便复用

操作：
    SPACE : 拍照
    C     : 计算内参
    S     : 保存结果
    Q     : 退出
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# ================================================================
# 标定板参数
# ================================================================
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


def run_calibration(obj_points, img_points, img_wh):
    """后台线程执行标定，返回结果 dict"""
    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_wh, None, None
        )
        errors = []
        for i in range(len(obj_points)):
            proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            errors.append(err)
        return {
            "rms": rms, "K": K, "dist": dist,
            "mean_err": np.mean(errors), "max_err": np.max(errors),
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def draw_ui(display, corners_detected, n_captured, result=None):
    """绘制界面：状态栏 + 右上角标定结果"""
    h, w = display.shape[:2]

    # 棋盘格角点
    if corners_detected is not None:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners_detected, True)

    # 左上角状态
    status = "DETECTED" if corners_detected is not None else "SEARCHING..."
    color = (0, 255, 0) if corners_detected is not None else (0, 0, 255)
    cv2.putText(display, f"Board: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"Captured: {n_captured}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 右上角标定结果
    if result is not None and result.get("success"):
        K = result["K"]
        texts = [
            f"RMS: {result['rms']:.4f}",
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}",
            f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}",
            f"mean_err: {result['mean_err']:.4f}",
        ]
        rms_color = (0, 255, 0) if result["rms"] < 0.5 else (0, 165, 255) if result["rms"] < 1.0 else (0, 0, 255)
        # 背景框
        box_h = len(texts) * 28 + 10
        box_w = 280
        cv2.rectangle(display, (w - box_w - 5, 5), (w - 5, box_h), (0, 0, 0), -1)
        cv2.rectangle(display, (w - box_w - 5, 5), (w - 5, box_h), rms_color, 2)
        for i, txt in enumerate(texts):
            c = rms_color if i == 0 else (255, 255, 255)
            cv2.putText(display, txt, (w - box_w, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
    elif result is not None and not result.get("success"):
        cv2.putText(display, f"Calib FAILED: {result.get('error','')}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 底部操作提示
    cv2.putText(display, "SPACE:capture  C:calibrate  S:save  Q/ESC:quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="DJI Action 5 Pro 内参标定 v2")
    parser.add_argument("--index", type=int, default=0, help="OpenCV 摄像头索引")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="采集高度")
    parser.add_argument("--square-size", type=float, default=0.025, help="方格尺寸(m)")
    parser.add_argument("--save-dir", type=str, default=None, help="保存目录")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else Path(__file__).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # 图片保存目录
    img_dir = save_dir / "dji_calib_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    checkerboard_3d = make_checkerboard_3d(args.square_size)

    # ── 打开摄像头 ──
    print(f"打开 DJI 相机 (index={args.index}, {args.width}x{args.height})...")
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(10):
        cap.grab()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"实际分辨率: {actual_w}x{actual_h}")
    print(f"图片保存目录: {img_dir}")

    # ── 标定数据 ──
    obj_points = []
    img_points = []
    calib_result = None  # 最新标定结果
    calib_lock = threading.Lock()
    calib_thread = None

    # 加载已有图片（如果有）
    existing_imgs = sorted(img_dir.glob("*.png"))
    if existing_imgs:
        print(f"\n发现已有 {len(existing_imgs)} 张标定图片，加载中...")
        for img_path in existing_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ok, corners = detect_corners(img)
            if ok:
                obj_points.append(checkerboard_3d.copy())
                img_points.append(corners.copy())
        print(f"成功加载 {len(obj_points)} 张有效图片")

    # ── 采集线程 ──
    latest_frame = [None]
    frame_lock = threading.Lock()
    running = [True]

    def capture_loop():
        while running[0]:
            ret, f = cap.read()
            if ret:
                with frame_lock:
                    latest_frame[0] = f.copy()
            else:
                time.sleep(0.01)

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    print(f"\n已有 {len(obj_points)} 张 | SPACE:拍照 | C:标定 | S:保存 | Q:退出\n")

    # ── 主显示循环 ──
    while running[0]:
        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.01)
            continue

        # 检测角点（每帧都检测，用于预览）
        success, corners = detect_corners(frame)

        # 绘制 UI
        display = frame.copy()
        with calib_lock:
            result = calib_result
        draw_ui(display, corners if success else None, len(obj_points), result)

        cv2.namedWindow("DJI Intrinsic Calibration", cv2.WINDOW_NORMAL)
        cv2.imshow("DJI Intrinsic Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        # 大小写兼容
        if key in (ord('q'), ord('Q'), 27):  # q / Q / ESC
            running[0] = False
            break

        elif key == ord(' '):  # 拍照
            if success:
                idx = len(obj_points) + 1
                obj_points.append(checkerboard_3d.copy())
                img_points.append(corners.copy())
                # 保存原图
                img_path = img_dir / f"calib_{idx:03d}.png"
                cv2.imwrite(str(img_path), frame)
                print(f"[拍照] #{idx}  coverage={compute_coverage(corners, frame.shape):.1%}  saved→{img_path.name}")
            else:
                print("[拍照] 棋盘格未检测到，跳过")

        elif key in (ord('c'), ord('C')):  # 标定
            n = len(obj_points)
            if n < 3:
                print(f"数据不足（{n} 张，至少 3 张）")
                continue

            if calib_thread is not None and calib_thread.is_alive():
                print("标定正在进行中...")
                continue

            print(f"\n开始标定（{n} 张，后台执行）...")

            def do_calibrate():
                nonlocal calib_result
                result = run_calibration(obj_points, img_points, (actual_w, actual_h))
                with calib_lock:
                    calib_result = result
                if result["success"]:
                    print(f"✅ RMS={result['rms']:.4f}  "
                          f"fx={result['K'][0,0]:.1f} fy={result['K'][1,1]:.1f}  "
                          f"mean_err={result['mean_err']:.4f}")
                else:
                    print(f"❌ 标定失败: {result.get('error')}")

            calib_thread = threading.Thread(target=do_calibrate, daemon=True)
            calib_thread.start()

        elif key in (ord('s'), ord('S')):  # 保存
            with calib_lock:
                result = calib_result
            if result is None or not result.get("success"):
                print("尚未标定或标定失败，按 C 先执行标定")
                continue

            K = result["K"]
            dist = result["dist"]
            rms = result["rms"]

            # JSON
            out = {
                "camera": "dji_action5",
                "resolution": {"width": actual_w, "height": actual_h},
                "rms_reprojection_error": float(rms),
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            }
            json_path = save_dir / "dji_action5_intrinsics.json"
            with open(json_path, "w") as f:
                json.dump(out, f, indent=2)

            npz_path = save_dir / "dji_action5_intrinsics.npz"
            np.savez(npz_path, camera_matrix=K, dist_coeffs=dist)

            print(f"\n已保存: {json_path}")
            print(f"已保存: {npz_path}")

            # 打印 camera.py 格式
            print("\n" + "=" * 60)
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

    cap.release()
    cv2.destroyAllWindows()
    print("退出")


if __name__ == "__main__":
    main()

"""
Phase2 RLDS 数据集构建脚本
============================
将 TransVTLA Phase2 真机采集数据转换为 RLDS (TFDS) 格式。

数据源结构:
  sessions/{object_type}_sampleN/session_YYYYMMDD_HHMMSS/
    ├── world_camera/
    │   ├── rgb/              0001.jpg ~ NNNN.jpg  (20 Hz)
    │   └── depth/            0001.png ~ NNNN.png  (20 Hz, 16-bit)
    ├── wrist_camera/
    │   ├── rgb/              0001.jpg ~ NNNN.jpg  (20 Hz)
    │   └── depth/            0001.png ~ NNNN.png  (20 Hz, 16-bit)
    ├── robot_state/          robot_state.csv      (200 Hz)
    └── preprocessed_pressure/ {session}.npz       (滑窗后的触觉帧)

核心策略:
  - 以视觉帧 (20 Hz) 为基准时间轴
  - 机械臂状态 (200 Hz) 通过最近邻时间戳下采样对齐到 20 Hz
  - 触觉数据 (npz, shape=(T, 16, 20)) 通过帧索引下采样对齐到 20 Hz
  - 注入物体属性字段，与 util/prompting.py 的 build_object_attr_inputs 对齐
"""

import os
import sys
import csv
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow_datasets as tfds
from session_schema import common_frame_stems, resolve_session_layout, rgb_path

# ============================================================
# 采样率常量
# ============================================================
VISUAL_FPS = 20
TACTILE_FPS = 200
ROBOT_ARM_FPS = 200

# ============================================================
# 触觉窗口参数 (与预处理脚本一致)
# ============================================================
TACTILE_WINDOW = 16   # npz data 每个帧的 taxel 数
TACTILE_TIME = 20     # npz data 每个帧的时间步数
TACTILE_SHAPE = (TACTILE_WINDOW, TACTILE_TIME)  # (16, 20)

# ============================================================
# 物体属性映射表
# 用于填充 prompt 注入模块所需的语义属性
# 数值应根据实际物理标定调整
# ============================================================
OBJECT_ATTR_MAP: Dict[str, Dict[str, Any]] = {
    "colacup": {
        "material": "plastic",
        "fragility": 0.7,
        "slip_risk": 0.6,
        "depth_uncertainty": 0.8,   # 透明/半透明
        "force_min": 0.5,
        "force_max": 3.0,
        "instruction": "pick up the cola cup",
    },
    "cube": {
        "material": "plastic",
        "fragility": 0.1,
        "slip_risk": 0.3,
        "depth_uncertainty": 0.2,
        "force_min": 1.0,
        "force_max": 8.0,
        "instruction": "pick up the cube",
    },
    "cup": {
        "material": "ceramic",
        "fragility": 0.5,
        "slip_risk": 0.4,
        "depth_uncertainty": 0.3,
        "force_min": 0.8,
        "force_max": 5.0,
        "instruction": "pick up the cup",
    },
    "softbottle": {
        "material": "silicone",
        "fragility": 0.2,
        "slip_risk": 0.8,
        "depth_uncertainty": 0.5,
        "force_min": 0.3,
        "force_max": 2.0,
        "instruction": "pick up the soft bottle",
    },
}

# ============================================================
# 提示词注入工具函数 (与 util/prompting.py 逻辑一致)
# ============================================================
OBJ_TYPE_VOCAB_SIZE = 128
MATERIAL_VOCAB_SIZE = 64
PHASE_VOCAB_SIZE = 32


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value = str(value).strip().lower()
    return value if value else default


def stable_string_to_id(value: Any, vocab_size: int) -> int:
    text = _normalize_text(value)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % vocab_size


def format_force_range(force_min: float, force_max: float) -> str:
    return f"{force_min:.3f}-{force_max:.3f}"


def build_action_prompt(
    instruction: str,
    obj_type: str,
    material: str,
    phase: str,
    force_range: str,
) -> str:
    return (
        f"What action should the robot take to {instruction}?\n"
        f"CTX:OBJ={obj_type}|MAT={material}|PHASE={phase}|FORCE={force_range}"
    )


# ============================================================
# 时间戳对齐工具
# ============================================================


def load_robot_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 robot_state.csv，返回 (timestamps_us, poses) 数组。
    poses shape = (N, 6)，列为 pose_1 ~ pose_6。
    """
    timestamps = []
    poses = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 14:
                continue
            timestamps.append(int(row[0]))
            poses.append([float(v) for v in row[8:14]])  # pose_1..pose_6
    return np.array(timestamps, dtype=np.int64), np.array(poses, dtype=np.float32)


def load_tactile_npz(npz_path: Path) -> np.ndarray:
    """读取预处理后的触觉 npz，返回 data 数组 shape=(T, 16, 20)。"""
    data = np.load(str(npz_path))
    return data["data"].astype(np.float32)


def nearest_indices(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    """对于 target_ts 中每个时间戳，找到 source_ts 中最近邻的索引。"""
    # source_ts 必须有序
    idx = np.searchsorted(source_ts, target_ts, side="left")
    idx = np.clip(idx, 1, len(source_ts) - 1)
    left = source_ts[idx - 1]
    right = source_ts[idx]
    pick_left = (target_ts - left) <= (right - target_ts)
    idx = np.where(pick_left, idx - 1, idx)
    return idx


def resample_by_index_step(total_source_frames: int, total_target_frames: int) -> np.ndarray:
    """将 source 维度均匀下采样到 target 帧数，返回 target 长度的索引数组。"""
    if total_target_frames <= 0 or total_source_frames <= 0:
        return np.array([], dtype=np.int64)
    indices = np.linspace(0, total_source_frames - 1, total_target_frames).astype(np.int64)
    return np.clip(indices, 0, total_source_frames - 1)


# ============================================================
# TFDS Builder
# ============================================================

_DESCRIPTION = """
TransVTLA Phase2 真机数据集 (RLDS 格式)。
- 以视觉帧 (20Hz) 为基准
- 包含触觉/压力数据 (16x20 窗口)
- 包含物体属性字段用于提示词注入
"""

_VERSION = tfds.core.Version("2.0.0")


class Phase2RobotData(tfds.core.GeneratorBasedBuilder):
    """Phase2 真机数据集 RLDS 转换器。"""

    VERSION = _VERSION
    RELEASE_NOTES = {
        "2.0.0": "Phase2: 加入触觉数据 + 物体属性提示词注入。",
    }

    def __init__(self, data_dir=None, source_root=None, **kwargs):
        if source_root is None:
            raise ValueError(
                "source_root 不能为空，请通过构造函数或命令行 --source-root 指定数据源路径"
            )
        self.source_root = Path(source_root)
        super().__init__(data_dir=data_dir, **kwargs)

    # ----------------------------------------------------------
    # RLDS Feature Schema
    # ----------------------------------------------------------
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "primary_image": tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, doc="World RealSense 主视角"
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, doc="RealSense 腕部视角"
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32, doc="6D Pose + 1D Gripper"
                        ),
                        "tactile": tfds.features.Tensor(
                            shape=(16, 20), dtype=np.float32,
                            doc="触觉/压力窗口 [taxel, timestep]"
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="6D 位姿增量 + 1D 夹爪增量"
                    ),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    # ---- 物体属性 (prompt injection) ----
                    "object_type": tfds.features.Text(),
                    "material": tfds.features.Text(),
                    "phase": tfds.features.Text(),
                    "fragility": tfds.features.Scalar(dtype=np.float32),
                    "slip_risk": tfds.features.Scalar(dtype=np.float32),
                    "depth_uncertainty": tfds.features.Scalar(dtype=np.float32),
                    "force_min": tfds.features.Scalar(dtype=np.float32),
                    "force_max": tfds.features.Scalar(dtype=np.float32),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "episode_id": tfds.features.Scalar(dtype=np.int64),
                    "source_file": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
        )

    # ----------------------------------------------------------
    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(self.source_root)}

    # ----------------------------------------------------------
    # 核心数据生成
    # ----------------------------------------------------------
    def _generate_examples(self, root_path: Path) -> Iterator[Tuple[str, Dict]]:
        # 两级遍历: {object_type}_sampleN / session_*
        sample_dirs = sorted([
            d for d in root_path.iterdir()
            if d.is_dir() and "_sample" in d.name
        ])
        if not sample_dirs:
            print(f"[WARN] 未找到任何 sample 目录于 {root_path}")
            return

        ep_idx = 0
        total_sessions = sum(len(list(d.glob("session_*"))) for d in sample_dirs)
        processed = 0

        for sample_dir in sample_dirs:
            # 从目录名提取物体类型: "colacup_sample3" → "colacup"
            obj_type_raw = sample_dir.name.split("_sample")[0]
            attrs = OBJECT_ATTR_MAP.get(obj_type_raw, {
                "material": "unknown",
                "fragility": 0.0,
                "slip_risk": 0.0,
                "depth_uncertainty": 0.0,
                "force_min": 0.0,
                "force_max": 0.0,
                "instruction": f"pick up the {obj_type_raw}",
            })

            obj_type_text = _normalize_text(obj_type_raw)
            material_text = _normalize_text(attrs["material"])
            phase_text = _normalize_text("grasp")
            fragility = float(attrs["fragility"])
            slip_risk = float(attrs["slip_risk"])
            depth_uncertainty = float(attrs["depth_uncertainty"])
            force_min = float(attrs["force_min"])
            force_max = float(attrs["force_max"])
            force_range_str = format_force_range(force_min, force_max)
            instruction = _normalize_text(attrs.get("instruction", f"pick up the {obj_type_raw}"))

            # 构建 prompt (与 util/prompting.py build_action_prompt 格式一致)
            action_prompt = build_action_prompt(
                instruction, obj_type_text, material_text, phase_text, force_range_str
            )

            session_dirs = sorted(sample_dir.glob("session_*"))
            for session_dir in session_dirs:
                processed += 1
                print(
                    f"  [{processed}/{total_sessions}] {sample_dir.name}/{session_dir.name}",
                    end="\r",
                )

                steps = self._process_session(
                    session_dir, obj_type_text, material_text, phase_text,
                    fragility, slip_risk, depth_uncertainty, force_min, force_max,
                    action_prompt,
                )
                if steps is None or len(steps) < 2:
                    continue

                yield f"phase2_{ep_idx:05d}", {
                    "steps": steps,
                    "episode_metadata": {
                        "episode_id": ep_idx,
                        "source_file": f"{sample_dir.name}/{session_dir.name}",
                    },
                }
                ep_idx += 1

        print(f"\n转换完成, 共 {ep_idx} 个 episode。")

    # ----------------------------------------------------------
    def _process_session(
        self,
        session_dir: Path,
        obj_type_text: str,
        material_text: str,
        phase_text: str,
        fragility: float,
        slip_risk: float,
        depth_uncertainty: float,
        force_min: float,
        force_max: float,
        action_prompt: str,
    ) -> Optional[List[Dict]]:
        """处理单个 session，返回 step 列表，以视觉帧为基准。"""

        try:
            layout = resolve_session_layout(session_dir)
        except FileNotFoundError:
            return None

        csv_path = session_dir / "robot_state" / "robot_state.csv"
        npz_path = session_dir / "preprocessed_pressure" / f"{session_dir.name}.npz"

        # 1) 收集视觉帧: 按文件名排序，获取帧数 N_vis
        stems = common_frame_stems(layout)
        if not stems:
            return None
        n_vis = len(stems)

        # 2) 加载机械臂状态 CSV
        if not csv_path.exists():
            return None
        robot_ts, robot_poses = load_robot_csv(csv_path)
        if len(robot_ts) == 0:
            return None

        # 3) 加载触觉 npz
        has_tactile = npz_path.exists()
        tactile_data = None
        if has_tactile:
            tactile_data = load_tactile_npz(npz_path)

        # 4) 以视觉帧为基准进行时间对齐
        #    视觉帧: 均匀分布在整个 recording 时间跨度上
        #    通过 robot CSV 的时间戳计算 recording 跨度
        rec_start = robot_ts[0]
        rec_end = robot_ts[-1]
        rec_duration_us = rec_end - rec_start
        if rec_duration_us <= 0:
            return None

        # 生成视觉帧的虚拟时间戳 (均匀间隔 1/VISUAL_FPS)
        vis_interval_us = int(1e6 / VISUAL_FPS)
        vis_timestamps = np.array(
            [rec_start + i * vis_interval_us for i in range(n_vis)],
            dtype=np.int64,
        )

        # 将机械臂状态下采样到视觉帧: 最近邻匹配
        robot_indices = nearest_indices(vis_timestamps, robot_ts)
        aligned_poses = robot_poses[robot_indices]  # (n_vis, 6)

        # 将触觉数据下采样到视觉帧: 均匀索引映射
        aligned_tactile = None
        if tactile_data is not None and len(tactile_data) > 0:
            tactile_indices = resample_by_index_step(len(tactile_data), n_vis)
            aligned_tactile = tactile_data[tactile_indices]  # (n_vis, 16, 20)

        # 5) 构建每个 step
        steps = []
        for i, stem in enumerate(stems):
            # 图像
            p_img = self._load_image(rgb_path(layout.world, stem))
            w_img = self._load_image(rgb_path(layout.wrist, stem))
            if p_img is None or w_img is None:
                continue

            # 机械臂状态: 6D pose + 1D gripper (暂填 0)
            state_vec = np.concatenate([aligned_poses[i], [0.0]]).astype(np.float32)

            # Action: 下一帧 pose 减去当前帧 pose
            if i < n_vis - 1:
                delta = aligned_poses[i + 1] - aligned_poses[i]
                action_vec = np.concatenate([delta, [0.0]]).astype(np.float32)
            else:
                action_vec = np.zeros(7, dtype=np.float32)

            # 触觉数据
            tactile_frame = (
                aligned_tactile[i] if aligned_tactile is not None
                else np.zeros(TACTILE_SHAPE, dtype=np.float32)
            )

            steps.append({
                "observation": {
                    "primary_image": p_img,
                    "wrist_image": w_img,
                    "state": state_vec,
                    "tactile": tactile_frame,
                },
                "action": action_vec,
                "reward": 0.0,
                "is_first": (i == 0),
                "is_last": (i == n_vis - 1),
                "is_terminal": (i == n_vis - 1),
                "language_instruction": action_prompt,
                # 物体属性 (prompt injection)
                "object_type": obj_type_text,
                "material": material_text,
                "phase": phase_text,
                "fragility": fragility,
                "slip_risk": slip_risk,
                "depth_uncertainty": depth_uncertainty,
                "force_min": force_min,
                "force_max": force_max,
            })

        return steps

    # ----------------------------------------------------------
    @staticmethod
    def _load_image(path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


# ============================================================
# 入口
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase2 RLDS 数据集构建脚本 — 将真机采集数据转换为 RLDS 格式"
    )
    parser.add_argument(
        "--source-root", type=str, required=True,
        help="数据源根目录，包含 {object_type}_sampleN/ 子目录的路径",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="RLDS 数据集输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)

    if not source_root.exists():
        print(f"[ERROR] 数据源路径不存在: {source_root}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"源数据: {source_root}")
    print(f"输出到: {output_dir}")

    builder = Phase2RobotData(data_dir=output_dir, source_root=source_root)
    builder.download_and_prepare()

    print(f"\nRLDS 数据集构建完成: {output_dir}")


if __name__ == "__main__":
    main()

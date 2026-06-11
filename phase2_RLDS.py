import tensorflow_datasets as tfds
import numpy as np
import json
import os
import sys
import cv2
from pathlib import Path
from typing import Iterator, Dict, Any, Tuple

# [核心修复] 强制该脚本仅使用 CPU，避免卡在 GPU 驱动注册环节
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 减少 TensorFlow 冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# ================= 数据集元数据 =================
_DESCRIPTION = """
RM75 真实数据转换脚本 - 已修复与仿真数据的兼容性问题。
1. 图像强制缩放为 224x224。
2. State 降维至 7 维 (6D Pose + 1D Gripper)。
3. 增加了处理进度打印，防止运行“假死”。
"""

class LocalRobotData(tfds.core.GeneratorBasedBuilder):
    """瑞尔曼机械臂真机数据集转换器 (标准兼容版)"""

    VERSION = tfds.core.Version('1.7.0')
    RELEASE_NOTES = {
        '1.7.0': '修正图像 Shape 为 (224,224,3)，State 降维至 7 维，增加运行进度反馈。',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """定义严格兼容 OpenVLA 的特征结构"""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'primary_image': tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, doc='DJI Action 主视角'
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, doc='RealSense 腕部视角'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,), dtype=np.float32, doc='6位姿 + 1夹爪'
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,), dtype=np.float32, doc='6维位姿增量 + 1维夹爪增量'
                    ),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Scalar(dtype=np.int64),
                    'source_file': tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # 注意：处理带有中文字符的路径
        target_abs_path = Path("/media/sysu/real_data")
        if not target_abs_path.exists():
            print(f"❌ [Error] 无法找到数据目录: {target_abs_path}")
            sys.exit(1)
        return {'train': self._generate_examples(target_abs_path)}

    def _generate_examples(self, root_path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
        session_dirs = sorted(list(root_path.glob("session_*")))
        total_sessions = len(session_dirs)
        print(f"🚀 [Builder] 准备转换 {total_sessions} 条轨迹...")
        
        for ep_idx, session_dir in enumerate(session_dirs):
            if not session_dir.is_dir(): continue
            
            # 实时进度反馈：防止用户认为脚本卡住
            print(f"  > [{ep_idx+1}/{total_sessions}] 正在处理: {session_dir.name} ...", end='\r')
            
            robot_state_dir = session_dir / "robot_state"
            dji_dir = session_dir / "dji"
            rs_dir = session_dir / "realsense_rgb"

            state_files = sorted(list(robot_state_dir.glob("*.json")))
            if not state_files: continue

            steps_raw = []
            for json_file in state_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    frame_id = data.get("frame", "")
                    d_img_path = dji_dir / f"{frame_id}.jpg"
                    r_img_path = rs_dir / f"{frame_id}.jpg"
                    
                    if not d_img_path.exists() or not r_img_path.exists(): continue
                        
                    inner = data.get("state", {}).get("data", {})
                    pose = np.array(inner.get("pose", []), dtype=np.float32)
                    
                    if len(pose) < 6: continue

                    # 图像处理：读取并强制缩放
                    def process_img(p):
                        img = cv2.imread(str(p))
                        if img is None: return None
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                    
                    p_img = process_img(d_img_path)
                    w_img = process_img(r_img_path)
                    
                    if p_img is None or w_img is None: continue
                        
                    steps_raw.append({
                        "pose": pose,
                        "primary_image": p_img,
                        "wrist_image": w_img
                    })
                except Exception as e:
                    # 打印具体错误，防止静默失败
                    print(f"\n⚠️ 解析帧 {json_file.name} 失败: {e}")
                    continue

            if not steps_raw: continue

            episode_steps = []
            num_steps = len(steps_raw)
            for i in range(num_steps):
                curr = steps_raw[i]
                # 降维至 7 维 [6D Pose + 1D Gripper]
                state_vec = np.concatenate([curr["pose"], [0.0]]).astype(np.float32)
                
                if i < num_steps - 1:
                    nxt = steps_raw[i+1]
                    delta_pose = nxt["pose"] - curr["pose"]
                    action_vec = np.concatenate([delta_pose, [0.0]]).astype(np.float32)
                else:
                    action_vec = np.zeros(7, dtype=np.float32)

                episode_steps.append({
                    'observation': {
                        'primary_image': curr["primary_image"],
                        'wrist_image': curr["wrist_image"],
                        'state': state_vec,
                    },
                    'action': action_vec,
                    'reward': 0.0,
                    'is_first': (i == 0),
                    'is_last': (i == num_steps - 1),
                    'is_terminal': (i == num_steps - 1),
                    'language_instruction': "Approach the transparent object, align the gripper center with the object's centroid, and hold still 5cm above it",
                })

            yield f"real_episode_{ep_idx}", {
                'steps': episode_steps,
                'episode_metadata': {
                    'episode_id': ep_idx,
                    'source_file': str(session_dir.name)
                }
            }
        print(f"\n✅ 数据迭代完成，正在写入 TFRecords...")

if __name__ == "__main__":
    current_dir = os.path.abspath(os.getcwd())
    # 实例化并指定输出路径
    builder = LocalRobotData(data_dir=current_dir)
    builder.download_and_prepare()
    print(f"\n✨ 转换任务结束！TFDS 数据集保存在: {os.path.join(current_dir, 'local_robot_data')}")
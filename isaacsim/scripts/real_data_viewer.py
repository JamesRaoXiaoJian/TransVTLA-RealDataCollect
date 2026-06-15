"""
真实数据查看器 - 交互式浏览真实环境采集的数据

数据结构:
    real_data/
        session_YYYYMMDD_HHMMSS/
            dji/               - DJI相机图像 (0001.jpg, 0002.jpg, ...)
            realsense_rgb/     - RealSense RGB相机图像 (0001.jpg, 0002.jpg, ...)
            robot_state/       - 机器人状态JSON (0001.json, 0002.json, ...)

使用方法:
    python real_data_viewer.py
    python real_data_viewer.py --session session_20251229_171149
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_ROOT = os.path.join(SCRIPT_DIR, "real_data")


def get_available_sessions():
    """获取所有可用的session"""
    if not os.path.exists(REAL_DATA_ROOT):
        return []
    
    sessions = [d for d in os.listdir(REAL_DATA_ROOT) 
                if os.path.isdir(os.path.join(REAL_DATA_ROOT, d)) 
                and d.startswith("session_")]
    return sorted(sessions)


def get_latest_session():
    """获取最新的session目录"""
    sessions = get_available_sessions()
    if not sessions:
        raise FileNotFoundError(f"未找到任何session数据: {REAL_DATA_ROOT}")
    return sessions[-1]


class RealDataViewer:
    """真实数据查看器 - 同时显示三个数据源"""
    
    def __init__(self, session_name=None):
        """
        Args:
            session_name: Session名称 (例如: "session_20251229_171149")
        """
        # 如果未指定session，使用最新的
        if session_name is None:
            session_name = get_latest_session()
            print(f"📁 使用最新的session: {session_name}")
        
        self.session_name = session_name
        self.session_dir = os.path.join(REAL_DATA_ROOT, session_name)
        
        if not os.path.exists(self.session_dir):
            raise FileNotFoundError(f"Session目录不存在: {self.session_dir}")
        
        # 数据目录
        self.dji_dir = os.path.join(self.session_dir, "dji")
        self.realsense_dir = os.path.join(self.session_dir, "realsense_rgb")
        self.robot_state_dir = os.path.join(self.session_dir, "robot_state")
        
        # 加载数据
        print(f"⏳ 加载session数据...")
        self.load_session_data()
        print(f"✅ 加载完成: {len(self.frames)} 帧")
        
        # 创建图形界面
        self.create_gui()
        
        # 初始显示
        self.current_idx = 0
        self.update(0)
        
        plt.show()
    
    def load_session_data(self):
        """加载session的所有数据索引"""
        # 获取所有帧编号（从robot_state获取，因为它是主要数据源）
        if not os.path.exists(self.robot_state_dir):
            raise FileNotFoundError(f"robot_state目录不存在: {self.robot_state_dir}")
        
        json_files = sorted([f for f in os.listdir(self.robot_state_dir) 
                            if f.endswith('.json')])
        
        if not json_files:
            raise ValueError(f"未找到任何JSON文件: {self.robot_state_dir}")
        
        self.frames = []
        for json_file in json_files:
            frame_num = json_file.replace('.json', '')
            
            frame_data = {
                'frame_num': frame_num,
                'robot_state_path': os.path.join(self.robot_state_dir, json_file),
                'dji_img_path': os.path.join(self.dji_dir, f"{frame_num}.jpg"),
                'realsense_img_path': os.path.join(self.realsense_dir, f"{frame_num}.jpg"),
            }
            
            # 检查文件是否存在
            frame_data['has_robot_state'] = os.path.exists(frame_data['robot_state_path'])
            frame_data['has_dji'] = os.path.exists(frame_data['dji_img_path'])
            frame_data['has_realsense'] = os.path.exists(frame_data['realsense_img_path'])
            
            self.frames.append(frame_data)
        
        # 统计
        total_frames = len(self.frames)
        has_dji = sum(1 for f in self.frames if f['has_dji'])
        has_realsense = sum(1 for f in self.frames if f['has_realsense'])
        has_robot = sum(1 for f in self.frames if f['has_robot_state'])
        
        print(f"   总帧数: {total_frames}")
        print(f"   DJI图像: {has_dji}/{total_frames}")
        print(f"   RealSense图像: {has_realsense}/{total_frames}")
        print(f"   机器人状态: {has_robot}/{total_frames}")
    
    def create_gui(self):
        """创建GUI界面"""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle(
            f'Real Data Viewer - {self.session_name}',
            fontsize=16, fontweight='bold'
        )
        
        # 布局 - 3行3列
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.4, wspace=0.3,
                              left=0.05, right=0.98, top=0.93, bottom=0.15)
        
        # 第一行: 两个相机图像
        self.ax_dji = self.fig.add_subplot(gs[0, 0:2])       # DJI相机
        self.ax_realsense = self.fig.add_subplot(gs[0, 2])   # RealSense相机
        
        # 第二行: 关节位置和姿态
        self.ax_joints = self.fig.add_subplot(gs[1, 0:2])    # 关节角度
        self.ax_pose = self.fig.add_subplot(gs[1, 2])        # 末端姿态
        
        # 第三行: 轨迹和状态信息
        self.ax_trajectory = self.fig.add_subplot(gs[2, 0:2]) # XYZ轨迹
        self.ax_info = self.fig.add_subplot(gs[2, 2])        # 详细信息
        
        # 控制面板 - 底部
        # Slider
        ax_slider = plt.axes([0.15, 0.065, 0.65, 0.025])
        self.slider = Slider(
            ax_slider, 'Frame', 
            0, len(self.frames)-1, 
            valinit=0, 
            valstep=1
        )
        self.slider.on_changed(self.update)
        
        # Frame number input box
        ax_frame_box = plt.axes([0.05, 0.065, 0.08, 0.025])
        self.frame_textbox = TextBox(ax_frame_box, 'Frame: ', initial='1')
        self.frame_textbox.on_submit(self.on_frame_submit)
        
        # Prev/Next按钮
        ax_prev = plt.axes([0.82, 0.065, 0.04, 0.025])
        ax_next = plt.axes([0.87, 0.065, 0.04, 0.025])
        ax_jump_10 = plt.axes([0.92, 0.065, 0.05, 0.025])
        
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_jump_10 = Button(ax_jump_10, '+10')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_jump_10.on_clicked(self.jump_10_frames)
        
        # 帧范围提示
        ax_range_label = plt.axes([0.05, 0.04, 0.08, 0.02])
        ax_range_label.axis('off')
        self.range_text = ax_range_label.text(
            0.5, 0.5, f'(1-{len(self.frames)})', 
            ha='center', va='center', fontsize=8, color='gray'
        )
    
    def on_frame_submit(self, text):
        """Frame number input submission"""
        try:
            frame_idx = int(text) - 1  # User input starts from 1, internal index from 0
            if 0 <= frame_idx < len(self.frames):
                self.slider.set_val(frame_idx)
            else:
                print(f"⚠️ Invalid frame: {text}. Range: 1-{len(self.frames)}")
                self.frame_textbox.set_val(str(self.current_idx + 1))
        except ValueError:
            print(f"⚠️ Invalid input, please enter a number")
            self.frame_textbox.set_val(str(self.current_idx + 1))
    
    def load_robot_state(self, path):
        """加载机器人状态JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 加载robot_state失败: {e}")
            return None
    
    def load_image(self, path):
        """加载图像"""
        try:
            if os.path.exists(path):
                return np.array(Image.open(path))
            return None
        except Exception as e:
            print(f"⚠️ 加载图像失败: {e}")
            return None
    
    def update(self, val):
        """更新显示"""
        idx = int(self.slider.val)
        self.current_idx = idx
        frame = self.frames[idx]
        
        # 加载数据
        robot_state = self.load_robot_state(frame['robot_state_path']) if frame['has_robot_state'] else None
        dji_img = self.load_image(frame['dji_img_path']) if frame['has_dji'] else None
        realsense_img = self.load_image(frame['realsense_img_path']) if frame['has_realsense'] else None
        
        # 更新帧号输入框
        self.frame_textbox.set_val(str(idx + 1))
        
        # 清空所有子图
        self.ax_dji.clear()
        self.ax_realsense.clear()
        self.ax_joints.clear()
        self.ax_pose.clear()
        self.ax_trajectory.clear()
        self.ax_info.clear()
        
        # 1. DJI相机图像
        if dji_img is not None:
            self.ax_dji.imshow(dji_img)
            self.ax_dji.set_title(f'DJI Camera - Frame {frame["frame_num"]}', fontsize=12, fontweight='bold')
        else:
            self.ax_dji.text(0.5, 0.5, 'No DJI Image', ha='center', va='center', fontsize=14)
            self.ax_dji.set_title('DJI Camera', fontsize=12)
        self.ax_dji.axis('off')
        
        # 2. RealSense相机图像
        if realsense_img is not None:
            self.ax_realsense.imshow(realsense_img)
            self.ax_realsense.set_title('RealSense RGB', fontsize=12, fontweight='bold')
        else:
            self.ax_realsense.text(0.5, 0.5, 'No RealSense Image', ha='center', va='center', fontsize=14)
            self.ax_realsense.set_title('RealSense RGB', fontsize=12)
        self.ax_realsense.axis('off')
        
        # 3. 关节角度
        if robot_state and robot_state.get('state', {}).get('code') == 0:
            joint_angles = robot_state['state']['data']['joint']
            joint_names = [f'J{i+1}' for i in range(len(joint_angles))]
            
            colors = ['tab:blue'] * (len(joint_angles) - 1) + ['tab:gray']  # 最后一个是gripper
            bars = self.ax_joints.bar(joint_names, joint_angles, color=colors, alpha=0.7, edgecolor='black')
            
            # 在每个柱子上显示数值
            for bar, angle in zip(bars, joint_angles):
                height = bar.get_height()
                self.ax_joints.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{angle:.1f}°', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9
                )
            
            self.ax_joints.set_ylabel('Angle (deg)', fontsize=11)
            self.ax_joints.set_title('Joint Angles', fontsize=12, fontweight='bold')
            self.ax_joints.grid(True, alpha=0.3, axis='y')
            self.ax_joints.axhline(0, color='k', linewidth=0.8)
        else:
            self.ax_joints.text(0.5, 0.5, 'No Robot State', ha='center', va='center', fontsize=14)
            self.ax_joints.set_title('Joint Angles', fontsize=12)
        
        # 4. 末端姿态 (6D pose)
        if robot_state and robot_state.get('state', {}).get('code') == 0:
            pose = robot_state['state']['data']['pose']
            pose_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
            pose_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:cyan']
            
            bars = self.ax_pose.barh(pose_labels, pose, color=pose_colors, alpha=0.7, edgecolor='black')
            
            # 在每个柱子上显示数值
            for bar, value in zip(bars, pose):
                width = bar.get_width()
                self.ax_pose.text(
                    width, bar.get_y() + bar.get_height()/2.,
                    f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center',
                    fontsize=9
                )
            
            self.ax_pose.set_xlabel('Value', fontsize=11)
            self.ax_pose.set_title('End-Effector Pose\n(Position: m, Rotation: rad)', fontsize=11, fontweight='bold')
            self.ax_pose.grid(True, alpha=0.3, axis='x')
            self.ax_pose.axvline(0, color='k', linewidth=0.8)
        else:
            self.ax_pose.text(0.5, 0.5, 'No Pose Data', ha='center', va='center', fontsize=14)
            self.ax_pose.set_title('End-Effector Pose', fontsize=11)
        
        # 5. 轨迹 (收集前面所有帧的pose数据)
        poses = []
        for i in range(min(idx + 1, len(self.frames))):
            f = self.frames[i]
            if f['has_robot_state']:
                rs = self.load_robot_state(f['robot_state_path'])
                if rs and rs.get('state', {}).get('code') == 0:
                    pose = rs['state']['data']['pose']
                    poses.append(pose[:3])  # 只取XYZ
        
        if poses:
            poses = np.array(poses)
            
            # Draw XY projection of 3D trajectory
            self.ax_trajectory.plot(poses[:, 0], poses[:, 1], 'b-', alpha=0.6, linewidth=2, label='Trajectory')
            self.ax_trajectory.plot(poses[-1, 0], poses[-1, 1], 'ro', markersize=10, label='Current')
            
            # Start point
            self.ax_trajectory.plot(poses[0, 0], poses[0, 1], 'go', markersize=8, label='Start')
            
            self.ax_trajectory.set_xlabel('X (m)', fontsize=11)
            self.ax_trajectory.set_ylabel('Y (m)', fontsize=11)
            self.ax_trajectory.set_title('End-Effector Trajectory (XY Plane)', fontsize=12, fontweight='bold')
            self.ax_trajectory.legend(fontsize=9)
            self.ax_trajectory.grid(True, alpha=0.3)
            self.ax_trajectory.axis('equal')
        else:
            self.ax_trajectory.text(0.5, 0.5, 'No Trajectory Data', ha='center', va='center', fontsize=14)
            self.ax_trajectory.set_title('End-Effector Trajectory', fontsize=12)
        
        # 6. 详细信息
        self.ax_info.axis('off')
        
        info_text = f"Session: {self.session_name}\n"
        info_text += f"{'='*35}\n\n"
        info_text += f"Frame: {idx+1}/{len(self.frames)}\n"
        info_text += f"Frame ID: {frame['frame_num']}\n\n"
        
        if robot_state:
            timestamp = robot_state.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                try:
                    dt = datetime.fromisoformat(timestamp)
                    info_text += f"Time: {dt.strftime('%H:%M:%S.%f')[:-3]}\n\n"
                except:
                    info_text += f"Time: {timestamp}\n\n"
            
            if robot_state.get('state', {}).get('code') == 0:
                data = robot_state['state']['data']
                
                # Joint information
                info_text += "Joint Angles (deg):\n"
                for i, angle in enumerate(data['joint'][:7]):  # First 7 are joints
                    info_text += f"  J{i+1}: {angle:7.2f}°\n"
                info_text += f"  Gripper: {data['joint'][6]:7.2f}°\n\n"
                
                # Pose information
                pose = data['pose']
                info_text += "End-Effector Pose:\n"
                info_text += f"  Position: ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) m\n"
                info_text += f"  Rotation: ({pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}) rad\n\n"
                
                # Error information
                if 'err' in data:
                    err_info = data['err']
                    if err_info['err_len'] > 0:
                        info_text += f"Error Code: {', '.join(err_info['err'])}\n"
                    else:
                        info_text += "Status: Normal\n"
            else:
                info_text += f"Robot State Code: {robot_state['state']['code']}\n"
        else:
            info_text += "Robot State: Unavailable\n"
        
        info_text += f"\nData Sources:\n"
        info_text += f"  DJI: {'✅' if frame['has_dji'] else '❌'}\n"
        info_text += f"  RealSense: {'✅' if frame['has_realsense'] else '❌'}\n"
        info_text += f"  Robot State: {'✅' if frame['has_robot_state'] else '❌'}\n"
        
        self.ax_info.text(
            0.05, 0.95, info_text, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2)
        )
        
        self.fig.canvas.draw_idle()
    
    def prev_frame(self, event):
        """Previous frame"""
        if self.current_idx > 0:
            self.slider.set_val(self.current_idx - 1)
    
    def next_frame(self, event):
        """Next frame"""
        if self.current_idx < len(self.frames) - 1:
            self.slider.set_val(self.current_idx + 1)
    
    def jump_10_frames(self, event):
        """Jump 10 frames forward"""
        new_idx = min(self.current_idx + 10, len(self.frames) - 1)
        self.slider.set_val(new_idx)


def main():
    parser = argparse.ArgumentParser(description='真实数据查看器')
    parser.add_argument('--session', type=str, default=None,
                       help='Session名称 (例如: session_20251229_171149)')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的sessions')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Real Data Viewer - 真实环境数据查看器")
    print("=" * 70)
    
    # 列出所有sessions
    if args.list:
        sessions = get_available_sessions()
        if sessions:
            print(f"\n找到 {len(sessions)} 个sessions:")
            for s in sessions:
                print(f"  - {s}")
        else:
            print(f"\n未找到任何session数据: {REAL_DATA_ROOT}")
        return
    
    print("\n使用说明:")
    print("  - 拖动滑块浏览帧")
    print("  - 点击 Prev/Next 按钮切换")
    print("  - 点击 +10 按钮快速跳过10帧")
    print("  - 在帧号输入框输入数字直接跳转")
    print("  - 关闭窗口退出\n")
    
    try:
        viewer = RealDataViewer(session_name=args.session)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print(f"\n可用的sessions:")
        sessions = get_available_sessions()
        for s in sessions:
            print(f"  - {s}")
        print(f"\n使用方法: python real_data_viewer.py --session <session_name>")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
高频位姿记录（30Hz）+ 键盘控制机械臂（1mm步进）
支持上下左右移动 pose 前三位（x, y, z），旋转不变
按键：↑↓ 控制Z轴，←→ 控制Y轴，QE 控制X轴，ESC 退出
"""

from Robotic_Arm.rm_robot_interface import *
import threading
import time
import keyboard
from datetime import datetime
from pathlib import Path
import json


class RobotController:
    def __init__(self, ip: str = "192.168.31.65", port: int = 8080):
        """初始化机械臂控制器"""
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(ip, port)
        print(f"机械臂ID：{self.handle.id}")
        
        # 获取初始位姿
        self.current_pose = self.robot.rm_get_current_arm_state()[1]['pose']
        print(f"初始位姿：{self.current_pose}")
        
        # 控制参数
        self.step_size = 0.001  # 1mm = 0.001m
        self.recording = True
        self.pose_log = []
        
        # 线程锁
        self.pose_lock = threading.Lock()
        self.movement_lock = threading.Lock()
        
    def record_pose_loop(self, frequency: float = 30.0):
        """高频记录位姿数据（默认30Hz）"""
        interval = 1.0 / frequency
        print(f"开始记录位姿数据，频率：{frequency} Hz")
        
        while self.recording:
            start_time = time.time()
            
            try:
                # 获取当前位姿
                state = self.robot.rm_get_current_arm_state()
                if state[0] == 0:  # 成功获取
                    pose = state[1]['pose']
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    with self.pose_lock:
                        self.pose_log.append({
                            'timestamp': timestamp,
                            'pose': pose.copy()
                        })
                    
                    # 每100条记录打印一次
                    if len(self.pose_log) % 100 == 0:
                        print(f"已记录 {len(self.pose_log)} 条位姿数据")
                        
            except Exception as e:
                print(f"记录位姿出错：{e}")
            
            # 精确控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)
    
    def move_robot(self, axis: int, direction: int):
        """
        移动机械臂
        axis: 0=x, 1=y, 2=z
        direction: 1=正方向, -1=负方向
        """
        with self.movement_lock:
            try:
                # 获取当前位姿
                state = self.robot.rm_get_current_arm_state()
                if state[0] == 0:
                    pose = state[1]['pose'].copy()
                    old_value = pose[axis]
                    
                    # 修改指定轴的位置（前三位）
                    pose[axis] += direction * self.step_size
                    
                    # 执行移动（保持旋转不变）
                    # v=50速度提高, r=0无交融, connect=0立即执行, block=1阻塞等待
                    result = self.robot.rm_movej_p(pose, 20, 0, 0, 1)
                    
                    if result == 0:
                        # 等待移动完成后再次读取位姿验证
                        time.sleep(0.05)
                        new_state = self.robot.rm_get_current_arm_state()
                        if new_state[0] == 0:
                            actual_pose = new_state[1]['pose']
                            axis_name = ['X', 'Y', 'Z'][axis]
                            dir_name = '+' if direction > 0 else '-'
                            actual_change = (actual_pose[axis] - old_value) * 1000
                            print(f"{axis_name}轴{dir_name}{self.step_size*1000:.1f}mm -> 目标: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] | 实际: [{actual_pose[0]:.4f}, {actual_pose[1]:.4f}, {actual_pose[2]:.4f}] | 变化: {actual_change:.2f}mm")
                            
                            with self.pose_lock:
                                self.current_pose = actual_pose
                    else:
                        print(f"移动失败，错误代码：{result}")
                        
            except Exception as e:
                print(f"移动机械臂出错：{e}")
    
    def keyboard_control_loop(self):
        """键盘控制循环"""
        print("\n键盘控制说明：")
        print("  W / ↑  : Z轴正方向 (+1mm)")
        print("  S / ↓  : Z轴负方向 (-1mm)")
        print("  A / ←  : Y轴正方向 (+1mm)")
        print("  D / →  : Y轴负方向 (-1mm)")
        print("  Q      : X轴正方向 (+1mm)")
        print("  E      : X轴负方向 (-1mm)")
        print("  ESC    : 退出程序")
        print("-" * 50)
        
        while self.recording:
            try:
                # 检测按键
                if keyboard.is_pressed('up') or keyboard.is_pressed('w'):
                    self.move_robot(2, 1)  # Z+
                    time.sleep(0.1)  # 防止连续触发
                    
                elif keyboard.is_pressed('down') or keyboard.is_pressed('s'):
                    self.move_robot(2, -1)  # Z-
                    time.sleep(0.1)
                    
                elif keyboard.is_pressed('left') or keyboard.is_pressed('a'):
                    self.move_robot(1, 1)  # Y+
                    time.sleep(0.1)
                    
                elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
                    self.move_robot(1, -1)  # Y-
                    time.sleep(0.1)
                    
                elif keyboard.is_pressed('q'):
                    self.move_robot(0, 1)  # X+
                    time.sleep(0.1)
                    
                elif keyboard.is_pressed('e'):
                    self.move_robot(0, -1)  # X-
                    time.sleep(0.1)
                    
                elif keyboard.is_pressed('esc'):
                    print("\n接收到退出指令...")
                    self.recording = False
                    break
                    
                time.sleep(0.01)  # 减少CPU占用
                
            except Exception as e:
                print(f"键盘控制出错：{e}")
                break
    
    def save_pose_log(self, output_dir: str = "pose_logs"):
        """保存位姿记录到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"pose_log_{timestamp}.json"
        
        with self.pose_lock:
            data = {
                'total_records': len(self.pose_log),
                'records': self.pose_log
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"\n位姿数据已保存：{filename}")
            print(f"总记录数：{len(self.pose_log)} 条")
    
    def run(self):
        """启动控制器"""
        # 启动位姿记录线程
        record_thread = threading.Thread(target=self.record_pose_loop, args=(30,), daemon=True)
        record_thread.start()
        
        # 启动键盘控制线程
        keyboard_thread = threading.Thread(target=self.keyboard_control_loop, daemon=True)
        keyboard_thread.start()
        
        # 等待线程结束
        try:
            keyboard_thread.join()
        except KeyboardInterrupt:
            print("\n程序被中断")
            self.recording = False
        
        # 等待记录线程结束
        time.sleep(0.5)
        
        # 保存数据
        self.save_pose_log()
        
        # 清理资源
        self.robot.rm_delete_robot_arm()
        print("机械臂连接已断开")


def main():
    """主函数"""
    try:
        controller = RobotController(ip="192.168.31.65", port=8080)
        controller.run()
    except Exception as e:
        print(f"程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
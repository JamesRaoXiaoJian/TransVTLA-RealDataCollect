from Robotic_Arm.rm_robot_interface import *

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.33.80", 8080)
print("机械臂ID：", handle.id)
state = robot.rm_get_current_arm_state()
print("机械臂状态：", state)

robot.rm_delete_robot_arm()
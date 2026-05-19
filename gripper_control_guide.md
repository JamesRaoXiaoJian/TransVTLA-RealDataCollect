# 睿尔曼机械臂夹爪控制参考文档

本文档总结本项目中睿尔曼机械臂连接知行 CTAG2F120 夹爪的控制方式、测试结论和常用代码。当前实验环境中，夹爪实际应通过末端 24V + Modbus RTU 寄存器控制，而不是通过睿尔曼内置 `GripperControl` 状态接口做闭环。

## 1. 设备与连接

| 项目 | 当前配置 |
|------|----------|
| 机械臂 IP | `172.25.5.243` |
| 机械臂 API 端口 | `8080` |
| 夹爪型号 | 知行 CTAG2F120 |
| 通讯方式 | 末端接口板 RS485，Modbus RTU 主站 |
| Modbus 端口 | `1` |
| 波特率 | `115200` |
| 从站地址 | `1` |
| 末端供电 | 24V，`voltage_type=3` |

## 2. 两套夹爪控制方式的区别

### 2.1 睿尔曼内置 GripperControl API

睿尔曼 SDK 提供以下接口：

```python
arm.rm_set_gripper_position(position, block, timeout)
arm.rm_set_gripper_release(speed, block, timeout)
arm.rm_set_gripper_pick(speed, force, block, timeout)
arm.rm_get_gripper_state()
```

这些接口适用于睿尔曼控制器内置夹爪协议。测试中，这些写入接口能返回 `0`，夹爪也可能产生动作，但 `rm_get_gripper_state()` 读到的数据长期为：

```json
{
  "enable_state": 1,
  "status": 0,
  "error": 0,
  "mode": 0,
  "current_force": 0,
  "temperature": 0,
  "actpos": 0
}
```

因此在当前知行 CTAG2F120 方案中，不建议依赖 `rm_get_gripper_state()` 判断夹爪位置、力或在线状态。

### 2.2 知行 CTAG2F120 Modbus 控制

操作手册和实测均表明，知行 CTAG2F120 通过末端 Modbus RTU 控制。核心流程是：

1. 连接机械臂。
2. 开启末端 24V 输出。
3. 设置末端 RS485 为 Modbus RTU 主站。
4. 写力矩寄存器。
5. 写目标位置寄存器。
6. 写运行寄存器触发动作。
7. 可选读回寄存器确认写入值。
8. 关闭 Modbus 模式和末端电源。

这是本项目推荐使用的夹爪控制方案。

## 3. 寄存器说明

| 功能 | 寄存器地址 | 写入方式 | 说明 |
|------|------------|----------|------|
| 目标位置 | `258` | 多寄存器写入，2 个寄存器 / 4 字节 | `0` 为打开，`1000` 为闭合 |
| 运行触发 | `264` | 单寄存器写入 | 写 `1` 后执行当前位置目标 |
| 力矩 | `284` | 单寄存器写入 | 常用 `50`，表示 50% |

位置值采用 4 字节大端格式写入。例如：

| 位置 | 十六进制 | 写入字节 |
|------|----------|----------|
| `0` | `0x00000000` | `[0, 0, 0, 0]` |
| `500` | `0x000001F4` | `[0, 0, 1, 244]` |
| `1000` | `0x000003E8` | `[0, 0, 3, 232]` |

## 4. 推荐测试脚本

项目中已提供测试脚本：

```bash
python zhixing_ctag2f120_modbus_test.py
```

默认执行一次打开和闭合：

```text
open position: 0
close position: 1000
moment: 50
cycles: 1
```

常用命令：

```bash
# 打开后闭合，并打印详细步骤
python zhixing_ctag2f120_modbus_test.py

# 只打开
python zhixing_ctag2f120_modbus_test.py --sequence open

# 只闭合
python zhixing_ctag2f120_modbus_test.py --sequence close

# 移动到指定位置
python zhixing_ctag2f120_modbus_test.py --sequence position --position 500

# 循环开合 5 次
python zhixing_ctag2f120_modbus_test.py --cycles 5

# 写入后读回寄存器，验证读写
python zhixing_ctag2f120_modbus_test.py --read-back
```

## 5. 读写验证结论

实测 `--read-back` 输出显示读写均成功：

```text
set gripper moment 50%
return_code: 0 (success)

read moment register
return_code: 0 (success)
data: 50
```

说明力矩寄存器 `284` 写入 `50` 后可正确读回。

```text
write open position 0
return_code: 0 (success)

read position registers after open
return_code: 0 (success)
data: [0, 0, 0, 0]
```

说明位置寄存器 `258` 写入打开位置 `0` 后可正确读回。

```text
write close position 1000
return_code: 0 (success)

read position registers after close
return_code: 0 (success)
data: [0, 0, 3, 232]
```

`[0, 0, 3, 232]` 即 `0x000003E8`，十进制为 `1000`，说明闭合位置写入和读回正确。

## 6. 最小控制代码

下面是一个最小的打开/闭合示例：

```python
import time
from Robotic_Arm.rm_robot_interface import *

ROBOT_IP = "172.25.5.243"
ROBOT_PORT = 8080

MODBUS_PORT = 1
BAUDRATE = 115200
TIMEOUT = 20
DEVICE_ADDR = 1

REG_POSITION = 258
REG_RUN = 264
REG_MOMENT = 284


def make_params(address, num=1):
    return rm_peripheral_read_write_params_t(
        port=MODBUS_PORT,
        address=address,
        device=DEVICE_ADDR,
        num=num,
    )


def position_to_bytes(position):
    return [
        (position >> 24) & 0xFF,
        (position >> 16) & 0xFF,
        (position >> 8) & 0xFF,
        position & 0xFF,
    ]


def set_position(arm, position):
    data = position_to_bytes(position)
    code = arm.rm_write_registers(make_params(REG_POSITION, 2), data)
    print("write position:", code)
    if code != 0:
        return code

    code = arm.rm_write_single_register(make_params(REG_RUN, 1), 1)
    print("trigger run:", code)
    return code


arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm(ROBOT_IP, ROBOT_PORT)
print("handle id:", handle.id)

try:
    print("set 24V:", arm.rm_set_tool_voltage(3))
    print("set modbus:", arm.rm_set_modbus_mode(MODBUS_PORT, BAUDRATE, TIMEOUT))
    print("set moment:", arm.rm_write_single_register(make_params(REG_MOMENT, 1), 50))

    set_position(arm, 0)
    time.sleep(2)
    set_position(arm, 1000)
    time.sleep(2)
finally:
    print("close modbus:", arm.rm_close_modbus_mode(MODBUS_PORT))
    print("turn off 24V:", arm.rm_set_tool_voltage(0))
    print("delete arm:", arm.rm_delete_robot_arm())
    print("destroy:", arm.rm_destroy())
```

## 7. 返回码判断

常见返回码：

| 返回码 | 含义 |
|--------|------|
| `0` | 成功 |
| `1` | 控制器返回失败，参数错误或机械臂状态错误 |
| `-1` | 数据发送失败，通信异常 |
| `-2` | 数据接收失败或超时 |
| `-3` | 返回值解析失败 |
| `-4` | 当前控制器不支持该接口 |

实验中判断一次动作是否成功，至少要检查：

```python
write_position_code == 0
trigger_run_code == 0
```

如果需要确认寄存器值是否写入成功，再读回：

```python
code, data = arm.rm_read_multiple_holding_registers(
    rm_peripheral_read_write_params_t(
        port=1,
        address=258,
        device=1,
        num=2,
    )
)
```

## 8. 常见问题

### 8.1 为什么 `rm_get_gripper_state()` 读不到正确位置？

`rm_get_gripper_state()` 属于睿尔曼内置夹爪协议状态接口，而知行 CTAG2F120 当前是通过末端 Modbus RTU 寄存器控制。两者不是同一路协议，所以 `rm_get_gripper_state()` 返回成功并不代表能读到知行夹爪的真实位置和力。

当前应使用 Modbus 寄存器读回确认写入值。

### 8.2 为什么必须先开 24V？

夹爪由机械臂末端供电。未开启末端 24V 时，Modbus 命令可能能发出，但夹爪不会正常响应或动作。

对应接口：

```python
arm.rm_set_tool_voltage(3)
```

关闭时：

```python
arm.rm_set_tool_voltage(0)
```

### 8.3 为什么要关闭 Modbus 模式？

末端 RS485 进入 Modbus RTU 模式后，会占用该通讯口。脚本结束时关闭 Modbus 模式可以恢复端口状态，避免影响后续机械臂或其他末端设备操作。

```python
arm.rm_close_modbus_mode(1)
```

### 8.4 读回位置是否等于实际当前位置？

当前读回的是位置目标寄存器 `258` 的内容，能证明“写入值已进入夹爪寄存器”。它不一定等价于实时物理位置反馈。

如果需要判断是否夹住物体，建议使用：

- 外部压力/触觉传感器；
- 相机视觉判断；
- 知行夹爪手册中额外的状态/电流/错误寄存器，如果后续能拿到完整寄存器表。

## 9. 推荐集成方式

在采集程序中建议封装三个动作：

```python
open_gripper()     # 写 position=0，再写 run=1
close_gripper()    # 写 position=1000，再写 run=1
move_gripper(pos)  # 写 position=pos，再写 run=1
```

每次动作至少记录：

| 字段 | 说明 |
|------|------|
| `timestamp_us` | 动作时间戳 |
| `target_position` | 目标位置，0~1000 |
| `moment` | 力矩百分比 |
| `write_position_code` | 写位置返回码 |
| `trigger_run_code` | 运行触发返回码 |
| `readback_code` | 可选，读回返回码 |
| `readback_data` | 可选，读回寄存器数据 |

这样即使没有实时位置反馈，也能完整记录每次夹爪命令是否成功下发和寄存器是否写入。

from omni.isaac.kit import SimulationApp

# 1. 启动 Isaac Sim
simulation_app = SimulationApp({"headless": False})

print(">>> step1: script started")

from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core import World
import omni.usd

USD_PATH = r"C:/isaac-sim/USDFiles/scene.usd"
print(">>> loading usd:", USD_PATH)

# 2. 打开 USD
open_stage(USD_PATH)

# 3. 创建 world
world = World()
world.reset()

# 4. 打印 stage 里的 prim（关键！）
stage = omni.usd.get_context().get_stage()
print(">>> root prims:")
for prim in stage.GetPseudoRoot().GetChildren():
    print("   ", prim.GetPath())

print(">>> entering simulation loop")

# 5. 让仿真跑起来
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()

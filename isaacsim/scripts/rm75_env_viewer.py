# RM75 Environment Viewer — Original env.usd Scene (no Franka)
#
# Loads env.usd and runs the simulation with RM75 visible.
# All original scene elements (table, cameras, cube, RM75) are preserved.
#
# Usage:
#   ./python.sh scripts/rm75_env_viewer.py

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
from pxr import Usd, UsdGeom

from isaacsim.core.utils.stage import add_reference_to_stage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "USDFiles")
ENV_USD_PATH = os.path.join(USD_DIR, "env.usd")
ENV_ROOT = "/World/env"

print("Loading env.usd (original RM75 environment)...", flush=True)
add_reference_to_stage(ENV_USD_PATH, ENV_ROOT)

stage = simulation_app.context.get_stage()

# Verify key prims
for path, label in [
    (f"{ENV_ROOT}/env/table_instanceable/RM75_B_V", "RM75"),
    (f"{ENV_ROOT}/env/table_instanceable/Cube", "Cube"),
    (f"{ENV_ROOT}/env/World_Camera", "World Camera"),
]:
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim) if prim.IsValid() else None
    pos = None
    if xf:
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = np.array([m.ExtractTranslation()[i] for i in range(3)])
    print(f"  {label}: {path} -> {'valid' if prim.IsValid() else 'MISSING'}", end="")
    if pos is not None:
        print(f", pos={pos}", end="")
    print(flush=True)

import omni.timeline
omni.timeline.get_timeline_interface().play()

print("Scene running. Close window to exit.", flush=True)
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()

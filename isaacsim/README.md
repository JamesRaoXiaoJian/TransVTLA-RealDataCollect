# Isaac Sim Collection Scripts

This directory vendors the Isaac Sim collection scripts copied from:

```text
/home/james/isaacsim/scripts
```

Run scripts through the local wrapper, which forwards to the real Isaac Sim install:

```bash
cd isaacsim
./python.sh scripts/franka_data_collecter.py --headless
```

From the repository root, this also works:

```bash
isaacsim/python.sh isaacsim/scripts/franka_data_collecter.py --headless
isaacsim/python.sh scripts/franka_data_collecter.py --headless
```

The Franka collection scripts default to writing data under:

```text
isaacsim/scripts/runs/collected_data
```

For multi-process Franka collection:

```bash
cd isaacsim
scripts/start_franka_multigpu_screen.sh --dry-run
```

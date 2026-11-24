# Microduck

A ~34 cm, 15-DOF bipedal duck robot built on **Feetech STS3215** servos, trained in simulation with reinforcement learning.

## Roadmap

1. **CAD**: adapt the structure to the STS3215 servo → `cad/`
2. **Robot description**: URDF / MJCF with correct masses and joint frames for simulation
3. **Simulation + RL**: train walking and other skills (MuJoCo / Isaac Lab)
4. **Hardware**: print, assemble, deploy policies on the robot

## Repo layout

```
cad/
  baseline_hd1910/     starting point: printable parts for the HD1910 servo build (STL, 3MF)
  original_xl330/      original XL330 geometry, assembled (STL), for comparison
  servos/              servo models (STEP)
  sts3215/             STS3215 skeleton: part generator, STLs, collision check, MJCF generator
  README.md            part catalogue, joint chain, what has to change for the STS3215
sim/
  sts3215/             MuJoCo model of the STS3215 robot (generated), smoke test
rl/                    Isaac Lab training tasks for the STS3215 robot (see rl/README.md)
pyproject.toml         Python env (uv): mujoco, trimesh, onshape-to-robot
```

## Setup

```bash
uv sync
```

## License

CAD files in `cad/` are CC BY-NC-SA 4.0. See [`cad/README.md`](cad/README.md#license).

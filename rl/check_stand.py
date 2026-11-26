"""Sanity check of the training model: hold the standing pose (zero actions) and report height, tilt and torque.

    ~/IsaacLab/.venv/bin/python rl/check_stand.py
"""

import gymnasium as gym
import torch

import microduck_sts3215  # noqa: F401  (registers the task)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Microduck-STS3215-Velocity-Flat"

cfg = parse_env_cfg(TASK, num_envs=4)
cfg.events.push_robot = None
cfg.commands.base_velocity.rel_standing_envs = 1.0
env = gym.make(TASK, cfg=cfg)
env.reset()
robot = env.unwrapped.scene["robot"]
stall = robot.cfg.actuators["servos"].actuator_effort_limit
servos = [i for i, n in enumerate(robot.joint_names) if not n.startswith("passive_")]
peak = torch.zeros(len(servos), device=env.unwrapped.device)
steps = int(3.0 / env.unwrapped.step_dt)
for i in range(steps):
    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    tau = robot.actuators.applied_effort.torch[:, servos]
    peak = torch.maximum(peak, tau.abs().max(0).values)
    if i in (0, steps // 2, steps - 1):
        pos = robot.data.root_pos_w.torch if hasattr(robot.data.root_pos_w, "torch") else robot.data.root_pos_w
        g = robot.data.projected_gravity_b
        g = g.torch if hasattr(g, "torch") else g
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
        print(f"t={i * env.unwrapped.step_dt:4.2f}s trunk z {pos[:, 2].mean() * 1000:6.1f} mm, tilt {tilt.max():5.1f} deg")
names = [robot.joint_names[i] for i in servos]
top = torch.argsort(peak, descending=True)[:4]
print("mass", float(robot.data.default_mass.torch.sum(-1)[0]) if hasattr(robot.data.default_mass, "torch")
      else float(robot.data.default_mass.sum(-1)[0]), "kg")
print("peak torque % of stall:", {names[k]: round(100 * float(peak[k]) / stall, 1) for k in top})
env.close()

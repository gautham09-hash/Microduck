"""Terms the STS3215 task adds to Isaac Lab's MicroDuck terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction, BiasedJointPositionActionCfg
from isaaclab_tasks.contrib.microduck.mdp.observations import zero_non_finite
from isaaclab_tasks.contrib.microduck.mdp.rewards import _command_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class RateLimitedJointPositionAction(BiasedJointPositionAction):
    """Joint position action as the STS3215 firmware applies it.

    The firmware does not track a new goal position directly: it moves an internal target towards it at no more
    than ``max_velocity`` and runs its position controller on that target (Rhoban BAM's STS3215 model). The internal
    target starts from the measured position after a reset, as when the servo powers on.

    With Newton's actuators the physics runs all substeps of a policy step in one call, so this term runs once per
    policy step: the internal target then advances by the whole step's allowance at once.
    """

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._last_sim_step = 0
        self._fw_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._seed = torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._seed.fill_(True)
        else:
            self._seed[env_ids] = True

    def apply_actions(self):
        # in place throughout: the step may run as a captured CUDA graph
        goal = self.processed_actions - self.encoder_bias
        q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._fw_target.copy_(torch.where(self._seed, q, self._fw_target))
        self._seed.fill_(False)
        # allowance for the physics steps since the last call (1 or the decimation, depending on who owns it)
        steps = max(self._env._sim_step_counter - self._last_sim_step, 1)
        self._last_sim_step = self._env._sim_step_counter
        max_step = self.cfg.max_velocity * self._env.physics_dt * steps
        self._fw_target.copy_(torch.clamp(goal, self._fw_target - max_step, self._fw_target + max_step))
        self._asset.set_joint_position_target_index(target=self._fw_target, joint_ids=self._joint_ids)


@configclass
class RateLimitedJointPositionActionCfg(BiasedJointPositionActionCfg):
    class_type: type[RateLimitedJointPositionAction] = RateLimitedJointPositionAction
    max_velocity: float = 5.1
    """Fastest the firmware moves its internal target [rad/s]."""


def trunk_height_exp(
    env: ManagerBasedRLEnv, target: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping the trunk frame near ``target`` [m] above flat ground, with tolerance ``std`` [m]."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return zero_non_finite(torch.exp(-torch.square((height - target) / std)))


def feet_down_when_standing(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.01
) -> torch.Tensor:
    """1 while the command is to stand and both feet are on the ground, else 0."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    down = (sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).all(dim=1)
    standing = _command_magnitude(env, command_name) < command_threshold
    return (down & standing).float()


def touching_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 2.0) -> torch.Tensor:
    """True when a selected body of a filtered contact sensor is pushed by something other than its filter bodies.

    The sensor's net force counts every contact and its force matrix only those with the filter bodies (the rest
    of the robot), so what remains is the floor.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    robot = sensor.data.force_matrix_w.torch[:, sensor_cfg.body_ids].sum(dim=2)
    return (zero_non_finite(net - robot).norm(dim=-1) > threshold).any(dim=1)

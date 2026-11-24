"""Flat velocity walking for the STS3215 Microduck.

Starts from Isaac Lab's MicroDuck flat walking task (same observations, actions, rewards and curriculum; the
robot model uses the same body and joint names) and changes what differs for this robot: the model and servos,
slower commands, foot heights and step timing scaled to the bigger robot, and randomization of what isn't
measured yet (servo friction and gains, trunk and head mass).
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg

from .robot import SOLE_TO_ANKLE, STS3215_CFG

SWING_HEIGHT = 0.026
"""Foot lift during swing [m] (the original 20 mm, scaled 1.3x)."""

AIR_TIME = (0.18, 0.45)
"""Rewarded swing duration [s]: the original 0.125-0.3 s, slowed for the STS3215's lower speed."""


@configclass
class Sts3215VelocityFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """STS3215 Microduck flat velocity walking with Newton MJWarp and DC-motor servos."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = STS3215_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.use_newton_actuators = False  # the DC motors are computed by Isaac Lab, not Newton

        # slower walking: the servo's no-load speed is about half the original servo's
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (-0.15, 0.25)
        ranges.lin_vel_y = (-0.1, 0.1)
        ranges.ang_vel_z = (-0.6, 0.6)

        target = SWING_HEIGHT + SOLE_TO_ANKLE
        self.rewards.foot_clearance.params["target_height"] = target
        self.rewards.foot_swing_height.params["target_height"] = target
        self.rewards.foot_swing_height.weight = -0.25 * (target / 0.02) ** 2
        self.rewards.air_time.params["threshold_min"], self.rewards.air_time.params["threshold_max"] = AIR_TIME

        # randomize what isn't measured (replaces the original servo model's friction randomization)
        all_joints = SceneEntityCfg("robot", joint_names=[".*"])
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={"asset_cfg": all_joints, "friction_distribution_params": (0.5, 1.5), "operation": "scale"},
        )
        self.events.randomize_servo_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": all_joints,
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.7, 1.3),
                "operation": "scale",
            },
        )
        # trunk and head contents are estimates (battery, electronics, shells)
        self.events.mass_inertia.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.head_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"]),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

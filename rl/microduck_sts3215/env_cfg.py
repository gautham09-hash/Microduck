"""Flat velocity walking for the STS3215 Microduck.

Starts from Isaac Lab's MicroDuck flat walking task with its gearbox-play variant (same observations, actions,
rewards and curriculum; the robot model uses the same body and joint names) and changes what differs for this robot:

- the model and servos (identified STS3215, 10-40 ms command delay, firmware target rate limit, gearbox play);
- slower commands, foot heights and step timing scaled to the bigger robot;
- rewards for what the first run got wrong: torque, joint acceleration and landing-force penalties against
  shaking and overheating servos, trunk height and a pose reward while walking against crouching, both feet down
  when told to stand, a contact penalty for the thighs against the trunk and hip yaw servos, and an episode end
  when a thigh touches the floor;
- randomization of what isn't measured yet (servo friction, trunk and head mass).
"""

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    _FOOT_SENSOR_CFG,
    MicroDuckVelocityFlatEnvCfg,
    _schedule,
)
from isaaclab_tasks.utils.hydra import PresetCfg

from . import mdp as sts_mdp
from .robot import BAM, SOLE_TO_ANKLE, STAND_HEIGHT, STS3215_CFG

SWING_HEIGHT = 0.026
"""Foot lift during swing [m] (the original 20 mm, scaled 1.3x)."""

AIR_TIME = (0.18, 0.45)
"""Rewarded swing duration [s]: the original 0.125-0.3 s, slowed for the STS3215's lower speed."""

TRUNK_HEIGHT = STAND_HEIGHT - 0.008
"""Rewarded trunk frame height [m]: the stand pose, less a little knee bend for walking."""

LANDING_FORCE = 25.0
"""Foot force above which landing is penalized [N]: about 1.7 times the robot's weight."""

_SERVOS = SceneEntityCfg("robot", joint_names=["^(?!passive_).*"])
_GEOMETRY = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base"


def _choose_backlash(cfg) -> None:
    """Replace each preset in ``cfg``'s fields with its ``backlash`` option (or ``default`` if it has none)."""
    for name, value in list(vars(cfg).items()):
        if isinstance(value, PresetCfg):
            setattr(cfg, name, getattr(value, "backlash", None) or value.default)


@configclass
class Sts3215VelocityFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """STS3215 Microduck flat velocity walking with Newton MJWarp, BAM servos and gearbox play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = STS3215_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for group in (self.observations.policy, self.observations.critic, self.rewards):
            _choose_backlash(group)

        # the firmware rate-limits its target, so the policy's commands reach the motor no faster than this
        self.actions.joint_pos = sts_mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.actions.joint_pos.joint_names,
            preserve_order=True,
            scale=1.0,
            use_default_offset=True,
            max_velocity=BAM["max_velocity"],
        )

        # thighs against the trunk (with the hip yaw servos), and foot against foot
        self.scene.self_collision = ContactSensorCfg(
            prim_path=f"{_GEOMETRY}/.*/(upper_leg_left|upper_leg_right|ankle_left)",
            filter_prim_paths_expr=[_GEOMETRY, f"{_GEOMETRY}/.*/ankle_right"],
            update_period=self.sim.dt,
        )

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

        # effort: weights sized so the first run's gait costs about 0.2-0.7 per second each (tracking pays ~6.5)
        self.rewards.joint_torques = RewTerm(func=base_mdp.joint_torques_l2, weight=-0.05, params={"asset_cfg": _SERVOS})
        self.rewards.joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-1e-5, params={"asset_cfg": _SERVOS})
        self.rewards.landing_force = RewTerm(
            func=base_mdp.contact_forces, weight=-0.1, params={"threshold": LANDING_FORCE, "sensor_cfg": _FOOT_SENSOR_CFG}
        )
        self.curriculum.action_rate_weight = _schedule(
            "rewards.action_rate_l2.weight", (500, -0.3), (1000, -0.6), (2000, -1.0), (3000, -1.5), (4000, -2.0)
        )

        # posture: stand tall, stay near the stand pose while walking too, and keep both feet down when standing
        self.rewards.trunk_height = RewTerm(
            func=sts_mdp.trunk_height_exp, weight=1.0, params={"target": TRUNK_HEIGHT, "std": 0.025}
        )
        pose = self.rewards.pose.params
        pose["standing_only"] = False
        pose["std_standing"] = {k: 1.5 * v for k, v in pose["std_standing"].items()}
        self.rewards.feet_down_standing = RewTerm(
            func=sts_mdp.feet_down_when_standing,
            weight=1.0,
            params={"command_name": "base_velocity", "sensor_cfg": _FOOT_SENSOR_CFG},
        )

        # the thighs can rest on the floor, and kneeling on them with the feet in the air would collect the
        # step-timing reward
        self.terminations.knelt = DoneTerm(
            func=sts_mdp.touching_ground,
            params={"sensor_cfg": SceneEntityCfg("self_collision", body_names=["upper_leg_.*"])},
        )

        # randomize what isn't measured: one servo was identified, and the trunk and head contents are estimates
        self.events.randomize_joint_friction.params["scale_range"] = (0.8, 1.2)
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

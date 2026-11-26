"""Build the MuJoCo model of the STS3215 Microduck from layout.json and the exported part STLs.

Writes sim/sts3215/robot.xml + scene.xml (all parts collide with the floor, MuJoCo position actuators: for the
viewer and smoke_test.py) and robot_walk.xml + scene_walk.xml (for training, where the trainer models the servos:
no actuators, no joint damping or friction, the identified servo armature, a passive gearbox-play hinge behind each
servo, the soles colliding with the floor, and the thighs colliding with the trunk and hip yaw servos). Each body's mass, centre of mass and inertia come from
its printed parts (mesh volume x PRINT_DENSITY), the servos it holds and the payload estimates in layout.json.

Bodies use the original MicroDuck names (trunk_base, ankle_left, jaw_soft, ...) so training code written for that
robot applies unchanged; MJCF_NAME maps them to the link names in layout.json.

Servo model: position actuator with the force limited to the stall torque, and joint damping = stall torque / no-load
speed, which gives the motor's torque-speed line. Armature and friction are starting values to be identified on a
real servo.

Run: uv run python cad/sts3215/build_mjcf.py [--servo 19kg|30kg]
"""

import argparse
import json
import pathlib

import mujoco
import numpy as np
import trimesh

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT = ROOT / "sim" / "sts3215"
LAYOUT = json.loads((HERE / "layout.json").read_text())
MANIFEST = json.loads((HERE / "stl" / "manifest.json").read_text())

PRINT_DENSITY = 0.75e-6   # kg/mm^3: PLA 1.24 g/cm^3 at ~60 % effective fill
ARMATURE = 0.02           # kg m^2, reflected rotor inertia (estimate)
FRICTION = 0.05           # N m (estimate)
KP = 18.0                 # N m / rad
SOLE_T = 3.0              # mm, sole pad under the foot plate
# walk model: thighs (and the servos they hold) against the trunk and hip yaw servos, which the policy can hit
SELF_COLLIDE = {"left_thigh": "thigh", "right_thigh": "thigh", "trunk": "trunk"}
SELF_COL = {"thigh": dict(contype=4, conaffinity=8), "trunk": dict(contype=8, conaffinity=4)}
# default standing pose (rad), left leg / neck / head / right leg, same order as the joints below
STAND = dict(left_hip_yaw=0.0, left_hip_roll=-0.0873, left_hip_pitch=-0.4579, left_knee=-0.0049, left_ankle=0.4530,
             neck_pitch=0.3491, head_pitch=0.3491, head_yaw=0.0, head_roll=0.0, right_hip_yaw=0.0,
             right_hip_roll=0.0873, right_hip_pitch=0.4579, right_knee=0.0049, right_ankle=-0.4530)
JOINTS = list(STAND)
TRUNK_ORIGIN = np.array([0.0, 0.0, 165.0])   # mm: trunk body frame (and IMU) at the zero pose
MJCF_NAME = dict(trunk="trunk_base", left_hip_yaw_link="yaw2roll", left_hip_cross="hip_l", left_thigh="upper_leg_left",
                 left_shin="leg", left_foot="ankle_left", right_hip_yaw_link="bearing_roll", right_hip_cross="hip_l_2",
                 right_thigh="upper_leg_right", right_shin="leg_2", right_foot="ankle_right", neck="neck",
                 neck_top="neck_pitch", head_yaw_link="yaw_roll_motion", head="jaw_soft")


def mesh_mass(mesh, mass=None, density=None):
    """(mass kg, com mm, inertia about com kg mm^2) of a closed mesh in mm."""
    m = mesh.copy()
    m.density = 1.0
    vol = m.volume
    mass = mass if mass is not None else density * vol
    return mass, m.center_mass, m.moment_inertia * mass / vol


def box_mass(mass, com, size):
    a, b, c = size
    return mass, np.asarray(com, float), mass / 12 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


def combine(items):
    M = sum(i[0] for i in items)
    com = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, c, Ic in items:
        r = c - com
        I += Ic + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return M, com, I


def anchor(links, name):
    return TRUNK_ORIGIN if links[name]["parent"] is None else np.array(links[name]["anchor"], float)


def build(sv, servo_name, walk):
    """MjSpec of the robot. walk=True: the training model (see the module docstring)."""
    damping = 0.0 if walk else sv["stall_nm"] / sv["no_load_rad_s"]
    bam = LAYOUT["servo"]["bam"]
    play = np.radians(LAYOUT["servo"]["backlash_deg"])
    servo_mass = LAYOUT["servo"]["mass_g"] / 1000
    links = LAYOUT["links"]
    stl = HERE / "stl"
    servo_mesh = trimesh.load(stl / "sts3215.stl")
    col = dict(contype=0, conaffinity=0) if walk else dict(contype=2, conaffinity=1)

    def geom_col(link, servo):
        """Collision bits of a part or servo geom: in the walk model only the SELF_COLLIDE pairs collide."""
        group = SELF_COLLIDE.get(link)
        if not walk or group is None or (group == "trunk" and servo is not None and not servo.endswith("hip_yaw")):
            return col
        return SELF_COL[group]

    spec = mujoco.MjSpec()
    spec.modelname = f"microduck_sts3215_{servo_name}" + ("_walk" if walk else "")
    spec.compiler.degree = False
    spec.meshdir = str(stl)
    spec.option.timestep = 0.002
    spec.add_mesh(name="sts3215", file="sts3215.stl", scale=[1e-3] * 3)
    for p in MANIFEST:
        spec.add_mesh(name=p, file=p + ".stl", scale=[1e-3] * 3)

    bodies, report = {}, []
    order = ["trunk"] + [k for k in links if k != "trunk"]
    for name in order:
        e = links[name]
        a = anchor(links, name)
        if e["parent"] is None:
            b = spec.worldbody.add_body(name=MJCF_NAME[name], pos=a / 1000)
            b.add_freejoint(name="floating_base")
            b.add_site(name="imu", pos=[0, 0, 0])
        else:
            pa = anchor(links, e["parent"])
            lo, hi = np.radians(e["range_deg"])
            hinge = dict(type=mujoco.mjtJoint.mjJNT_HINGE, axis=e["axis"])
            b = bodies[e["parent"]].add_body(name=MJCF_NAME[name], pos=(a - pa) / 1000)
            if walk:
                # servo joint, then the gearbox play hinge in series on the same axis (the USD converter splits
                # them with a dummy body, as in Isaac Lab's MicroDuck backlash model); the trainer's servo model
                # adds friction and damping
                b.add_joint(name=e["joint"], range=[lo, hi], armature=bam["armature"], **hinge)
                b.add_joint(name=f"passive_{e['joint']}_backlash", range=[-play, play], damping=0.01, **hinge)
            else:
                b.add_joint(name=e["joint"], range=[lo, hi], damping=damping, armature=ARMATURE,
                            frictionloss=FRICTION, **hinge)
        bodies[name] = b
        items = []
        for part, link in MANIFEST.items():
            if link != name:
                continue
            mesh = trimesh.load(stl / (part + ".stl"))
            items.append(mesh_mass(mesh, density=PRINT_DENSITY))
            b.add_geom(name=part, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part, pos=-a / 1000, group=1,
                       rgba=[0.85, 0.55, 0.2, 1], mass=0, **geom_col(name, None))
        for joint, s in LAYOUT["servos"].items():
            if s["link"] != name:
                continue
            T = np.array(s["transform_mm"])
            m = servo_mesh.copy()
            m.apply_transform(T)
            items.append(mesh_mass(m, mass=servo_mass))
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
            b.add_geom(name="servo_" + joint, type=mujoco.mjtGeom.mjGEOM_MESH, meshname="sts3215",
                       pos=(T[:3, 3] - a) / 1000, quat=q, group=1, rgba=[0.15, 0.15, 0.15, 1], mass=0,
                       **geom_col(name, joint))
        if "payload" in e:
            pl = e["payload"]
            items.append(box_mass(pl["mass_g"] / 1000, pl["com_mm"], pl["size_mm"]))
        if name.endswith("_foot"):
            # sole pad: the contact surface
            plate = [g for g in MANIFEST if MANIFEST[g] == name][0]
            lo, hi = trimesh.load(stl / (plate + ".stl")).bounds
            c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - SOLE_T / 2])
            b.add_geom(name=name + "_sole", type=mujoco.mjtGeom.mjGEOM_BOX, pos=(c - a) / 1000,
                       size=[(hi[0] - lo[0]) / 2000, (hi[1] - lo[1]) / 2000, SOLE_T / 2000],
                       contype=2, conaffinity=1, friction=[1.0, 0.02, 0.01], rgba=[0.2, 0.2, 0.2, 1], mass=0)
            b.add_site(name=name + "_sole", pos=(c - a) / 1000)
        M, com, I = combine(items)
        b.explicitinertial = True
        b.mass = M
        b.ipos = (com - a) / 1000
        b.fullinertia = [I[0, 0] * 1e-6, I[1, 1] * 1e-6, I[2, 2] * 1e-6, I[0, 1] * 1e-6, I[0, 2] * 1e-6,
                         I[1, 2] * 1e-6]
        report.append((name, M))

    if not walk:
        for j in JOINTS:
            act = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=KP)
            act.forcelimited = True
            act.forcerange = [-sv["stall_nm"], sv["stall_nm"]]
        for typ, n in ((mujoco.mjtSensor.mjSENS_GYRO, "imu_gyro"),
                       (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "imu_acc"),
                       (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "imu_quat")):
            spec.add_sensor(name=n, type=typ, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    # keyframes: base height so the soles touch the floor
    model = spec.compile()
    data = mujoco.MjData(model)
    soles = [model.geom(f"{s}_foot_sole").id for s in ("left", "right")]
    heights = {}
    for kname, pose in (("zero", {}), ("stand", STAND)):
        data.qpos[:] = model.qpos0
        for j, v in pose.items():
            data.qpos[model.joint(j).qposadr[0]] = v
        mujoco.mj_kinematics(model, data)
        low = min(data.geom_xpos[g][2] - model.geom_size[g][2] for g in soles)
        q = data.qpos.copy()
        q[2] -= low
        heights[kname] = q[2]
        if walk:
            spec.add_key(name=kname, qpos=q)
        else:
            spec.add_key(name=kname, qpos=q, ctrl=np.array([pose.get(j, 0.0) for j in JOINTS]))
    return spec, report, damping, heights


SCENE = """<mujoco model="microduck_sts3215 scene">
  <include file="{robot}"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="135" elevation="-20" offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.25 0.3 0.35" rgb2="0.2 0.25 0.3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1"/>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", default="19kg", choices=list(LAYOUT["servo"]["variants"]))
    args = ap.parse_args()
    sv = LAYOUT["servo"]["variants"][args.servo]
    stl = HERE / "stl"
    OUT.mkdir(parents=True, exist_ok=True)
    for walk, robot, scene in ((False, "robot.xml", "scene.xml"), (True, "robot_walk.xml", "scene_walk.xml")):
        spec, report, damping, heights = build(sv, args.servo, walk)
        xml = spec.to_xml().replace(f'meshdir="{stl}"', 'meshdir="../../cad/sts3215/stl"')
        (OUT / robot).write_text(xml)
        (OUT / scene).write_text(SCENE.format(robot=robot))
        m = mujoco.MjModel.from_xml_path(str(OUT / scene))
        print(f"{OUT / scene}: {m.nbody - 1} bodies, {m.njnt} joints, {m.nu} actuators, servo {args.servo} "
              f"(stall {sv['stall_nm']} N m, joint damping {damping:.2f} N m s/rad), "
              f"trunk height at stand {heights['stand'] * 1000:.1f} mm")
    total = sum(r[1] for r in report)
    print("mass (g): " + ", ".join(f"{MJCF_NAME[n]} {M * 1000:.0f}" for n, M in report) + f" | total {total * 1000:.0f}")


if __name__ == "__main__":
    main()

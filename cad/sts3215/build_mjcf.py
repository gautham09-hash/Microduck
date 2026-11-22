"""Build the MuJoCo model of the STS3215 Microduck from layout.json and the exported part STLs.

Writes sim/sts3215/robot.xml and sim/sts3215/scene.xml. Each body's mass, centre of mass and inertia come from
its printed parts (mesh volume x PRINT_DENSITY), the servos it holds and the payload estimates in layout.json.

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
# default standing pose (rad), left leg / neck / head / right leg, same order as the joints below
STAND = dict(left_hip_yaw=0.0, left_hip_roll=-0.0873, left_hip_pitch=-0.4579, left_knee=-0.0049, left_ankle=0.4530,
             neck_pitch=0.3491, head_pitch=0.3491, head_yaw=0.0, head_roll=0.0, right_hip_yaw=0.0,
             right_hip_roll=0.0873, right_hip_pitch=0.4579, right_knee=0.0049, right_ankle=-0.4530)
JOINTS = list(STAND)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", default="19kg", choices=list(LAYOUT["servo"]["variants"]))
    args = ap.parse_args()
    sv = LAYOUT["servo"]["variants"][args.servo]
    damping = sv["stall_nm"] / sv["no_load_rad_s"]
    servo_mass = LAYOUT["servo"]["mass_g"] / 1000
    links = LAYOUT["links"]
    stl = HERE / "stl"
    servo_mesh = trimesh.load(stl / "sts3215.stl")

    spec = mujoco.MjSpec()
    spec.modelname = f"microduck_sts3215_{args.servo}"
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
        a = np.array(e["anchor"], float)
        if e["parent"] is None:
            b = spec.worldbody.add_body(name=name, pos=[0, 0, 0])
            b.add_freejoint(name="floating_base")
            b.add_site(name="imu", pos=[0, 0, 0.165])
        else:
            pa = np.array(links[e["parent"]]["anchor"], float)
            b = bodies[e["parent"]].add_body(name=name, pos=(a - pa) / 1000)
            lo, hi = np.radians(e["range_deg"])
            b.add_joint(name=e["joint"], type=mujoco.mjtJoint.mjJNT_HINGE, axis=e["axis"], range=[lo, hi],
                        damping=damping, armature=ARMATURE, frictionloss=FRICTION)
        bodies[name] = b
        items = []
        for part, link in MANIFEST.items():
            if link != name:
                continue
            mesh = trimesh.load(stl / (part + ".stl"))
            items.append(mesh_mass(mesh, density=PRINT_DENSITY))
            b.add_geom(name=part, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part, pos=-a / 1000,
                       contype=2, conaffinity=1, group=1, rgba=[0.85, 0.55, 0.2, 1], mass=0)
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
                       pos=(T[:3, 3] - a) / 1000, quat=q, contype=2, conaffinity=1, group=1,
                       rgba=[0.15, 0.15, 0.15, 1], mass=0)
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

    for j in JOINTS:
        act = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=KP)
        act.forcelimited = True
        act.forcerange = [-sv["stall_nm"], sv["stall_nm"]]
    for typ, n in ((mujoco.mjtSensor.mjSENS_GYRO, "imu_gyro"), (mujoco.mjtSensor.mjSENS_ACCELEROMETER, "imu_acc"),
                   (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "imu_quat")):
        spec.add_sensor(name=n, type=typ, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    # keyframes: base height so the soles touch the floor
    model = spec.compile()
    data = mujoco.MjData(model)
    soles = [model.geom(f"{s}_foot_sole").id for s in ("left", "right")]

    def keyed(pose):
        data.qpos[:] = model.qpos0
        for j, v in pose.items():
            data.qpos[model.joint(j).qposadr[0]] = v
        mujoco.mj_kinematics(model, data)
        low = min(data.geom_xpos[g][2] - model.geom_size[g][2] for g in soles)
        q = data.qpos.copy()
        q[2] -= low
        ctrl = np.array([pose.get(j, 0.0) for j in JOINTS])
        return q, ctrl

    for kname, pose in (("zero", {}), ("stand", STAND)):
        q, ctrl = keyed(pose)
        spec.add_key(name=kname, qpos=q, ctrl=ctrl)

    OUT.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml().replace(f'meshdir="{stl}"', 'meshdir="../../cad/sts3215/stl"')
    (OUT / "robot.xml").write_text(xml)
    (OUT / "scene.xml").write_text(f"""<mujoco model="microduck_sts3215 scene">
  <include file="robot.xml"/>
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
""")
    m = mujoco.MjModel.from_xml_path(str(OUT / "scene.xml"))
    total = sum(r[1] for r in report)
    print(f"{OUT / 'scene.xml'}: {m.nbody - 1} bodies, {m.njnt} joints, {m.nu} actuators, servo {args.servo} "
          f"(stall {sv['stall_nm']} N m, damping {damping:.2f} N m s/rad)")
    print("mass (g): " + ", ".join(f"{n} {M * 1000:.0f}" for n, M in report) + f" | total {total * 1000:.0f}")


if __name__ == "__main__":
    main()

"""Smoke test of the STS3215 model: does it stand, survive a small drop, and sit down / stand up slowly under
position control within the servos' torque limits? No learned policy is involved.

Run: uv run python sim/sts3215/smoke_test.py [--servo 19kg|30kg] [--render out.png]
(build the model first: uv run python cad/sts3215/build_mjcf.py --servo ...)
"""

import argparse
import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
CROUCH_DROP = 0.045   # m: how far the trunk comes down in the crouch


def crouch_pose(m, stand, names, drop):
    """Leg angles that lower the trunk by ``drop`` with the soles flat and in place (trunk held level)."""
    d = mujoco.MjData(m)

    def sole(q, side):
        d.qpos[:] = m.qpos0
        d.qpos[7:] = 0
        for i, n in enumerate(names):
            d.qpos[m.joint(n).qposadr[0]] = q[i]
        mujoco.mj_kinematics(m, d)
        s = m.site(f"{side}_foot_sole").id
        R = d.site_xmat[s].reshape(3, 3)
        return np.r_[d.site_xpos[s], np.arctan2(R[0, 2], R[2, 2])]   # position + pitch of the sole

    q = stand.copy()
    for side in ("left", "right"):
        idx = [names.index(f"{side}_{j}") for j in ("hip_pitch", "knee", "ankle")]
        goal = sole(stand, side) + np.array([0, 0, drop, 0])
        for _ in range(50):
            f = sole(q, side) - goal
            f[:2] *= 1.0
            if np.abs(f).max() < 1e-6:
                break
            J = np.zeros((4, 3))
            for k, i in enumerate(idx):
                dq = q.copy()
                dq[i] += 1e-6
                J[:, k] = (sole(dq, side) - goal - f) / 1e-6
            q[idx] -= np.linalg.lstsq(J, f, rcond=None)[0]
    return q


def tilt_deg(d, m):
    R = d.xmat[m.body("trunk").id].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))


def run(m, ctrl_fn, secs, drop=0.0):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    d.qpos[2] += drop
    stall = m.actuator_forcerange[:, 1]
    peak, tilt, zs = np.zeros(m.nu), 0.0, []
    for i in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = ctrl_fn(d.time)
        mujoco.mj_step(m, d)
        if d.time > 0.3:
            peak = np.maximum(peak, np.abs(d.actuator_force) / stall)
            tilt = max(tilt, tilt_deg(d, m))
            zs.append(d.xipos[m.body("trunk").id][2])
    return d, peak, tilt, np.array(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", help="save a picture of the stand pose")
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    names = [m.actuator(i).name for i in range(m.nu)]
    stand = m.key("stand").ctrl.copy()
    crouch = crouch_pose(m, stand, names, CROUCH_DROP)
    print("crouch pose (deg):", {n: round(float(np.degrees(v)), 1) for n, v in zip(names, crouch) if n.startswith("left")})
    print(m.opt.timestep, m.nu, "actuators", m.body_subtreemass[m.body("trunk").id].round(3), "kg")
    ok = True

    def report(label, d, peak, tilt, zs, need_up=True):
        nonlocal ok
        up = tilt < 20 and zs.min() > 0.08
        worst = np.argsort(peak)[::-1][:3]
        print(f"{label:28s} max tilt {tilt:5.1f} deg, trunk CoM height {zs.min() * 1000:4.0f}-{zs.max() * 1000:4.0f} mm, "
              f"{'UPRIGHT' if up else 'FELL'} | peak torque: " +
              ", ".join(f"{names[k]} {100 * peak[k]:.0f}%" for k in worst))
        ok &= up or not need_up

    report("hold stand 5 s", *run(m, lambda t: stand, 5.0))
    report("drop 10 mm, hold 3 s", *run(m, lambda t: stand, 3.0, drop=0.010))

    def squat(t, T=6.0):
        s = 0.5 - 0.5 * np.cos(2 * np.pi * min(t, T) / T)       # stand -> crouch -> stand over T seconds
        return stand + s * (crouch - stand)
    report("crouch and rise, 6 s", *run(m, squat, 7.0))
    report("crouch and rise, 2 s", *run(m, lambda t: squat(t, 2.0), 3.0))

    if args.render:
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        mujoco.mj_forward(m, d)
        with mujoco.Renderer(m, 600, 600) as r:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, 0, 0.17]
            cam.distance, cam.azimuth, cam.elevation = 0.75, 145, -15
            r.update_scene(d, cam)
            import PIL.Image
            PIL.Image.fromarray(r.render()).save(args.render)
            print("render ->", args.render)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

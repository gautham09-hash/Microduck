"""Collision check of the STS3215 parts over the joint ranges.

Loads layout.json (kinematics, servo poses), stl/*.stl (parts at the zero pose, world mm) and
stl/sts3215_check.stl (servo case without its discs, with the cable-plug zone), then checks every pair of
objects on different links with exact mesh collision (FCL):

  1. each joint swept alone over its design range (layout.json range_deg), others at zero;
  2. neighbouring joint pairs on a 5 deg grid;
  3. optionally, recorded joint trajectories (--traj file.npz: arrays of joint angles in degrees per motion,
     plus a 'joints' array naming the columns).

Run: uv run --with python-fcl python cad/sts3215/check_rom.py [--traj traj.npz]
"""

import argparse
import itertools
import json
import pathlib

import fcl
import numpy as np
import trimesh

HERE = pathlib.Path(__file__).parent
LAYOUT = json.loads((HERE / "layout.json").read_text())
LINKS = LAYOUT["links"]


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class Robot:
    def __init__(self):
        self.objs = []          # (name, link, fcl.CollisionObject, local AABB corners)
        manifest = json.loads((HERE / "stl" / "manifest.json").read_text())
        for name, link in manifest.items():
            self._add(name, link, trimesh.load(HERE / "stl" / (name + ".stl")))
        servo = trimesh.load(HERE / "stl" / "sts3215_check.stl")
        for joint, s in LAYOUT["servos"].items():
            m = servo.copy()
            m.apply_transform(np.array(s["transform_mm"]))
            self._add("servo:" + joint, s["link"], m)
        self.pairs = [(i, j) for i, j in itertools.combinations(range(len(self.objs)), 2)
                      if self.objs[i][1] != self.objs[j][1]]
        self.joint_link = {v["joint"]: k for k, v in LINKS.items() if "joint" in v}
        self.debug, self.seen = False, set()

    def _add(self, name, link, mesh):
        bvh = fcl.BVHModel()
        bvh.beginModel(len(mesh.vertices), len(mesh.faces))
        bvh.addSubModel(mesh.vertices, mesh.faces)
        bvh.endModel()
        corners = np.array(list(itertools.product(*zip(*mesh.bounds))))
        self.objs.append((name, link, fcl.CollisionObject(bvh, fcl.Transform()), corners))

    def link_poses(self, q):
        """q: {joint: deg} -> {link: (R, t)} mapping zero-pose world points to posed world points."""
        poses = {}

        def pose(link):
            if link in poses:
                return poses[link]
            e = LINKS[link]
            if e["parent"] is None:
                poses[link] = (np.eye(3), np.zeros(3))
            else:
                Rp, tp = pose(e["parent"])
                a = np.array(e["anchor"])
                Rj = rot(e["axis"], q.get(e["joint"], 0.0))
                R = Rp @ Rj
                poses[link] = (R, Rp @ (a - Rj @ a) + tp)
            return poses[link]

        for k in LINKS:
            pose(k)
        return poses

    def check(self, q, want_gap=False):
        """Returns list of (obj_a, obj_b, distance) for colliding pairs (distance <= 0), and the min gap."""
        poses = self.link_poses(q)
        boxes = []
        for name, link, obj, corners in self.objs:
            R, t = poses[link]
            obj.setTransform(fcl.Transform(R, t))
            w = corners @ R.T + t
            boxes.append((w.min(0), w.max(0)))
        hits, gap = [], (np.inf, None)
        for i, j in self.pairs:
            (lo1, hi1), (lo2, hi2) = boxes[i], boxes[j]
            sep = np.max(np.maximum(lo1 - hi2, lo2 - hi1))
            if sep > (5.0 if want_gap else 0.0):
                continue
            a, b = self.objs[i], self.objs[j]
            res = fcl.CollisionResult()
            if fcl.collide(a[2], b[2], fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), res):
                hits.append((a[0], b[0]))
                if self.debug and (a[0], b[0]) not in self.seen:
                    self.seen.add((a[0], b[0]))
                    p = np.array(res.contacts[0].pos)
                    Ra, ta = poses[a[1]]
                    print(f"      contact {a[0]} / {b[0]} at {q_str(q)}: world {p.round(1)}, "
                          f"zero-pose in {a[1]} {(Ra.T @ (p - ta)).round(1)}")
            elif want_gap:
                d = fcl.distance(a[2], b[2], fcl.DistanceRequest(), fcl.DistanceResult())
                if d < gap[0]:
                    gap = (d, f"{a[0]} <-> {b[0]}")
        return hits, gap


def q_str(q):
    return {k: round(float(v)) for k, v in q.items() if abs(v) > 0.5}


HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]


def leg_joints(side):
    return [f"{side}_{j}" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", help="npz of recorded joint trajectories (deg)")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--debug", action="store_true", help="print the first contact point of each colliding pair")
    args = ap.parse_args()
    rb = Robot()
    rb.debug = args.debug
    rng = {v["joint"]: v["range_deg"] for v in LINKS.values() if "range_deg" in v}
    ok = True

    hits, gap = rb.check({}, want_gap=True)
    print(f"zero pose: {'COLLISION ' + str(hits) if hits else 'clear'}; min gap {gap[0]:.2f} mm ({gap[1]})")
    ok &= not hits

    print("\n1. single-joint sweeps over the design range")
    for group in (leg_joints("left"), leg_joints("right"), HEAD_JOINTS):
        for j in group:
            lo, hi = rng[j]
            bad, worst = [], (np.inf, None)
            for v in np.arange(lo, hi + 1e-9, args.step):
                h, g = rb.check({j: v}, want_gap=True)
                if h:
                    bad.append((round(float(v), 1), h))
                elif g[0] < worst[0]:
                    worst = (g[0], f"{g[1]} @ {v:.0f} deg")
            msg = f"COLLIDES at {len(bad)} angles, e.g. {bad[0][0]} deg {bad[0][1][:2]}" if bad else \
                f"clear, min gap {worst[0]:.1f} mm ({worst[1]})"
            print(f"   {j:16s} [{lo:4.0f},{hi:4.0f}]  {msg}")
            ok &= not bad

    print("\n2. neighbouring joint pairs, 5 deg grid (information: full-range corners no motion uses)")
    pairs = []
    for side in ("left", "right"):
        y, r, p, k, a = leg_joints(side)
        pairs += [(y, r), (r, p), (p, k), (k, a)]
    pairs += [("left_hip_yaw", "right_hip_yaw"), ("left_hip_roll", "right_hip_roll"),
              ("left_hip_pitch", "right_hip_pitch"), ("neck_pitch", "head_pitch"), ("head_pitch", "head_yaw"),
              ("head_yaw", "head_roll")]
    for j1, j2 in pairs:
        n = bad = 0
        who = {}
        for v1 in np.arange(rng[j1][0], rng[j1][1] + 1e-9, 5.0):
            for v2 in np.arange(rng[j2][0], rng[j2][1] + 1e-9, 5.0):
                n += 1
                h, _ = rb.check({j1: v1, j2: v2})
                if h:
                    bad += 1
                    for x in h:
                        who[x] = who.get(x, 0) + 1
        top = sorted(who.items(), key=lambda kv: -kv[1])[:2]
        print(f"   {j1} x {j2}: {bad}/{n} colliding" + (f"  {top}" if top else ""))

    if args.traj:
        print("\n3. recorded trajectories (every 2nd frame)")
        z = np.load(args.traj)
        names = list(z["joints"])
        for motion in [k for k in z.files if k != "joints"]:
            Q = z[motion][::2]
            bad, who = 0, {}
            for row in Q:
                h, _ = rb.check(dict(zip(names, row)))
                if h:
                    bad += 1
                    for x in h:
                        who[x] = who.get(x, 0) + 1
            top = sorted(who.items(), key=lambda kv: -kv[1])[:3]
            print(f"   {motion:10s} {bad}/{len(Q)} frames colliding" + (f"  {top}" if top else ""))
            ok &= bad == 0
    print("\nRESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

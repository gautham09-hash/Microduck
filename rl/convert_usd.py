"""Convert sim/sts3215/robot_walk.xml to the USD the training task loads (sim/sts3215/usd/robot_walk/robot_walk.usda).

Run it with Isaac Lab's Python after every build_mjcf.py run:

    ~/IsaacLab/.venv/bin/python rl/convert_usd.py

Conversion runs here rather than at training start because inside the training process the importer fails with
"Cannot find a valid schema ... PhysicsJointStateAPI". Isaac Lab needs the importers extra
(uv sync --extra importers).

robot_walk.xml puts each servo joint and its gearbox-play hinge in the same body. The importer makes both joints
connect the same two bodies, which Newton reads as one 2-DOF joint, so afterwards a near-massless dummy body is
inserted between them (servo: parent -> dummy, play: dummy -> link), as in Isaac Lab's MicroDuck backlash model.
The play hinges also get that model's stiff limit settings, which the importer does not write: with the solver's
defaults a hinge on a near-massless body barely stops at its limit (40 degrees of play instead of 0.5).
"""

import pathlib
import shutil

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

REPO = pathlib.Path(__file__).resolve().parents[1]
# the importer writes to robot_walk_1/ if robot_walk/ exists, so start clean
shutil.rmtree(REPO / "sim" / "sts3215" / "usd", ignore_errors=True)

cfg = MjcfConverterCfg(
    asset_path=str(REPO / "sim" / "sts3215" / "robot_walk.xml"),
    usd_dir=str(REPO / "sim" / "sts3215" / "usd"),
    fix_base=False,
    self_collision=False,
    force_usd_conversion=True,
)
usd_path = MjcfConverter(cfg).usd_path

# limit settings of the play hinges in Isaac Lab's microduck_walk_backlash.usd
PLAY_LIMIT = {"mjc:solreflimit": [0.01, 1.0], "mjc:solimplimit": [0.95, 0.999, 0.0001, 0.5, 2.0]}


def split_backlash_joints(path: str) -> int:
    """Insert a dummy body between each servo joint and its sibling passive_<joint>_backlash hinge."""
    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    stage.SetEditTarget(stage.GetRootLayer())
    count = 0
    for prim in list(stage.Traverse()):
        name = prim.GetName()
        if not prim.IsA(UsdPhysics.RevoluteJoint) or name.startswith("passive_"):
            continue
        play_prim = prim.GetParent().GetChild(f"passive_{name}_backlash")
        if not play_prim:
            continue
        servo, play = UsdPhysics.RevoluteJoint(prim), UsdPhysics.RevoluteJoint(play_prim)
        parent_path, child_path = servo.GetBody0Rel().GetTargets()[0], servo.GetBody1Rel().GetTargets()[0]
        child = stage.GetPrimAtPath(child_path)
        assert child.GetParent().GetPath() == parent_path, f"{child_path} is not nested under {parent_path}"
        # the dummy sits where the link rests, so both joints keep the link-side frame
        dummy = UsdGeom.Xform.Define(stage, parent_path.AppendChild(f"passive_{name}_backlash_dummy"))
        rest = UsdGeom.Xformable(child).GetLocalTransformation()
        dummy.AddTransformOp().Set(rest)
        UsdPhysics.RigidBodyAPI.Apply(dummy.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(dummy.GetPrim())
        mass.CreateMassAttr(1e-6)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9, 1e-9, 1e-9))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        servo.GetBody1Rel().SetTargets([dummy.GetPath()])
        play.GetBody0Rel().SetTargets([dummy.GetPath()])
        play.GetLocalPos0Attr().Set(servo.GetLocalPos1Attr().Get())
        play.GetLocalRot0Attr().Set(servo.GetLocalRot1Attr().Get())
        play_prim.ApplyAPI("MjcJointAPI")
        for attr, value in PLAY_LIMIT.items():
            play_prim.CreateAttribute(attr, Sdf.ValueTypeNames.DoubleArray).Set(value)
        count += 1
    stage.GetRootLayer().Save()
    return count


print("USD:", usd_path, f"({split_backlash_joints(usd_path)} servo/play joint pairs split)")

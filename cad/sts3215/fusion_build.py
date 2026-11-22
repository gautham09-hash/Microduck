"""Fusion script: build the parts in build/parts.json into the open design and export their STLs.

Run from Fusion (Scripts and Add-Ins) with microduck_sts3215 open. It replaces the component ``parts_v0`` on every
run, so edit parts.py, not the Fusion bodies. Needs the ``sts3215`` component (the servo STEP) in the design.
Writes cad/sts3215/stl/<part>.stl, stl/sts3215.stl (the servo) and stl/sts3215_check.stl (servo without its discs,
plus the cable-plug zone, for the collision check).
"""

import json
import os

import adsk.core
import adsk.fusion

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "build", "parts.json")
STL_DIR = os.path.join(HERE, "stl")
ROOT_NAME = "parts_v0"
OLD_NAMES = ("legs_v0",)
CM = 0.1  # mm -> cm


def _p(v):
    return adsk.core.Point3D.create(v[0] * CM, v[1] * CM, v[2] * CM)


def _v(v):
    return adsk.core.Vector3D.create(*v)


def _prim(tb, q):
    if q["t"] == "box":
        R = q["R"]
        col = lambda i: _v([R[0][i], R[1][i], R[2][i]])
        h = q["h"]
        obb = adsk.core.OrientedBoundingBox3D.create(_p(q["c"]), col(0), col(1), 2 * h[0] * CM, 2 * h[1] * CM,
                                                     2 * h[2] * CM)
        return tb.createBox(obb)
    c, a, h = q["c"], q["a"], q["h"]
    p0 = [c[i] - a[i] * h for i in range(3)]
    p1 = [c[i] + a[i] * h for i in range(3)]
    return tb.createCylinderOrCone(_p(p0), q["r"] * CM, _p(p1), q["r"] * CM)


def _csg(tb, part):
    body = None
    for q in part["solids"]:
        b = _prim(tb, q)
        if body is None:
            body = b
        else:
            tb.booleanOperation(body, b, adsk.fusion.BooleanTypes.UnionBooleanType)
    for q in part["cuts"]:
        tb.booleanOperation(body, _prim(tb, q), adsk.fusion.BooleanTypes.DifferenceBooleanType)
    if part.get("clip"):
        keep = None
        for q in part["clip"]:
            b = _prim(tb, q)
            if keep is None:
                keep = b
            else:
                tb.booleanOperation(keep, b, adsk.fusion.BooleanTypes.UnionBooleanType)
        tb.booleanOperation(body, keep, adsk.fusion.BooleanTypes.IntersectionBooleanType)
    return body


def _export(design, body, path):
    opt = design.exportManager.createSTLExportOptions(body, path)
    opt.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
    opt.isBinaryFormat = True
    design.exportManager.execute(opt)


def run(_context):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    tb = adsk.fusion.TemporaryBRepManager.get()
    spec = json.load(open(PARTS))
    os.makedirs(STL_DIR, exist_ok=True)

    for occ in list(root.occurrences):
        if occ.component.name == ROOT_NAME or occ.component.name in OLD_NAMES:
            occ.deleteMe()
    top = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    top.component.name = ROOT_NAME
    report = []
    for part in spec["parts"]:
        body = _csg(tb, part)
        occ = top.component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        comp = occ.component
        comp.name = part["name"]
        base = comp.features.baseFeatures.add()
        base.startEdit()
        b = comp.bRepBodies.add(body, base)
        base.finishEdit()
        b = comp.bRepBodies.item(0)
        b.name = part["name"]
        _export(design, b, os.path.join(STL_DIR, part["name"] + ".stl"))
        report.append(f'{part["name"]}: {b.volume * 1000:.0f} mm3, lumps {b.lumps.count}')

    # servo check mesh: discs belong to the child part, the cable plugs to the servo
    servo = [o for o in root.occurrences if o.component.name == "sts3215"][0].component.bRepBodies.item(0)
    chk = tb.copy(servo)
    for z0, z1 in ((16.05, 30.0), (-30.0, -15.75)):
        tb.booleanOperation(chk, tb.createCylinderOrCone(_p([12.5, 0, z0]), 10.4 * CM, _p([12.5, 0, z1]), 10.4 * CM),
                            adsk.fusion.BooleanTypes.DifferenceBooleanType)
    plug = adsk.core.OrientedBoundingBox3D.create(_p([-1.6, 0, -19.95]), _v([1, 0, 0]), _v([0, 1, 0]),
                                                  4.0 * CM, 19.2 * CM, 8.1 * CM)
    tb.booleanOperation(chk, tb.createBox(plug), adsk.fusion.BooleanTypes.UnionBooleanType)
    tmp = top.component.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    base = tmp.component.features.baseFeatures.add()
    base.startEdit()
    tmp.component.bRepBodies.add(chk, base)
    base.finishEdit()
    _export(design, tmp.component.bRepBodies.item(0), os.path.join(STL_DIR, "sts3215_check.stl"))
    tmp.deleteMe()
    _export(design, servo, os.path.join(STL_DIR, "sts3215.stl"))
    print("\n".join(report))

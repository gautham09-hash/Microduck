# STS3215 build: skeleton v0

Fit-test skeleton for the whole robot, built around the 1.3× joint layout, plus the MuJoCo model generated from it.
The parts are blocky on purpose: they exist to check servo fit, screw positions, range of motion and mass before
any shaping. Shells, beak and the head's contents are not designed yet; their mass is an estimate in the model.

| File | What it is |
|---|---|
| `layout.json` | Joint frames, design ranges, servo poses, payload estimates (mm, Z up, +X forward) |
| `parts.py` | Defines each part as boxes and cylinders minus holes and servo keep-outs. Edit parameters here |
| `fusion_build.py` | Fusion script: builds the parts into the open design (component `parts_v0`) and exports `stl/` |
| `check_rom.py` | Collision check of all parts and servos over the joint ranges (exact mesh collision) |
| `build_mjcf.py` | Generates the MuJoCo model in `sim/sts3215/` from `layout.json` and `stl/` |
| `stl/` | Exported parts at the zero pose, in world coordinates (re-orient for printing), the servo, `manifest.json` |

## Regenerate

```bash
uv run python cad/sts3215/parts.py
```

Run `fusion_build.py` in Fusion with `microduck_sts3215` open. It replaces `parts_v0`, so hand edits to those bodies
are lost. Then:

```bash
uv run --with python-fcl python cad/sts3215/check_rom.py
```

```bash
uv run python cad/sts3215/build_mjcf.py --servo 19kg
```

```bash
uv run python sim/sts3215/smoke_test.py
```

To look at the model, open the MuJoCo viewer. Use plain `python`, not `mjpython`, which crashes here on MuJoCo 4.0.1:

```bash
uv run python -m mujoco.viewer --mjcf=sim/sts3215/scene.xml
```

In the viewer's Simulation panel, set Key to 1 (`stand`), click Load key, then press Space to run.

## Parts (17)

| Link | Parts | Holds |
|---|---|---|
| trunk | `trunk` | both hip yaw servos; clevis for the neck pitch servo |
| hip yaw link ×2 | `*_hip_yaw_bracket` + `*_hip_yaw_cap` (4 × M2) | hip roll servo |
| hip cross ×2 | `*_hip_cross` | — (clevis on roll and pitch) |
| thigh ×2 | `*_thigh` | hip pitch and knee servos |
| shin ×2 | `*_shin` | ankle servo |
| foot ×2 | `*_foot` | — (plus a 3 mm sole pad) |
| neck | `neck` | neck pitch and head pitch servos |
| neck top | `neck_top` | — (clevis on head pitch, plate on the head yaw horn) |
| head yaw link | `head_yaw_link` | head yaw servo; plate on the head roll horn |
| head | `head` | head roll servo (shells, beak and mouth servo still to design) |

Joints are clevises screwed to both the horn and the idler disc (4 × M3 each, no separate bearings), except the
hip yaw, head yaw and head roll, which are horn-only. Servos sit in cradles around the back of the case, screwed
through the cover holes (M2 self-tapping). The cable connectors on the idler side are left clear.

Printed volume is 284 cm³ in total. At ~0.75 g/cm³ (PLA, ~60 % fill) that is about 210 g.

## Status

`check_rom.py` passes. That covers every joint swept alone over its design range, plus recorded walking, turning,
sidestepping, sit-stand and fall-recovery motions, with at least 0.7 mm between parts. The neighbouring-joint grids
still collide at full-range corners that no motion uses.

| Joint | Design range (left; right is negated) |
|---|---|
| hip yaw | ±30° |
| hip roll | ±25° |
| hip pitch | −90° to +40° |
| knee | −75° to +80° |
| ankle | −40° to +90° |
| neck pitch | −90° to +60° |
| head pitch | −45° to +90° |
| head yaw | ±90° |
| head roll | ±25° |

The model weighs 1.49 kg (servos 770 g, printed parts ~210 g, payload estimates 510 g). `smoke_test.py` passes for
both servo versions. It holds the stand pose, survives a 10 mm drop, and crouches 45 mm and rises again under
position control. Peak torque in a 2 s crouch is 56 % of stall at the knees (19 kg servo) and 50 % (30 kg).

## Still open

- **Untested on hardware.** Horn screw holes (M3 clearance) and the cable-plug zone are unverified. Print one
  clevis and one cradle and check them on a real servo before printing the set.
- **Upper body unfinished.** Head shells, beak and mouth servo mount, trunk shell, battery holder, electronics
  mounts and cable routing are not designed.
- **Sim servo constants are starting values.** Armature, friction and kp need a step-response measurement on a
  real STS3215.
- **Unfinished geometry.** No fillets or lightening; the foot plate is solid 6 mm.

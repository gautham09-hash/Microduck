# CAD

## What's here

| Path | Contents | Format |
|---|---|---|
| `baseline_hd1910/parts/` | 38 printable parts, one STL each (mm) | STL mesh |
| `baseline_hd1910/openmicroduck-assembly-opt.stl` | Full assembly of the parts above (mm, Y-up, 264 mm tall) | STL mesh |
| `baseline_hd1910/openmicroduck.3mf` | The same parts laid out on Bambu Studio build plates | 3MF |
| `original_xl330/microduck_combined*.stl` | Original XL330 robot, assembled (meters, Z-up). `_rollers` has roller feet | STL mesh |
| `servos/HD-1910-C001-20260902.stp` | Feetech HD-1910 servo | STEP |
| `servos/XL330.STEP` | Dynamixel XL330 servo | STEP |

None of the printed parts have an editable (parametric) source. Every part is a mesh, so changing one means
remodelling it in a CAD tool using the STL as a reference, not editing the STL directly.

## Robot structure

15 servos: 5 per leg, 4 in the neck/head, 1 for the mouth. The mouth servo isn't simulated, so sim models have
14 actuated joints.

| Chain | Joints, trunk outward | Servo IDs |
|---|---|---|
| Right leg | hip_yaw → hip_roll → hip_pitch → knee → ankle | 10–14 |
| Left leg | hip_yaw → hip_roll → hip_pitch → knee → ankle | 20–24 |
| Neck/head | neck_pitch → head_pitch → head_yaw → head_roll, + mouth | 30–34 |

Reference figures: about 0.74 kg total (XL330 build), 264 mm tall. The control loop runs at 50 Hz.

## Part catalogue (`baseline_hd1910/parts/`)

Sizes are bounding boxes in mm. Files ending in `1`/`2` (e.g. `hip_l1`, `leg2`) are the second copy for the other side.

| Group | Parts |
|---|---|
| Head | `top_head_shell` 123×46×92, `bottom_head_shell`, `face_part`, `jaw`, `jaw_soft`, `soft_mouth_top`, `noenoeil` (eye), `banana_pcb_locker` |
| Neck | `neck`, `neck1`, `neck_pitch` |
| Trunk & power | `trunk_base`, `left_shell`/`right_shell` (or `*_battery_opt` variants), `power_support` (or `power_support_battery_opt`), `battery_holder_18650` / `battery_holder_21700`, `motor_support` |
| Hips | `hip_l`, `hip_l1`, `yaw2roll`, `yaw2roll1`, `yaw_roll_motion` |
| Legs | `upper_leg_left`, `upper_leg_right`, `upper_leg_rigidity_plate` ×2, `leg`, `leg2` |
| Feet | `ankle_left`, `ankle_right`, `foot_left`, `foot_right`, `sole_left`, `sole_right` (TPU or silicone pad) |

### Changes already made in the HD1910 baseline vs. the XL330 original

The HD1910's rear hub sticks out where the XL330's is recessed. That one difference drives most of these changes:

- **Head**: screw boss trimmed where it hit the servo's rear hub. `jaw_soft` shortened so it no longer rubs `bottom_head_shell`.
- **Neck**: link thickened from 1 mm, which was fragile to print.
- **Trunk**: `trunk_base` thickened. Every servo held by 3 screws instead of 2. `power_support` now locates on the servos. IMU mount fits an LSM6DSV16X board.
- **Hip**: the yaw bracket was two clamping parts, now one piece (`yaw2roll`). Leaves room to drive the trunk screws.
- **Upper leg**: rigidity plate thickened. Two locating pins removed to clear the servo and ease wiring.
- **Ankle**: curved interior so the foot can rotate. Snap rib to mount on the leg and retain the bearing from outside.

Bearings: 6700K (10×15×3) ×3, ET2216 (16×22×4) ×11.

## Next: STS3215 adaptation

The STS3215 is much larger and heavier than the servos this structure was designed around:

| Servo | Body (mm) | Mass |
|---|---|---|
| Dynamixel XL330 | 20 × 34 × 26 | 18 g |
| Feetech HD-1910 | 20 × 34 × 23 | 21 g |
| Feetech STS3215 | 24.7 × 45.2 × 35 | 55 g |

Every servo pocket, horn interface and screw pattern changes. The link lengths and the overall size probably have
to grow too, and servo mass alone goes from about 0.35 kg to about 0.8 kg. Steps:

1. Add an STS3215 STEP model to `servos/`.
2. Decide whether to keep the 25 cm proportions or scale the robot up.
3. Remodel the servo-holding parts (trunk, hips, legs, ankles, neck) in a parametric CAD tool, using the baseline STLs as reference.
4. Export new STLs to `cad/sts3215/` and an assembly for the robot description.

## License

The files in this folder are licensed under **CC BY-NC-SA 4.0** ([`LICENSE-CC-BY-NC-SA-4.0.txt`](LICENSE-CC-BY-NC-SA-4.0.txt)).
They are based on Pollen Robotics' Microduck 3D models (CC BY-NC-SA,
https://github.com/pollen-robotics/microduck_rl), with HD1910 adaptations by JoyandAI
(https://github.com/JoyandAI/OpenMicroDuck). Derivatives must keep this license and may not be used commercially.
This project is not affiliated with or endorsed by Pollen Robotics.

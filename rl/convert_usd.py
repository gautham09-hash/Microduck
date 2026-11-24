"""Convert sim/sts3215/robot_walk.xml to the USD the training task loads (sim/sts3215/usd/robot_walk/robot_walk.usda).

Run it with Isaac Lab's Python after every build_mjcf.py run:

    ~/IsaacLab/.venv/bin/python rl/convert_usd.py

Conversion runs here rather than at training start because inside the training process the importer fails with
"Cannot find a valid schema ... PhysicsJointStateAPI". Isaac Lab needs the importers extra
(uv sync --extra importers).
"""

import pathlib
import shutil

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
print("USD:", MjcfConverter(cfg).usd_path)

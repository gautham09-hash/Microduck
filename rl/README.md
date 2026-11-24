# Training (Isaac Lab)

Isaac Lab tasks for the STS3215 Microduck. Physics is Newton / MuJoCo-Warp; Isaac Sim is not needed.

| Task | What it trains |
|---|---|
| `Microduck-STS3215-Velocity-Flat` | walk on flat ground following forward, sideways and turn commands |

It starts from Isaac Lab's MicroDuck walking task (same observations, actions and rewards; the model uses the same
body and joint names) and changes what differs for this robot:

- **Robot model:** `sim/sts3215/robot_walk.xml`, converted to USD.
- **Servos:** DC-motor servos, 1.91 N·m stall and 5.45 rad/s no-load, from the label on the servo box.
- **Commands:** slower, with forward speed −0.15 to 0.25 m/s, sideways ±0.1 m/s and turning ±0.6 rad/s.
- **Gait:** foot lift and step timing scaled to the bigger robot.
- **Randomization:** servo friction and gains, and trunk and head mass, since those are not measured yet.

## Setup (Linux, NVIDIA GPU)

Isaac Lab's environment needs the `importers` extra for the MJCF converter:

```bash
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
```

```bash
cd ~/IsaacLab && uv sync --extra importers
```

Install this package into that environment:

```bash
uv pip install -e ~/microduck/rl --python ~/IsaacLab/.venv/bin/python
```

A plain `uv sync` removes the extra and this package again. Rerun both commands above after any sync.

After every model change (`cad/sts3215/build_mjcf.py`), convert the model to USD:

```bash
cd ~/microduck && ~/IsaacLab/.venv/bin/python rl/convert_usd.py
```

Check that it stands with the servos holding the default pose:

```bash
cd ~/microduck && ~/IsaacLab/.venv/bin/python rl/check_stand.py
```

## Train

Use tmux so the run survives a dropped connection:

```bash
tmux new -s md_train
```

```bash
cd ~/IsaacLab && .venv/bin/python scripts/reinforcement_learning/train.py --rl_library rsl_rl --task Microduck-STS3215-Velocity-Flat --num_envs 4096
```

On an RTX 4000 Ada this runs at about 108k steps/s, 0.9 s per iteration, so the default 8,000 iterations take
about 2 h. Checkpoints go to `~/IsaacLab/logs/rsl_rl/microduck_sts3215_flat/<run>/model_<iteration>.pt` every
250 iterations.

For a quick config check, add `--num_envs 64 --max_iterations 3`.

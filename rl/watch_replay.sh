#!/usr/bin/env bash
# Follow training in the Viser web viewer (port 8080): replay the latest run with its newest checkpoint and
# swap in each newer checkpoint as training saves it (rl/live_replay.py), without restarting the viewer.
#
#   tmux new -d -s md_replay "bash ~/microduck/rl/watch_replay.sh"
#
# Then open http://<box>:8080 (or tunnel it: ssh -N -L 8080:127.0.0.1:8080 <box>). It shares the GPU with
# training and slows it by about 20 %. Stop it with: tmux kill-session -t md_replay
set -u
ISAACLAB=${ISAACLAB:-$HOME/IsaacLab}
TASK=${TASK:-Microduck-STS3215-Velocity-Flat}
NUM_ENVS=${NUM_ENVS:-16}
HERE=$(cd "$(dirname "$0")" && pwd)

cd "$ISAACLAB" && exec .venv/bin/python "$HERE/live_replay.py" --rl_library rsl_rl --task "$TASK" \
    --num_envs "$NUM_ENVS" --checkpoint latest --real-time --viz viser

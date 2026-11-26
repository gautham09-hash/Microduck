"""Replay a training run while it trains: the policy is swapped for each newer checkpoint, in place.

Same as Isaac Lab's RSL-RL replay (play.py), but every few seconds it looks for a newer model_*.pt in the run's
folder and loads it into the running policy, so the viewer stays connected and follows training. Run from the
Isaac Lab folder (checkpoint selectors look in ./logs):

    cd ~/IsaacLab && .venv/bin/python ~/microduck/rl/live_replay.py --rl_library rsl_rl \
        --task Microduck-STS3215-Velocity-Flat --num_envs 16 --checkpoint latest --real-time --viz viser

It shares the GPU with training and slows it by about 20 %.
"""

import contextlib
import glob
import os
import re
import sys
import time

import warp as wp

wp.config.enable_backward = False  # as Isaac Lab's play.py

import numpy as np  # noqa: E402
from newton.viewer import ViewerViser  # noqa: E402

import microduck_sts3215  # noqa: E402, F401  (registers the tasks)
from isaaclab.app import launch_simulation  # noqa: E402

# Isaac Lab's Viser visualizer calls ViewerViser._to_numpy, which the installed Newton build no longer has; the
# viewer dies ~20 s in when it redraws the ground grid. Put the helper back.
if not hasattr(ViewerViser, "_to_numpy"):

    def _to_numpy(array):
        if array is None:
            return None
        if hasattr(array, "numpy"):
            return array.cpu().numpy() if hasattr(array, "cpu") else array.numpy()
        return np.asarray(array)

    ViewerViser._to_numpy = staticmethod(_to_numpy)
from isaaclab.envs import DirectMARLEnvCfg

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint
from isaaclab_rl.entrypoints.backends.runner_rsl_rl import create_rsl_rl_runner
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env, run_playback
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import resolve_task_config

CHECK_EVERY_S = 10.0
SETTLE_S = 5.0  # only load checkpoints at least this old, so training has finished writing them


def newest_checkpoint(run_dir):
    found = [(int(re.search(r"model_(\d+)\.pt$", p).group(1)), p) for p in glob.glob(os.path.join(run_dir, "model_*.pt"))]
    found = [(i, p) for i, p in found if time.time() - os.path.getmtime(p) > SETTLE_S]
    return max(found) if found else (None, None)


def main(argv):
    if argv and argv[0] == "--rl_library":
        argv = argv[2:]  # accepted for symmetry with play.py; this script is RSL-RL only
    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root)
        run_dir = os.path.dirname(path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli,
                                  convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg))
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
        runner.load(path, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        current = int(re.search(r"model_(\d+)\.pt$", path).group(1))
        print(f"{time.strftime('%X')} showing {os.path.basename(run_dir)}/model_{current}.pt", flush=True)
        obs = env.get_observations()
        last_check = time.monotonic()

        def step():
            nonlocal obs, policy, current, last_check
            if time.monotonic() - last_check > CHECK_EVERY_S:
                last_check = time.monotonic()
                it, newest = newest_checkpoint(run_dir)
                if it is not None and it > current:
                    runner.load(newest, map_location=agent_cfg.device)
                    policy = runner.get_inference_policy(device=env.unwrapped.device)
                    current = it
                    print(f"{time.strftime('%X')} showing {os.path.basename(run_dir)}/model_{it}.pt", flush=True)
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)

        run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)


if __name__ == "__main__":
    main(sys.argv[1:])

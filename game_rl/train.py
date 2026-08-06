#!/usr/bin/env python3
"""Train a masked conv policy on the shapez build env.

    PYTHONPATH=. .venv/bin/python train.py --timesteps 200000
"""

import argparse
import os
import sys

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shapez_rl.encoding import DEFAULT_BUILDINGS, ROTATIONS
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.expert import HUB_TILE
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.policy import SpatialMaskablePolicy
from shapez_rl.wrappers import FlatAction


def bounds_arg(text):
    try:
        x, y, w, h = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected four integers: x,y,w,h")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return {"x": x, "y": y, "w": w, "h": h}


def buildings_arg(text):
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("expected at least one building")
    unknown = [name for name in names if name not in DEFAULT_BUILDINGS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown: {', '.join(unknown)}; valid: {', '.join(DEFAULT_BUILDINGS)}"
        )
    return names


def make_env(base_url, budget, ticks, target_shape, servers, bounds, buildings, log_path=None):
    """Env factory. With no base_url, each env gets its own fake server."""

    def _init():
        url = base_url
        if url is None:
            server = FakeShapezServer().start()
            servers.append(server)
            url = server.base_url

        env = ShapezBuildEnv(
            base_url=url,
            bounds=bounds,
            buildings=buildings,
            placement_budget=budget,
            run_ticks=ticks,
            target_shape=target_shape,
        )
        # info_keywords lands the reward-ladder rungs in the CSV alongside reward.
        return Monitor(
            FlatAction(env),
            filename=log_path,
            info_keywords=("mined", "delivered", "placements_succeeded"),
        )

    return _init


def build_model(venv, args):
    example = venv.get_attr("env")[0]  # unwrap Monitor -> FlatAction

    return MaskablePPO(
        SpatialMaskablePolicy,
        venv,
        policy_kwargs=dict(
            n_buildings=len(example.env.buildings),
            n_rotations=len(ROTATIONS),
            feature_channels=args.channels,
        ),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=1.0,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.tensorboard,
        device=args.device,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="Real API; omit to use the fake")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--budget", type=int, default=24, help="Buildings placed per episode")
    parser.add_argument("--ticks", type=int, default=3000, help="Ticks in the run phase")
    parser.add_argument("--target-shape", default=None, help="Score one shape, e.g. CuCuCuCu")
    parser.add_argument(
        "--bounds",
        type=bounds_arg,
        default=None,
        help="Play area as x,y,w,h; default -16,-16,32,32",
    )
    parser.add_argument(
        "--buildings",
        type=buildings_arg,
        default=DEFAULT_BUILDINGS,
        help="Comma-separated subset to allow, e.g. miner,belt",
    )
    parser.add_argument("--channels", type=int, default=64, help="Conv trunk width")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout per env; default 10 episodes")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: one episode")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tensorboard", default=None, help="Log dir; omit to disable")
    parser.add_argument("--save", default="runs/ppo_shapez", help="Checkpoint path, no extension")
    parser.add_argument(
        "--log-dir",
        default="runs/logs",
        help="Per-episode CSV dir; empty string to disable",
    )
    args = parser.parse_args()

    if args.n_steps is None:
        args.n_steps = args.budget * 10
    if args.batch_size is None:
        args.batch_size = args.budget

    if args.base_url is None:
        print("no --base-url given, using the offline fake API\n")

    if args.bounds is not None:
        hub_x, hub_y = HUB_TILE
        inside = (
            args.bounds["x"] <= hub_x < args.bounds["x"] + args.bounds["w"]
            and args.bounds["y"] <= hub_y < args.bounds["y"] + args.bounds["h"]
        )
        if not inside:
            # Without the hub in frame nothing can be delivered, so reward stays 0.
            print(f"warning: bounds exclude the hub at {HUB_TILE}; reward is unreachable\n")

    servers = []
    log_dir = args.log_dir or None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"per-episode CSV -> {os.path.join(log_dir, '<n>.monitor.csv')}\n")

    venv = DummyVecEnv(
        [
            make_env(
                args.base_url,
                args.budget,
                args.ticks,
                args.target_shape,
                servers,
                args.bounds,
                args.buildings,
                os.path.join(log_dir, str(i)) if log_dir else None,
            )
            for i in range(args.n_envs)
        ]
    )

    try:
        model = build_model(venv, args)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model.save(args.save)
        print(f"\nsaved {args.save}.zip")
    finally:
        venv.close()
        for server in servers:
            server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

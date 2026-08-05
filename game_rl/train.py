#!/usr/bin/env python3
"""Train a masked conv policy on the shapez build env.

    PYTHONPATH=. .venv/bin/python train.py --timesteps 200000
"""

import argparse
import sys

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shapez_rl.encoding import ROTATIONS
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.policy import SpatialMaskablePolicy
from shapez_rl.wrappers import FlatAction


def make_env(base_url, budget, ticks, target_shape, servers):
    """Env factory. With no base_url, each env gets its own fake server."""

    def _init():
        url = base_url
        if url is None:
            server = FakeShapezServer().start()
            servers.append(server)
            url = server.base_url

        env = ShapezBuildEnv(
            base_url=url,
            placement_budget=budget,
            run_ticks=ticks,
            target_shape=target_shape,
        )
        return Monitor(FlatAction(env))

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
    args = parser.parse_args()

    if args.n_steps is None:
        args.n_steps = args.budget * 10
    if args.batch_size is None:
        args.batch_size = args.budget

    if args.base_url is None:
        print("no --base-url given, using the offline fake API\n")

    servers = []
    venv = DummyVecEnv(
        [
            make_env(args.base_url, args.budget, args.ticks, args.target_shape, servers)
            for _ in range(args.n_envs)
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

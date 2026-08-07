#!/usr/bin/env python3
"""Score a trained policy against the expert and random baselines.

Every policy sees the *same* maps: each episode is replayed from a fixed seed, so
the three columns are comparable row by row rather than only on average.

    PYTHONPATH=. .venv/bin/python evaluate.py --model runs/ppo_shapez
"""

import argparse
import sys

import numpy as np
from sb3_contrib import MaskablePPO

from shapez_rl.encoding import DEFAULT_BUILDINGS

from shapez_rl.env import ShapezBuildEnv
from shapez_rl.expert import run_episode as run_expert_episode
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.wrappers import FlatAction
from train import bounds_arg, buildings_arg


def run_policy_episode(env, model, seed, deterministic):
    obs, _info = env.reset(seed=seed)

    reward, info, terminated = 0.0, {}, False
    while not terminated:
        action, _state = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=deterministic,
        )
        obs, reward, terminated, _truncated, info = env.step(action)

    return reward, info


def run_random_episode(env, rng, seed):
    env.reset(seed=seed)

    reward, terminated = 0.0, False
    while not terminated:
        legal = np.flatnonzero(env.action_masks())
        _obs, reward, terminated, _truncated, _info = env.step(int(rng.choice(legal)))

    return reward


def evaluate(env, model, args):
    rng = np.random.default_rng(args.seed)
    rows = []

    print(f"{'seed':>6}  {'policy':>22}  {'expert':>22}  {'random':>8}")
    for offset in range(args.episodes):
        seed = args.seed_start + offset

        policy_reward, info = run_policy_episode(env, model, seed, args.deterministic)
        expert_reward, _expert_info, planned = run_expert_episode(
            env.env, seed=seed, target_shape=args.target_shape
        )
        random_reward = run_random_episode(env, rng, seed)

        rows.append((policy_reward, expert_reward, random_reward))
        print(
            f"{seed:>6}  "
            f"{policy_reward:>8.1f} ({info['delivered']:>3} del, "
            f"{info['placements_succeeded']:>2} placed)  "
            f"{expert_reward:>8.1f} ({planned:>3} planned)          "
            f"{random_reward:>8.1f}"
        )

    policy, expert, random_ = (np.array(column) for column in zip(*rows))
    print(
        f"\nmean policy {policy.mean():.2f}   "
        f"mean expert {expert.mean():.2f}   "
        f"mean random {random_.mean():.2f}"
    )
    print(f"policy beats random on {int((policy > random_).sum())}/{len(rows)} maps")
    print(f"policy matches expert on {int((policy >= expert).sum())}/{len(rows)} maps")

    return policy.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint path, with or without .zip")
    parser.add_argument("--base-url", default=None, help="Real API; omit to use the fake")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--budget", type=int, default=24, help="Buildings placed per episode")
    parser.add_argument("--ticks", type=int, default=3000, help="Ticks in the run phase")
    parser.add_argument("--target-shape", default=None, help="Score one shape, e.g. CuCuCuCu")
    parser.add_argument(
        "--goal-level",
        type=int,
        default=None,
        help="Hub level to start at; must match what the model trained on",
    )
    parser.add_argument(
        "--bounds",
        type=bounds_arg,
        default=None,
        help="Play area as x,y,w,h; must match what the model trained on",
    )
    parser.add_argument(
        "--buildings",
        type=buildings_arg,
        default=DEFAULT_BUILDINGS,
        help="Comma-separated subset; must match what the model trained on",
    )
    parser.add_argument("--seed-start", type=int, default=1000, help="First map seed")
    parser.add_argument("--seed", type=int, default=0, help="Random baseline's RNG seed")
    parser.add_argument("--deterministic", action="store_true", help="Argmax instead of sampling")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)

    server = None
    base_url = args.base_url
    if base_url is None:
        print("no --base-url given, using the offline fake API\n")
        server = FakeShapezServer().start()
        base_url = server.base_url

    env = FlatAction(
        ShapezBuildEnv(
            base_url=base_url,
            bounds=args.bounds,
            buildings=args.buildings,
            placement_budget=args.budget,
            run_ticks=args.ticks,
            target_shape=args.target_shape,
            goal_level=args.goal_level,
        )
    )

    expected = model.observation_space.shape
    actual = env.observation_space.shape
    if expected != actual:
        raise SystemExit(
            f"model expects observations {expected} but this env produces {actual}; "
            "pass the --bounds and --buildings the model was trained with"
        )

    try:
        evaluate(env, model, args)
    finally:
        env.close()
        if server is not None:
            server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

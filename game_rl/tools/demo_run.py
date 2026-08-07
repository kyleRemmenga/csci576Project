#!/usr/bin/env python3
"""Play one episode slowly against a live game so it can be screen-recorded.

Point this at a running shapez instance and watch its window: the policy places
one building at a time with a pause in between, then the factory runs.

    python -m tools.demo_run --model runs/cutter_bc2/ppo_cutter_bc \
        --base-url http://127.0.0.1:17872 --seed 1000 --delay 0.6 \
        --bounds -12,-3,18,14 --buildings miner,belt,cutter,trash \
        --target-shape=----CuCu --goal-level 2 --budget 32 --ticks 1200
"""

import argparse
import time

from sb3_contrib import MaskablePPO

from shapez_rl.client import stored_shapes
from shapez_rl.encoding import DEFAULT_BUILDINGS
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.wrappers import FlatAction
from train import bounds_arg, buildings_arg


def describe(env, action):
    building_id, x, y, rotation = env.env._decode_action(env.action(action))
    return f"{building_id:<6} at ({x:>3},{y:>3}) rot {rotation:>3}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:17872")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds to play back-to-back in one recording, e.g. 1000,1005,1014",
    )
    parser.add_argument("--delay", type=float, default=0.6, help="Seconds between placements")
    parser.add_argument("--lead-in", type=float, default=3.0, help="Pause after reset, to start recording")
    parser.add_argument("--run-chunks", type=int, default=20, help="Extra visible tick batches after the build")
    parser.add_argument("--chunk-ticks", type=int, default=60)
    parser.add_argument("--chunk-delay", type=float, default=0.4)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--target-shape", default=None)
    parser.add_argument("--goal-level", type=int, default=None)
    parser.add_argument("--bounds", type=bounds_arg, default=None)
    parser.add_argument("--buildings", type=buildings_arg, default=DEFAULT_BUILDINGS)
    parser.add_argument("--sample", action="store_true", help="Sample instead of argmax")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model, device=args.device)
    env = FlatAction(
        ShapezBuildEnv(
            base_url=args.base_url,
            bounds=args.bounds,
            buildings=args.buildings,
            placement_budget=args.budget,
            run_ticks=args.ticks,
            target_shape=args.target_shape,
            goal_level=args.goal_level,
        )
    )

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    try:
        for episode_index, seed in enumerate(seeds):
            obs, _info = env.reset(seed=seed)
            print(f"\n=== episode {episode_index + 1}/{len(seeds)}  seed {seed} ===")
            print(f"seed {seed} loaded; recording starts in {args.lead_in:.0f}s")
            time.sleep(args.lead_in)

            step, terminated, reward, info = 0, False, 0.0, {}
            while not terminated:
                action, _state = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=not args.sample
                )
                print(f"  {step + 1:>2}/{args.budget}  {describe(env, int(action))}")
                obs, reward, terminated, _truncated, info = env.step(action)
                step += 1
                if not terminated:
                    time.sleep(args.delay)

            print(
                f"\nreward {reward:.1f}  delivered {info['delivered']}  "
                f"progress {info['progress']:.2f}  placed {info['placements_succeeded']}"
            )

            for round_index in range(args.run_chunks):
                state = env.env.client.tick(args.chunk_ticks)
                stored = sum(stored_shapes(state).values())
                print(f"  run {round_index + 1:>2}/{args.run_chunks}  stored shapes {stored}")
                time.sleep(args.chunk_delay)
    finally:
        env.close()


if __name__ == "__main__":
    main()

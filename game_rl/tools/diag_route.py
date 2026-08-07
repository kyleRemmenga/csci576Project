"""Diagnostic: does a raw miner->belt->hub line earn source-route credit?"""

import argparse

from shapez_rl.env import ShapezBuildEnv, belt_item_keys, belt_progress
from shapez_rl.expert import plan_mining_line, plan_to_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:17872")
    parser.add_argument("--goal-level", type=int, default=2)
    parser.add_argument("--target-shape", default="----CuCu")
    parser.add_argument("--bounds", default="-12,-3,18,14")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    x, y, w, h = (int(v) for v in args.bounds.split(","))
    env = ShapezBuildEnv(
        base_url=args.base_url,
        bounds={"x": x, "y": y, "w": w, "h": h},
        buildings=("miner", "belt", "cutter", "trash"),
        placement_budget=64,
        run_ticks=1200,
        target_shape=args.target_shape,
        goal_level=args.goal_level,
    )

    env.reset(seed=args.seed)
    print("wanted :", sorted(env._wanted))
    print("sources:", sorted(env._sources))
    print("waste  :", sorted(env._waste))

    plan = plan_mining_line(env._map_cache, env.bounds, target_shape="CuCuCuCu")
    if plan is None:
        print("NO PLAN -- no reachable CuCuCuCu patch in bounds")
        return
    print(f"plan length: {len(plan)} (miner + {len(plan) - 1} belts)")

    actions = plan_to_actions(plan, env)
    env.placement_budget = len(actions)
    for action in actions[:-1]:
        env.step(action)
    _, reward, _, _, info = env.step(actions[-1])

    print("placements ok :", env._placed_ok, "/", len(actions))
    print("failures      :", env._failures)
    print("reward (last) :", reward)
    print("info          :", {k: info[k] for k in ("mined", "progress", "delivered")})

    state = env.client.gamestate()
    print("belt items    :", sorted(belt_item_keys(state)))
    print("progress wanted :", belt_progress(state, env._wanted))
    print("progress source :", belt_progress(state, env._sources))


if __name__ == "__main__":
    main()

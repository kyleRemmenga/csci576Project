"""Diagnostic: does the scripted cutter expert actually deliver the cut half?"""

import argparse

from shapez_rl import expert as expert_mod
from shapez_rl.env import ShapezBuildEnv, belt_item_keys, belt_progress
from shapez_rl.expert import ScriptedMiner


def run(env, seed, slot, verbose=False):
    expert_mod.GOAL_OUTPUT_SLOT = slot
    env.reset(seed=seed)
    agent = ScriptedMiner(env)
    planned = agent.reset(env._map_cache)
    if planned == 0:
        return {"slot": slot, "planned": 0}

    total, terminated, info = 0.0, False, {}
    while not terminated:
        _obs, reward, terminated, _trunc, info = env.step(agent.act())
        total += reward

    state = env.client.gamestate()
    dump = state.get("savegame", {}).get("dump", {})
    if verbose:
        origins = {}
        for entity in dump.get("entities", []):
            static = entity.get("components", {}).get("StaticMapEntity", {})
            if static.get("origin") is not None:
                origins[entity.get("uid")] = (static["origin"]["x"], static["origin"]["y"])
        print("  plan:", agent.plan)
        for path in dump.get("beltPaths", []):
            ep = path.get("entityPath") or []
            items = sorted({e[1]["data"] for e in path.get("items", [])
                            if isinstance(e, (list, tuple)) and isinstance(e[1], dict) and e[1].get("data")})
            print(f"  path {origins.get(ep[0]) if ep else None} -> "
                  f"{origins.get(ep[-1]) if ep else None} len={len(ep)} items={items}")

    return {
        "slot": slot,
        "planned": planned,
        "reward": round(total, 2),
        "delivered": info.get("delivered"),
        "waste_routed": info.get("waste_routed"),
        "progress": info.get("progress"),
        "belt_items": sorted(belt_item_keys(state)),
        "failures": dict(env._failures),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:17872")
    parser.add_argument("--goal-level", type=int, default=2)
    parser.add_argument("--target-shape", default="----CuCu")
    parser.add_argument("--bounds", default="-12,-3,18,14")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--slots", default="0")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    x, y, w, h = (int(v) for v in args.bounds.split(","))
    env = ShapezBuildEnv(
        base_url=args.base_url,
        bounds={"x": x, "y": y, "w": w, "h": h},
        buildings=("miner", "belt", "cutter", "trash"),
        placement_budget=args.budget,
        run_ticks=args.ticks,
        target_shape=args.target_shape,
        goal_level=args.goal_level,
    )

    for seed in (int(s) for s in args.seeds.split(",")):
        for slot in (int(s) for s in args.slots.split(",")):
            env._failures = {}
            print(f"seed {seed} {run(env, seed, slot, args.verbose)}")


if __name__ == "__main__":
    main()

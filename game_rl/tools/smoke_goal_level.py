"""Check that --goal-level really unlocks what the goal needs.

Setting hubGoals.level alone leaves buildings locked, which shows up as a run
that trains for hours and scores zero. Run this against one game first.
"""

import argparse

from shapez_rl.client import ShapezClient
from shapez_rl.env import hub_goal_shape, source_shapes, waste_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:17872")
    parser.add_argument("--goal-level", type=int, default=2)
    parser.add_argument("--buildings", default="miner,belt,cutter,trash")
    args = parser.parse_args()

    client = ShapezClient(args.base_url)
    state = client.reset(seed=1, goal_level=args.goal_level)

    goal = hub_goal_shape(state)
    print(f"goal level {args.goal_level} -> goal shape {goal}")
    print(f"  source {sorted(source_shapes(goal))}  waste {sorted(waste_shapes(goal))}")

    failed = []
    for index, building in enumerate(args.buildings.split(",")):
        x = -10 + index * 2
        try:
            client.place_building(building, x, 10, rotation=0)
            print(f"  {building:<8} placed at ({x}, 10)")
        except Exception as ex:  # noqa: BLE001 - report whatever the API says
            print(f"  {building:<8} FAILED: {ex}")
            failed.append(building)

    if failed:
        print(f"\nlocked or rejected: {', '.join(failed)} - do not start a long run")
        return 1
    print("\nall buildings placeable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Expert-policy tests.

The point of the expert is to make the reward falsifiable, so these assert it
scores where random does not.
"""

import sys
import unittest

import numpy as np

from shapez_rl.env import ShapezBuildEnv
from shapez_rl.expert import (
    HUB_INPUT,
    ScriptedMiner,
    direction_to_rotation,
    find_route,
    plan_mining_line,
    run_episode,
)
from shapez_rl.fake_api import FakeShapezServer

BOUNDS = {"x": -16, "y": -16, "w": 32, "h": 32}


class RoutingTest(unittest.TestCase):
    def test_direction_to_rotation_matches_game_convention(self):
        # rotation is clockwise from up, y grows downward
        self.assertEqual(direction_to_rotation((0, 0), (0, -1)), 0)
        self.assertEqual(direction_to_rotation((0, 0), (1, 0)), 90)
        self.assertEqual(direction_to_rotation((0, 0), (0, 1)), 180)
        self.assertEqual(direction_to_rotation((0, 0), (-1, 0)), 270)

    def test_non_adjacent_tiles_are_rejected(self):
        with self.assertRaises(ValueError):
            direction_to_rotation((0, 0), (2, 0))

    def test_route_is_contiguous_and_ends_at_goal(self):
        route = find_route((-10, 5), HUB_INPUT, set(), BOUNDS)
        self.assertIsNotNone(route)
        self.assertEqual(route[0], (-10, 5))
        self.assertEqual(route[-1], HUB_INPUT)
        for a, b in zip(route, route[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_route_avoids_blocked_tiles(self):
        wall = {(-6, y) for y in range(-16, 16)}
        wall.discard((-6, 10))  # leave one gap
        route = find_route((-10, 5), HUB_INPUT, wall, BOUNDS)
        self.assertIsNotNone(route)
        self.assertTrue(set(route).isdisjoint(wall))
        self.assertIn((-6, 10), route)

    def test_route_returns_none_when_walled_off(self):
        wall = {(-6, y) for y in range(-16, 16)}
        self.assertIsNone(find_route((-10, 5), HUB_INPUT, wall, BOUNDS))


class PlanTest(unittest.TestCase):
    def test_plan_starts_with_miner_on_a_patch(self):
        payload = {
            "compact": True,
            "resources": [[-8, 6, "shape", "CuCuCuCu"]],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS)
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0][:3], ("miner", -8, 6))
        self.assertTrue(all(step[0] == "belt" for step in plan[1:]))

    def test_miner_faces_the_first_belt(self):
        """A miner ejects along its facing; pointed elsewhere it delivers nothing.

        Caught against the real game: rotation 0 scored 0, facing the belt scored 34.
        """
        payload = {
            "compact": True,
            "resources": [[-8, 6, "shape", "CuCuCuCu"]],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS)
        miner, first_belt = plan[0], plan[1]
        self.assertEqual(
            miner[3], direction_to_rotation((miner[1], miner[2]), (first_belt[1], first_belt[2]))
        )

    def test_plan_reaches_the_hub_input(self):
        payload = {
            "compact": True,
            "resources": [[-8, 6, "shape", "CuCuCuCu"]],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS)
        self.assertEqual((plan[-1][1], plan[-1][2]), HUB_INPUT)
        # The final belt must point into the hub, i.e. east.
        self.assertEqual(plan[-1][3], 90)

    def test_plan_never_overlaps_existing_buildings(self):
        payload = {
            "compact": True,
            "resources": [[-8, 6, "shape", "CuCuCuCu"]],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS)
        hub_tiles = {(x, y) for x in range(-2, 2) for y in range(-2, 2)}
        self.assertTrue({(s[1], s[2]) for s in plan}.isdisjoint(hub_tiles))

    def test_plan_prefers_the_nearest_patch(self):
        payload = {
            "compact": True,
            "resources": [
                [-14, 12, "shape", "CuCuCuCu"],
                [-5, 3, "shape", "CuCuCuCu"],
            ],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS)
        self.assertEqual((plan[0][1], plan[0][2]), (-5, 3))

    def test_target_shape_filters_patches(self):
        payload = {
            "compact": True,
            "resources": [[-5, 3, "shape", "CuCuCuCu"], [-9, 7, "shape", "RuRuRuRu"]],
            "buildings": [["hub", -2, -2, 4, 4, 0]],
        }
        plan = plan_mining_line(payload, BOUNDS, target_shape="RuRuRuRu")
        self.assertEqual((plan[0][1], plan[0][2]), (-9, 7))

    def test_no_patches_gives_no_plan(self):
        payload = {"compact": True, "resources": [], "buildings": []}
        self.assertIsNone(plan_mining_line(payload, BOUNDS))


class ExpertAgainstFakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeShapezServer().start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def make_env(self, **kwargs):
        kwargs.setdefault("placement_budget", 64)
        kwargs.setdefault("run_ticks", 3000)
        return ShapezBuildEnv(base_url=self.server.base_url, **kwargs)

    def test_expert_scores_and_random_does_not(self):
        """The whole point: a reward that cannot tell these apart is broken."""
        env = self.make_env()
        expert_reward, expert_info, planned = run_episode(env, seed=42)
        self.assertGreater(planned, 1, "expert planned nothing to build")
        self.assertGreater(expert_info["delivered"], 0)
        self.assertGreater(expert_reward, 0)

        rng = np.random.default_rng(0)
        env.reset(seed=42)
        terminated, random_reward = False, 0.0
        while not terminated:
            legal = np.flatnonzero(env.action_masks())
            action = np.unravel_index(int(rng.choice(legal)), env.action_space.nvec)
            _obs, random_reward, terminated, _t, _info = env.step(action)

        self.assertGreater(expert_reward, random_reward)

    def test_expert_is_reproducible_on_a_seed(self):
        env = self.make_env()
        first, info_first, _ = run_episode(env, seed=7)
        second, info_second, _ = run_episode(env, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(info_first["delivered"], info_second["delivered"])

    def test_expert_placements_are_accepted(self):
        env = self.make_env(placement_budget=40)
        _reward, info, planned = run_episode(env, seed=42)
        self.assertGreater(planned, 1)
        self.assertNotIn("placement-blocked", info["placement_failures"])
        self.assertNotIn("invalid-variant", info["placement_failures"])

    def test_filler_keeps_away_from_the_line(self):
        """Filler placed next to the line would re-curve it and break the layout."""
        env = self.make_env(placement_budget=64)
        env.reset(seed=42)
        expert = ScriptedMiner(env)
        planned = expert.reset(env._map_cache)
        for _ in range(planned):
            env.step(expert.act())

        filler = expert.act()
        tile = (env.bounds["x"] + int(filler[1]), env.bounds["y"] + int(filler[2]))
        nearest = min(
            abs(tile[0] - rx) + abs(tile[1] - ry) for rx, ry in expert._route
        )
        self.assertGreater(nearest, 1, "filler landed adjacent to the mining line")


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())

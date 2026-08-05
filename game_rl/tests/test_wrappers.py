#!/usr/bin/env python3
"""Flat-action wrapper tests.

The index mapping is the easiest thing here to get subtly wrong, and a wrong
mapping does not crash - it just trains badly. So these use a *non-square* window,
where a col/row transposition actually shows up.
"""

import sys
import unittest

import numpy as np

from shapez_rl.env import ShapezBuildEnv
from shapez_rl.expert import plan_to_actions
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.wrappers import FlatAction

# Deliberately not square: 24 wide, 16 tall.
BOUNDS = {"x": -12, "y": -8, "w": 24, "h": 16}


class FlatActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeShapezServer().start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def make_env(self, **kwargs):
        kwargs.setdefault("placement_budget", 4)
        kwargs.setdefault("run_ticks", 600)
        kwargs.setdefault("bounds", BOUNDS)
        return FlatAction(ShapezBuildEnv(base_url=self.server.base_url, **kwargs))

    def test_action_space_size_is_the_product(self):
        env = self.make_env()
        self.assertEqual(env.action_space.n, len(env.env.buildings) * 24 * 16 * 4)

    def test_roundtrip_is_identity_on_a_non_square_window(self):
        env = self.make_env()
        nvec = env.nvec
        rng = np.random.default_rng(0)
        for _ in range(200):
            multi = [int(rng.integers(n)) for n in nvec]
            flat = env.flatten_action(multi)
            np.testing.assert_array_equal(env.action(flat), np.array(multi))

    def test_mask_index_matches_action_index(self):
        """Mask entry i must correspond to flat action i, or masking is meaningless."""
        env = self.make_env()
        env.reset(seed=5)
        mask = env.action_masks()
        occupied = env.env.encoder.occupancy(env.env._map_cache)

        rng = np.random.default_rng(1)
        checked = 0
        for flat in rng.integers(0, env.action_space.n, size=400):
            building, col, row, _rot = env.action(int(flat))
            self.assertEqual(bool(mask[int(flat)]), not bool(occupied[row, col]))
            checked += 1
        self.assertEqual(checked, 400)

    def test_mask_length_matches_action_space(self):
        env = self.make_env()
        env.reset(seed=5)
        self.assertEqual(env.action_masks().shape, (env.action_space.n,))

    def test_flat_actions_place_where_intended(self):
        env = self.make_env(placement_budget=1)
        env.reset(seed=11)
        belt = env.env.buildings.index("belt")
        # col 20, row 3 -> world (8, -5); asymmetric so a transposition would miss.
        flat = env.flatten_action([belt, 20, 3, 0])
        env.step(flat)
        occupied = env.env.encoder.occupancy(env.env._map_cache)
        self.assertTrue(occupied[3, 20])

    def test_expert_actions_survive_flattening(self):
        env = self.make_env(placement_budget=32)
        env.reset(seed=42)
        from shapez_rl.expert import plan_mining_line

        plan = plan_mining_line(env.env._map_cache, env.env.bounds)
        if plan is None:
            self.skipTest("no routable patch in this window")
        for multi in plan_to_actions(plan, env.env):
            flat = env.flatten_action(multi)
            np.testing.assert_array_equal(env.action(flat), multi)


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())

import hashlib
import sys
import unittest

import numpy as np

from shapez_rl.client import RlApiError, ShapezClient
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.fake_api import FakeShapezServer


def fingerprint(observation):
    return hashlib.sha256(observation.tobytes()).hexdigest()[:16]


class EnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeShapezServer().start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def make_env(self, **kwargs):
        kwargs.setdefault("placement_budget", 8)
        kwargs.setdefault("run_ticks", 600)
        return ShapezBuildEnv(base_url=self.server.base_url, **kwargs)

    def test_observation_matches_declared_space(self):
        env = self.make_env()
        obs, _info = env.reset(seed=1)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(obs))

    def test_action_space_covers_window_and_buildings(self):
        env = self.make_env()
        self.assertEqual(
            list(env.action_space.nvec), [len(env.buildings), 32, 32, 4]
        )

    def test_same_seed_reproduces_observation(self):
        env = self.make_env()
        a, info_a = env.reset(seed=42)
        b, info_b = env.reset(seed=42)
        self.assertEqual(info_a["map_seed"], 42)
        self.assertEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(info_a["map_seed"], info_b["map_seed"])

    def test_different_seed_changes_observation(self):
        env = self.make_env()
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=99)
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_reset_clears_previous_episode(self):
        env = self.make_env(placement_budget=3)
        env.reset(seed=7)
        before = env.encoder.occupancy(env._map_cache).sum()
        for _ in range(3):
            env.step(self._legal_action(env))
        env.reset(seed=7)
        self.assertEqual(env.encoder.occupancy(env._map_cache).sum(), before)
        self.assertEqual(env._placements_used, 0)
        self.assertEqual(env._placed_ok, 0)

    def _legal_action(self, env, rng=None):
        rng = rng or np.random.default_rng(0)
        legal = np.flatnonzero(env.action_masks())
        return np.unravel_index(int(rng.choice(legal)), env.action_space.nvec)

    def test_episode_terminates_exactly_at_budget(self):
        env = self.make_env(placement_budget=5)
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        steps, terminated = 0, False
        while not terminated:
            _obs, _r, terminated, truncated, _info = env.step(self._legal_action(env, rng))
            steps += 1
            self.assertFalse(truncated)
            self.assertLessEqual(steps, 5)
        self.assertEqual(steps, 5)

    def test_reward_is_zero_until_final_step(self):
        env = self.make_env(placement_budget=4)
        env.reset(seed=3)
        rng = np.random.default_rng(0)
        for i in range(4):
            _obs, reward, terminated, _t, _info = env.step(self._legal_action(env, rng))
            if not terminated:
                self.assertEqual(reward, 0.0, f"step {i} paid out before the run phase")

    def test_masked_actions_avoid_occupied_tiles(self):
        env = self.make_env()
        env.reset(seed=5)
        occupied = env.encoder.occupancy(env._map_cache)
        mask = env.action_masks().reshape(
            len(env.buildings), env.bounds["w"], env.bounds["h"], 4
        )
        # The hub occupies tiles; every action targeting one must be masked out.
        rows, cols = np.nonzero(occupied)
        for row, col in zip(rows[:20], cols[:20]):
            self.assertFalse(mask[:, col, row, :].any())

    def test_placements_land_where_the_action_says(self):
        env = self.make_env(placement_budget=1)
        env.reset(seed=11)
        # belt at window-local (20, 20) -> world (4, 4)
        belt = env.buildings.index("belt")
        env.step(np.array([belt, 20, 20, 0]))
        occupied = env.encoder.occupancy(env._map_cache)
        self.assertTrue(occupied[20, 20])

    def test_failed_placement_is_recorded_not_raised(self):
        """Only the hub blocks. Verified against the real API on 2026-08-02."""
        env = self.make_env(placement_budget=1)
        env.reset(seed=13)
        belt = env.buildings.index("belt")
        # Window-local (14, 14) -> world (-2, -2), the hub's corner.
        _obs, _r, _term, _t, info = env.step(np.array([belt, 14, 14, 0]))
        self.assertEqual(info["placements_succeeded"], 0)
        self.assertIn("placement-blocked", info["placement_failures"])

    def test_placement_replaces_removable_buildings(self):
        """Occupied tiles are not blocked - the real API replaces, issuing a new uid."""
        client = ShapezClient(self.server.base_url)
        client.reset(seed=13)
        first = client.place_building("belt", 8, 8, rotation=0)
        second = client.place_building("miner", 8, 8, rotation=0, variant="chainable")
        self.assertNotEqual(first["entityUid"], second["entityUid"])

        window = client.map_window(7, 7, 3, 3, compact=True)
        at_tile = [row for row in window["buildings"] if (row[1], row[2]) == (8, 8)]
        self.assertEqual(len(at_tile), 1)
        self.assertEqual(at_tile[0][0], "miner")

    def test_miner_uses_catalogue_variant_not_default(self):
        """The default-variant trap: an unlocked miner only offers `chainable`."""
        env = self.make_env()
        env.reset(seed=17)
        self.assertEqual(env._variant_for["miner"], "chainable")

        client = ShapezClient(self.server.base_url)
        with self.assertRaises(RlApiError) as caught:
            client.place_building("miner", 12, 12, rotation=0, variant="default")
        self.assertEqual(caught.exception.error, "invalid-variant")


    def test_reward_is_the_delta_over_the_reset_baseline(self):
        """Reward arithmetic only. End-to-end delivery is covered in test_expert."""
        env = self.make_env(placement_budget=1, run_ticks=600)
        env.reset(seed=42)
        self.server.game.stored_shapes["CuCuCuCu"] = 7

        _obs, reward, terminated, _t, info = env.step(np.array([0, 0, 0, 0]))
        self.assertTrue(terminated)
        self.assertEqual(info["delivered"], 7)
        self.assertEqual(reward, 7.0)

    def test_baseline_is_resnapshotted_each_episode(self):
        """A stale baseline would carry the previous episode's score into this one."""
        env = self.make_env(placement_budget=1, run_ticks=600)

        env.reset(seed=42)
        self.server.game.stored_shapes["CuCuCuCu"] = 5
        _obs, _reward, _term, _t, first = env.step(np.array([0, 0, 0, 0]))
        self.assertEqual(first["delivered"], 5)

        env.reset(seed=42)
        self.server.game.stored_shapes["CuCuCuCu"] = 2
        _obs, reward, _term, _t, second = env.step(np.array([0, 0, 0, 0]))
        self.assertEqual(second["delivered"], 2)
        self.assertEqual(reward, 2.0)

    def test_target_shape_scores_only_that_shape(self):
        env = self.make_env(target_shape="ZzZzZzZz", placement_budget=1)
        env.reset(seed=42)
        self.server.game.stored_shapes["CuCuCuCu"] = 99
        _obs, reward, _term, _t, info = env.step(np.array([0, 0, 0, 0]))
        self.assertEqual(info["delivered"], 0)
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())

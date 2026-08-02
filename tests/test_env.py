import hashlib
import sys
import unittest

import numpy as np

from shapez_rl.client import RlApiError, ShapezClient
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.fake_api import HUB_INPUT, FakeShapezServer


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
        env = self.make_env(placement_budget=2)
        env.reset(seed=13)
        belt = env.buildings.index("belt")
        # Same tile twice: the second must be rejected and counted, not thrown.
        env.step(np.array([belt, 20, 20, 0]))
        _obs, _r, _term, _t, info = env.step(np.array([belt, 20, 20, 0]))
        self.assertEqual(info["placements_succeeded"], 1)
        self.assertIn("placement-blocked", info["placement_failures"])

    def test_miner_uses_catalogue_variant_not_default(self):
        """The default-variant trap: an unlocked miner only offers `chainable`."""
        env = self.make_env()
        env.reset(seed=17)
        self.assertEqual(env._variant_for["miner"], "chainable")

        client = ShapezClient(self.server.base_url)
        with self.assertRaises(RlApiError) as caught:
            client.place_building("miner", 12, 12, rotation=0, variant="default")
        self.assertEqual(caught.exception.error, "invalid-variant")


    def test_reward_tracks_delivered_shapes(self):
        """A miner on a shape patch belted to the hub should score."""
        env = self.make_env(placement_budget=1, run_ticks=3000)
        env.reset(seed=42)

        game = self.server.game
        patch = next(
            ((x, y) for (x, y), (kind, _key) in game.resources.items() if kind == "shape"),
            None,
        )
        self.assertIsNotNone(patch, "seed 42 should contain a shape patch")

        client = ShapezClient(self.server.base_url)
        client.place_building("miner", patch[0], patch[1], rotation=0, variant="chainable")
        # Straight belt run from the patch to the hub input.
        for x in range(min(patch[0], HUB_INPUT[0]) + 1, max(patch[0], HUB_INPUT[0]) + 1):
            try:
                client.place_building("belt", x, patch[1], rotation=90)
            except RlApiError:
                pass
        for y in range(min(patch[1], HUB_INPUT[1]), max(patch[1], HUB_INPUT[1]) + 1):
            try:
                client.place_building("belt", HUB_INPUT[0], y, rotation=0)
            except RlApiError:
                pass

        env._map_cache = env._fetch_map()
        _obs, reward, terminated, _t, info = env.step(np.array([0, 0, 0, 0]))
        self.assertTrue(terminated)
        self.assertGreater(info["delivered"], 0, "belted miner delivered nothing")
        self.assertGreater(reward, 0)

    def test_target_shape_scores_only_that_shape(self):
        env = self.make_env(target_shape="ZzZzZzZz", placement_budget=1)
        env.reset(seed=42)
        self.server.game.stored_shapes["CuCuCuCu"] = 99
        _obs, reward, _term, _t, info = env.step(np.array([0, 0, 0, 0]))
        self.assertEqual(info["delivered"], 0)
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())

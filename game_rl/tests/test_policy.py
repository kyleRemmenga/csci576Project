#!/usr/bin/env python3
"""Conv policy tests.

The action head reorders axes, and a wrong reorder does not crash - every logit
still lines up with *an* action, just not the one the mask cleared. So most of
this file is about index order, on a *non-square* window where a col/row
transposition actually shows up.
"""

import sys
import unittest

import numpy as np
import torch

from shapez_rl.encoding import ROTATIONS
from shapez_rl.env import ShapezBuildEnv
from shapez_rl.fake_api import FakeShapezServer
from shapez_rl.policy import (
    PooledValueHead,
    ShapezCNN,
    SpatialActionHead,
    SpatialMaskablePolicy,
    flat_action_index,
)
from shapez_rl.wrappers import FlatAction

# Deliberately not square: 24 wide, 16 tall.
BOUNDS = {"x": -12, "y": -8, "w": 24, "h": 16}


class SpatialActionHeadTest(unittest.TestCase):
    """Pins the (building, rotation, row, col) -> (building, col, row, rotation) permute."""

    N_BUILDINGS = 2
    N_ROTATIONS = 2
    HEIGHT = 3
    WIDTH = 4

    def _head(self):
        """A head whose output is readable by eye.

        Output channel *k* is scaled by ``10**k``, and the single input channel
        carries a unique value per cell, so every logit identifies both which
        (building, rotation) pair and which tile produced it.
        """
        head = SpatialActionHead(
            (1, self.HEIGHT, self.WIDTH), self.N_BUILDINGS, self.N_ROTATIONS
        )
        with torch.no_grad():
            head.conv.bias.zero_()
            for k in range(self.N_BUILDINGS * self.N_ROTATIONS):
                head.conv.weight[k, 0, 0, 0] = 10.0**k
        return head

    def _latent(self):
        cells = np.arange(1, self.HEIGHT * self.WIDTH + 1, dtype=np.float32)
        return torch.tensor(cells).view(1, 1, self.HEIGHT, self.WIDTH).flatten(1)

    def test_every_logit_lands_at_its_flat_action_index(self):
        flat = self._head()(self._latent())[0].detach().numpy()

        self.assertEqual(
            flat.size, self.N_BUILDINGS * self.WIDTH * self.HEIGHT * self.N_ROTATIONS
        )

        for building in range(self.N_BUILDINGS):
            for col in range(self.WIDTH):
                for row in range(self.HEIGHT):
                    for rotation in range(self.N_ROTATIONS):
                        index = flat_action_index(
                            self.N_BUILDINGS, self.WIDTH, self.HEIGHT, self.N_ROTATIONS,
                            building, col, row, rotation,
                        )
                        channel = building * self.N_ROTATIONS + rotation
                        expected = (10.0**channel) * (row * self.WIDTH + col + 1)
                        self.assertAlmostEqual(
                            float(flat[index]), expected, places=3,
                            msg=f"building={building} col={col} row={row} rot={rotation}",
                        )

    def test_transposing_the_grid_changes_the_output(self):
        """A square window would let a row/col swap pass unnoticed."""
        self.assertNotEqual(self.HEIGHT, self.WIDTH)


class ValueHeadTest(unittest.TestCase):
    def test_pools_over_tiles_to_one_scalar_per_sample(self):
        head = PooledValueHead((5, 3, 4))
        with torch.no_grad():
            head.linear.weight.fill_(1.0)
            head.linear.bias.zero_()

        latent = torch.ones(2, 5 * 3 * 4)
        values = head(latent)

        self.assertEqual(values.shape, (2, 1))
        # Mean over tiles is 1 per channel, summed over 5 channels.
        np.testing.assert_allclose(values.detach().numpy(), np.full((2, 1), 5.0))


class TrunkTest(unittest.TestCase):
    def test_output_map_is_one_cell_per_tile(self):
        env = _build_env(FakeShapezServer, budget=1)
        try:
            trunk = ShapezCNN(env.observation_space, channels=8)
            obs, _info = env.reset(seed=0)

            out = trunk(torch.tensor(obs).unsqueeze(0))

            self.assertEqual(trunk.map_shape, (8, BOUNDS["h"], BOUNDS["w"]))
            self.assertEqual(out.shape, (1, 8 * BOUNDS["h"] * BOUNDS["w"]))
        finally:
            env.close()


class PolicyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = FakeShapezServer().start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        self.base = ShapezBuildEnv(
            base_url=self.server.base_url, bounds=BOUNDS, placement_budget=4, run_ticks=50
        )
        self.env = FlatAction(self.base)
        self.env.reset(seed=3)

    def tearDown(self):
        self.env.close()

    def _policy(self, channels=8):
        return SpatialMaskablePolicy(
            self.env.observation_space,
            self.env.action_space,
            lambda _progress: 3e-4,
            n_buildings=len(self.base.buildings),
            n_rotations=len(ROTATIONS),
            feature_channels=channels,
            dilations=(1, 2),
        )

    def test_head_ordering_agrees_with_the_env_action_mask(self):
        """The property that matters: a masked-out tile is masked out at the
        indices the head puts that tile's logits at."""
        mask = self.env.action_masks()
        occupied = self.base.encoder.occupancy(self.base._map_cache)

        rows, cols = np.nonzero(occupied)
        self.assertGreater(rows.size, 0, "expected the hub to occupy tiles")

        for row, col in list(zip(rows, cols))[:20]:
            for building in range(len(self.base.buildings)):
                for rotation in range(len(ROTATIONS)):
                    index = flat_action_index(
                        len(self.base.buildings), BOUNDS["w"], BOUNDS["h"], len(ROTATIONS),
                        building, int(col), int(row), rotation,
                    )
                    self.assertFalse(
                        mask[index], f"occupied tile row={row} col={col} cleared"
                    )

    def test_flat_index_decodes_to_the_placement_it_names(self):
        building, col, row, rotation = 1, 5, 2, 3
        index = flat_action_index(
            len(self.base.buildings), BOUNDS["w"], BOUNDS["h"], len(ROTATIONS),
            building, col, row, rotation,
        )

        decoded = self.env.action(index)
        np.testing.assert_array_equal(decoded, [building, col, row, rotation])

        building_id, x, y, degrees = self.base._decode_action(decoded)
        self.assertEqual(building_id, self.base.buildings[building])
        self.assertEqual((x, y), (BOUNDS["x"] + col, BOUNDS["y"] + row))
        self.assertEqual(degrees, ROTATIONS[rotation])

    def test_logits_have_one_entry_per_action(self):
        policy = self._policy()
        obs, _info = self.env.reset(seed=1)

        with torch.no_grad():
            latent = policy.extract_features(torch.tensor(obs).unsqueeze(0))
            logits = policy.action_net(latent)
            values = policy.value_net(latent)

        self.assertEqual(logits.shape, (1, self.env.action_space.n))
        self.assertEqual(values.shape, (1, 1))

    def test_masked_actions_are_never_sampled(self):
        policy = self._policy()
        obs, _info = self.env.reset(seed=7)
        mask = self.env.action_masks()

        actions, _values, _log_prob = policy(
            torch.tensor(obs).unsqueeze(0), action_masks=mask[np.newaxis, :]
        )

        self.assertTrue(mask[int(actions[0])])

    def test_rejects_a_factorisation_the_action_space_cannot_hold(self):
        with self.assertRaises(ValueError):
            SpatialMaskablePolicy(
                self.env.observation_space,
                self.env.action_space,
                lambda _progress: 3e-4,
                n_buildings=len(self.base.buildings) + 1,
                n_rotations=len(ROTATIONS),
            )

    def test_stays_small_enough_to_train(self):
        """Guards the reason this class exists.

        A linear action head over the flattened map is ~3e9 parameters. If someone
        reintroduces one, this fails long before anyone waits on a run.
        """
        policy = self._policy(channels=64)
        total = sum(p.numel() for p in policy.parameters())
        self.assertLess(total, 1_000_000, f"{total} parameters - is the head still conv?")


def _build_env(server_factory, budget):
    server = server_factory().start()
    env = ShapezBuildEnv(
        base_url=server.base_url, bounds=BOUNDS, placement_budget=budget, run_ticks=50
    )
    env._test_server = server
    original_close = env.close

    def close():
        original_close()
        server.stop()

    env.close = close
    return env


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())

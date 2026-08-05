"""Gymnasium environment for shapez."""

import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import RlApiError, ShapezClient, hub_level, map_seed, stored_shapes
from .encoding import DEFAULT_BUILDINGS, ROTATIONS, MapEncoder

DEFAULT_BOUNDS = {"x": -16, "y": -16, "w": 32, "h": 32}


# Delete this once the catalogue endpoint exists.
FALLBACK_VARIANTS = {"miner": "chainable"}


class ShapezBuildEnv(gym.Env):
    """Place a budget of buildings, then run the factory and score it."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url,
        bounds=None,
        placement_budget=64,
        run_ticks=3000,
        buildings=DEFAULT_BUILDINGS,
        target_shape=None,
        delivered_reward=1.0,
        placement_penalty=0.0,
        invalid_action_penalty=0.0,
        timeout=30.0,
    ):
        super().__init__()

        self.bounds = dict(bounds or DEFAULT_BOUNDS)
        self.placement_budget = placement_budget
        self.run_ticks = run_ticks
        self.target_shape = target_shape
        self.delivered_reward = delivered_reward
        self.placement_penalty = placement_penalty
        self.invalid_action_penalty = invalid_action_penalty

        self.encoder = MapEncoder(self.bounds, buildings=buildings)
        self.buildings = tuple(buildings)
        self.client = ShapezClient(base_url, timeout=timeout)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.encoder.shape, dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [len(self.buildings), self.bounds["w"], self.bounds["h"], len(ROTATIONS)]
        )

        self._map_cache = None
        self._placements_used = 0
        self._placed_ok = 0
        self._failures = {}
        self._baseline = {}
        self._variant_for = {}
        self._warned = set()

    def _warn_once(self, key, message):
        if key not in self._warned:
            self._warned.add(key)
            print(f"warning: {message}", file=sys.stderr)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        map_seed_request = None if seed is None else int(seed) % (2**32)
        state = self._reset_episode(map_seed_request)

        self._placements_used = 0
        self._placed_ok = 0
        self._failures = {}
        self._baseline = dict(stored_shapes(state))
        self._refresh_variants()
        self._map_cache = self._fetch_map()

        return self.encoder.encode(self._map_cache), self._info(state)

    def step(self, action):
        building_id, x, y, rotation = self._decode_action(action)

        reward = 0.0
        placed = self._try_place(building_id, x, y, rotation)
        if placed:
            self._placed_ok += 1
            # Refetched rather than patched locally: placing a belt changes the
            # rotation variant of its neighbours, so a patched cache would desync.
            self._map_cache = self._fetch_map()
            reward -= self.placement_penalty
        else:
            reward -= self.invalid_action_penalty

        self._placements_used += 1
        terminated = self._placements_used >= self.placement_budget

        if terminated:
            state = self.client.tick(self.run_ticks)
            reward += self.delivered_reward * self._delivered(state)
            info = self._info(state)
        else:
            info = {"placed": placed}

        return self.encoder.encode(self._map_cache), reward, terminated, False, info

    def close(self):
        pass

    def action_masks(self):
        """Boolean mask over the flattened (type, col, row, rotation) action space."""
        free = ~self.encoder.occupancy(self._map_cache)
        per_tile = free.T[np.newaxis, :, :, np.newaxis]
        mask = np.broadcast_to(
            per_tile,
            (len(self.buildings), self.bounds["w"], self.bounds["h"], len(ROTATIONS)),
        )
        return mask.reshape(-1).copy()

    def _decode_action(self, action):
        building_index, col, row, rotation_index = (int(v) for v in np.asarray(action).ravel())
        return (
            self.buildings[building_index],
            self.bounds["x"] + col,
            self.bounds["y"] + row,
            ROTATIONS[rotation_index],
        )

    def _reset_episode(self, map_seed_request):
        """Start a fresh episode, degrading if /rl/reset is missing."""
        try:
            return self.client.reset(seed=map_seed_request)
        except RlApiError as ex:
            if ex.status != 404:
                raise

        self._warn_once(
            "reset",
            "POST /rl/reset is missing; falling back to destroy-removable-buildings. "
            "hubGoals and the map do NOT reset and the seed is ignored - single "
            "episodes only, do not train against this.",
        )
        self.client.destroy_removable_buildings()
        return self.client.gamestate()

    def _refresh_variants(self):
        """Learn each building's legal variant, degrading if the catalogue is missing."""
        try:
            catalogue = self.client.buildings()
        except RlApiError as ex:
            if ex.status != 404:
                raise
            self._warn_once(
                "buildings",
                "GET /rl/buildings is missing; falling back to "
                f"{FALLBACK_VARIANTS}. Other buildings will use their default "
                "variant, which may be rejected.",
            )
            self._variant_for = dict(FALLBACK_VARIANTS)
            return

        self._variant_for = {
            entry["id"]: entry["variants"][0]["variant"]
            for entry in catalogue["buildings"]
            if entry["placeable"] and entry["variants"]
        }

    def _try_place(self, building_id, x, y, rotation):
        try:
            self.client.place_building(
                building_id, x, y,
                rotation=rotation,
                variant=self._variant_for.get(building_id),
            )
            return True
        except RlApiError as ex:
            self._failures[ex.error] = self._failures.get(ex.error, 0) + 1
            return False

    def _delivered(self, state):
        current = stored_shapes(state)
        if self.target_shape is not None:
            return max(0, current.get(self.target_shape, 0)
                       - self._baseline.get(self.target_shape, 0))
        return sum(
            max(0, count - self._baseline.get(key, 0)) for key, count in current.items()
        )

    def _fetch_map(self):
        return self.client.map_window(
            self.bounds["x"], self.bounds["y"], self.bounds["w"], self.bounds["h"]
        )

    def _info(self, state):
        return {
            "map_seed": map_seed(state),
            "hub_level": hub_level(state),
            "delivered": self._delivered(state),
            "placements_attempted": self._placements_used,
            "placements_succeeded": self._placed_ok,
            "placement_failures": dict(self._failures),
        }

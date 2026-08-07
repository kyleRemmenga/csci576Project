"""Gymnasium environment for shapez."""

import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import RlApiError, ShapezClient, hub_level, map_seed, stored_shapes
from .encoding import DEFAULT_BUILDINGS, ROTATIONS, MapEncoder, iter_resources
from .expert import HUB_INPUT

DEFAULT_BOUNDS = {"x": -16, "y": -16, "w": 32, "h": 32}


# Delete this once the catalogue endpoint exists.
FALLBACK_VARIANTS = {"miner": "chainable"}


def _dump(payload):
    savegame = payload.get("savegame")
    return savegame.get("dump", {}) if savegame else {}


def belt_item_keys(payload):
    """Distinct item keys riding on belts right now.
    """
    keys = set()
    for path in _dump(payload).get("beltPaths", []):
        for entry in path.get("items", []):
            item = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
            if isinstance(item, dict) and item.get("data"):
                keys.add(item["data"])
    return keys


def hub_goal_shape(payload):
    """Shape key the hub currently wants, read off its wire pin."""
    for entity in _dump(payload).get("entities", []):
        pins = entity.get("components", {}).get("WiredPins")
        if not pins:
            continue
        for slot in pins.get("slots", []):
            value = slot.get("value") or {}
            if value.get("$") == "shape" and value.get("data"):
                return value["data"]
    return None


def _entity_origins(dump):
    origins = {}
    for entity in dump.get("entities", []):
        static = entity.get("components", {}).get("StaticMapEntity", {})
        origin = static.get("origin")
        if origin is not None:
            origins[entity.get("uid")] = (origin["x"], origin["y"])
    return origins


def _path_item_keys(path):
    keys = set()
    for entry in path.get("items", []):
        item = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
        if isinstance(item, dict) and item.get("data"):
            keys.add(item["data"])
    return keys


def source_shapes(shape_key):
    """Raw mined shapes a goal shape is processed from.

    A goal like ``----CuCu`` is never mined directly; it comes out of a cutter fed
    with ``CuCuCuCu``. Without this the mining and routing bonuses never fire for
    processed goals and the reward collapses to all-or-nothing.
    """
    if not shape_key or len(shape_key) != 8:
        return set()
    quadrants = [shape_key[i:i + 2] for i in range(0, 8, 2)]
    filled = [q for q in quadrants if q != "--"]
    if not filled or len(filled) == 4:
        return set()
    return {filled[0] * 4}


def waste_shapes(shape_key):
    """The offcut a cutter produces alongside a half-shape goal.

    A cutter emits both halves and stalls if either output backs up, so the
    offcut has to be belted away. Seeing it move is the signal that the agent
    solved the jam rather than getting one lucky delivery.
    """
    if not shape_key or len(shape_key) != 8:
        return set()
    quadrants = [shape_key[i:i + 2] for i in range(0, 8, 2)]
    filled = [q for q in quadrants if q != "--"]
    if not filled or len(filled) == 4:
        return set()
    return {"".join("--" if q != "--" else filled[0] for q in quadrants)}


def belt_progress(payload, wanted):
    """How far the best belt run carrying a wanted item gets toward the hub, 0..1.

    ``entityPath`` is ordered along the flow, so its first tile is where items
    enter and its last is where they leave. Scoring the fraction of that gap
    closed means an unfinished chain still earns credit, while belts carrying
    nothing wanted are skipped so paving the map earns nothing.
    """
    if not wanted:
        return 0.0

    dump = _dump(payload)
    origins = _entity_origins(dump)
    best = 0.0

    for path in dump.get("beltPaths", []):
        entity_path = path.get("entityPath") or []
        if not entity_path or not (_path_item_keys(path) & wanted):
            continue
        source = origins.get(entity_path[0])
        end = origins.get(entity_path[-1])
        if source is None or end is None:
            continue
        span = abs(source[0] - HUB_INPUT[0]) + abs(source[1] - HUB_INPUT[1])
        if span <= 0:
            continue
        remaining = abs(end[0] - HUB_INPUT[0]) + abs(end[1] - HUB_INPUT[1])
        best = max(best, (span - remaining) / span)

    return min(best, 1.0)


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
        goal_level=None,
        delivered_reward=1.0,
        production_bonus=5.0,
        route_bonus=5.0,
        waste_bonus=2.0,
        source_route_scale=0.4,
        mining_bonus=1.0,
        placement_penalty=0.0,
        invalid_action_penalty=0.0,
        timeout=30.0,
    ):
        super().__init__()

        self.bounds = dict(bounds or DEFAULT_BOUNDS)
        self.placement_budget = placement_budget
        self.run_ticks = run_ticks
        self.target_shape = target_shape
        self.goal_level = goal_level
        self.delivered_reward = delivered_reward
        self.production_bonus = production_bonus
        self.route_bonus = route_bonus
        self.waste_bonus = waste_bonus
        self.source_route_scale = source_route_scale
        self.mining_bonus = mining_bonus
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
        self._wanted = set()
        self._sources = set()
        self._waste = set()
        self._produced = set()
        self._mined = False
        self._progress = 0.0
        self._waste_routed = False

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
        self._produced = set()
        self._wanted = self._goal_keys(state)
        self._sources = set()
        self._waste = set()
        for key in self._wanted:
            self._sources |= source_shapes(key)
            self._waste |= waste_shapes(key)
        self._mined = False
        self._progress = 0.0
        self._waste_routed = False
        self._refresh_variants()
        self._map_cache = self._fetch_map()

        return self.encoder.encode(self._map_cache), self._info(state)

    def step(self, action):
        building_id, x, y, rotation = self._decode_action(action)

        reward = 0.0
        placed = self._try_place(building_id, x, y, rotation)
        if placed:
            self._placed_ok += 1
            if not self._mined and self._mines_wanted(building_id, x, y):
                self._mined = True
                reward += self.mining_bonus
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
            reward += self.production_bonus * self._newly_produced(state)
            reward += self.route_bonus * self._route_credit(state)
            if self._waste and (belt_item_keys(state) & self._waste):
                self._waste_routed = True
                reward += self.waste_bonus
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
            return self.client.reset(seed=map_seed_request, goal_level=self.goal_level)
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

    def _goal_keys(self, state):
        """The shape that counts as progress this episode."""
        key = self.target_shape if self.target_shape is not None else hub_goal_shape(state)
        return {key} if key else set()

    def _mines_wanted(self, building_id, x, y):
        """True if this is a miner sitting on a tile of the goal resource.
        """
        useful = self._wanted | self._sources
        if building_id != "miner" or not useful:
            return False
        for rx, ry, _kind, key in iter_resources(self._map_cache):
            if rx == x and ry == y:
                return key in useful
        return False

    def _route_credit(self, state):
        """Routing score, discounted while the belt still carries raw material.

        Hauling uncut shapes to the hub is a dead end, so it earns only a
        fraction: enough to find the hub, not enough to beat cutting.
        """
        self._progress = belt_progress(state, self._wanted)
        if not self._sources:
            return self._progress
        raw = belt_progress(state, self._sources)
        return max(self._progress, self.source_route_scale * raw)

    def _newly_produced(self, state):
        """Wanted item keys seen on belts for the first time this episode.
        """
        if not self._wanted:
            return 0
        fresh = (belt_item_keys(state) & self._wanted) - self._produced
        self._produced |= fresh
        return len(fresh)

    def _info(self, state):
        return {
            "map_seed": map_seed(state),
            "hub_level": hub_level(state),
            "delivered": self._delivered(state),
            "mined": self._mined,
            "produced": sorted(self._produced),
            "waste_routed": self._waste_routed,
            "progress": round(self._progress, 3),
            "wanted": sorted(self._wanted),
            "placements_attempted": self._placements_used,
            "placements_succeeded": self._placed_ok,
            "placement_failures": dict(self._failures),
        }

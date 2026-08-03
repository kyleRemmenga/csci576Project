"""A scripted expert: mine one shape patch and belt it to the hub."""

from collections import deque

import numpy as np

from .encoding import ROTATIONS, iter_buildings, iter_resources

# The hub spans (-2,-2)..(1,1). Its left input is the free tile just outside that
# edge; a belt there pointing east feeds the hub.
HUB_INPUT = (-3, 1)
HUB_TILE = (-2, 1)

# rotation is degrees clockwise from up, and y grows downward.
DIRECTION_TO_ROTATION = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}

MAX_CANDIDATES = 12


def direction_to_rotation(a, b):
    """Rotation for a belt at ``a`` feeding ``b``. Adjacent tiles only."""
    step = (b[0] - a[0], b[1] - a[1])
    if step not in DIRECTION_TO_ROTATION:
        raise ValueError(f"{a} and {b} are not orthogonally adjacent")
    return DIRECTION_TO_ROTATION[step]


def occupied_tiles(map_payload):
    tiles = set()
    for _id, bx, by, bw, bh, _rot in iter_buildings(map_payload):
        if bx is None or by is None:
            continue
        for x in range(bx, bx + bw):
            for y in range(by, by + bh):
                tiles.add((x, y))
    return tiles


def shape_patches(map_payload, key=None):
    return [
        (x, y)
        for x, y, kind, resource_key in iter_resources(map_payload)
        if kind == "shape" and (key is None or resource_key == key)
    ]


def find_route(start, goal, blocked, bounds, max_nodes=20000):
    """Shortest tile path from ``start`` to ``goal`` avoiding ``blocked``."""
    if start == goal:
        return [start]

    min_x, min_y = bounds["x"], bounds["y"]
    max_x, max_y = bounds["x"] + bounds["w"] - 1, bounds["y"] + bounds["h"] - 1

    frontier = deque([start])
    came_from = {start: None}
    visited = 0

    while frontier and visited < max_nodes:
        current = frontier.popleft()
        visited += 1

        for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + step[0], current[1] + step[1])
            if nxt in came_from:
                continue
            if not (min_x <= nxt[0] <= max_x and min_y <= nxt[1] <= max_y):
                continue
            if nxt != goal and nxt in blocked:
                continue

            came_from[nxt] = current
            if nxt == goal:
                path = [nxt]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])
                return list(reversed(path))
            frontier.append(nxt)

    return None


def plan_mining_line(map_payload, bounds, target_shape=None):
    """Plan a miner plus a belt run from a shape patch to the hub input."""
    blocked = occupied_tiles(map_payload)
    patches = [p for p in shape_patches(map_payload, target_shape) if p not in blocked]
    if not patches:
        return None

    # Nearest first: shorter runs cost less budget and are less likely to be cut off.
    patches.sort(key=lambda p: abs(p[0] - HUB_INPUT[0]) + abs(p[1] - HUB_INPUT[1]))

    for patch in patches[:MAX_CANDIDATES]:
        route = find_route(patch, HUB_INPUT, blocked, bounds)
        if route is None or len(route) < 2:
            continue

        # The miner ejects in its facing direction, so it must point at the first
        # belt. Facing anywhere else mines into the void and delivers nothing
        plan = [("miner", patch[0], patch[1], direction_to_rotation(patch, route[1]))]
        # Each tile after the miner carries a belt pointing at the next tile; the
        # last belt points into the hub itself.
        for index in range(1, len(route)):
            tile = route[index]
            nxt = route[index + 1] if index + 1 < len(route) else HUB_TILE
            plan.append(("belt", tile[0], tile[1], direction_to_rotation(tile, nxt)))
        return plan

    return None


def plan_to_actions(plan, env):
    """Convert ``(building_id, x, y, rotation)`` tuples into env actions."""
    actions = []
    for building_id, x, y, rotation in plan:
        if building_id not in env.buildings:
            continue
        actions.append(
            np.array(
                [
                    env.buildings.index(building_id),
                    x - env.bounds["x"],
                    y - env.bounds["y"],
                    ROTATIONS.index(rotation),
                ]
            )
        )
    return actions


class ScriptedMiner:
    """Plans one mining line and replays it as env actions."""

    def __init__(self, env, target_shape=None):
        self.env = env
        self.target_shape = target_shape if target_shape is not None else env.target_shape
        self.plan = None
        self._queue = []
        self._route = set()

    def reset(self, map_payload):
        """Plan from the current map. Returns the number of planned placements."""
        self.plan = plan_mining_line(map_payload, self.env.bounds, self.target_shape)
        self._queue = plan_to_actions(self.plan, self.env) if self.plan else []
        self._route = {(x, y) for _id, x, y, _rot in (self.plan or [])}
        return len(self._queue)

    def act(self):
        """Next planned action, or harmless filler once the plan is done."""
        if self._queue:
            return self._queue.pop(0)
        return self._filler_action()

    def _filler_action(self):
        """A legal placement as far from the line as possible."""
        legal = np.flatnonzero(self.env.action_masks())
        if legal.size == 0:
            return self.env.action_space.sample()
        if not self._route:
            return np.unravel_index(int(legal[0]), self.env.action_space.nvec)

        free = ~self.env.encoder.occupancy(self.env._map_cache)
        best, best_distance = None, -1
        for row, col in zip(*np.nonzero(free)):
            tile = (self.env.bounds["x"] + col, self.env.bounds["y"] + row)
            distance = min(
                abs(tile[0] - rx) + abs(tile[1] - ry) for rx, ry in self._route
            )
            if distance > best_distance:
                best, best_distance = (int(col), int(row)), distance

        if best is None:
            return np.unravel_index(int(legal[0]), self.env.action_space.nvec)
        return np.array([self.env.buildings.index("belt"), best[0], best[1], 0])


def run_episode(env, seed=None, target_shape=None):
    """Reset, execute the plan, run the factory. Returns ``(reward, info, planned)``."""
    env.reset(seed=seed)
    expert = ScriptedMiner(env, target_shape=target_shape)
    planned = expert.reset(env._map_cache)

    reward, info, terminated = 0.0, {}, False
    while not terminated:
        _obs, reward, terminated, _truncated, info = env.step(expert.act())

    return reward, info, planned

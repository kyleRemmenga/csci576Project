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
MAX_CUTTER_SITES = 40

# Which of the cutter's two ejectors carries the goal half. Verified against the
# live game rather than read off the source, since the cut order is not obvious.
GOAL_OUTPUT_SLOT = 0


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


def _in_bounds(tile, bounds):
    return (
        bounds["x"] <= tile[0] < bounds["x"] + bounds["w"]
        and bounds["y"] <= tile[1] < bounds["y"] + bounds["h"]
    )


def _belts_along(route, final_target):
    """A belt on every tile of ``route``, each pointing at the next one."""
    belts = []
    for index, tile in enumerate(route):
        nxt = route[index + 1] if index + 1 < len(route) else final_target
        belts.append(("belt", tile[0], tile[1], direction_to_rotation(tile, nxt)))
    return belts


def _cutter_geometry(origin):
    """Tiles a rotation-0 cutter uses: (body, input, slot-0 out, slot-1 out).

    The default cutter is 2x1. Its acceptor sits on the left tile facing
    ``bottom``, so items enter from below; both ejectors face ``top``, so the
    two halves come out of the tiles directly above the body.
    """
    cx, cy = origin
    body = ((cx, cy), (cx + 1, cy))
    return body, (cx, cy + 1), (cx, cy - 1), (cx + 1, cy - 1)


def _cutter_plan_at(origin, patch, blocked, bounds, goal_slot):
    body, feed_tile, out_a, out_b = _cutter_geometry(origin)
    goal_out, waste_out = (out_a, out_b) if goal_slot == 0 else (out_b, out_a)

    used = set(body) | {feed_tile, out_a, out_b}
    if any(not _in_bounds(t, bounds) or t in blocked for t in used):
        return None

    # The offcut has to keep moving or the cutter stalls, so the trash sits
    # right on the waste ejector and never needs a belt of its own.
    hub_route = find_route(goal_out, HUB_INPUT, blocked | set(body) | {feed_tile, waste_out}, bounds)
    if hub_route is None:
        return None

    feed_route = find_route(patch, feed_tile, blocked | set(body) | {out_a, out_b} | set(hub_route), bounds)
    if feed_route is None or len(feed_route) < 2:
        return None

    # Placement order matters: the game snaps a belt towards whatever acceptor is
    # already next to it, so a trash placed before the goal belt steals the cut
    # half. Building the trash last leaves the goal belt pointing at the hub.
    return (
        [("miner", patch[0], patch[1], direction_to_rotation(patch, feed_route[1]))]
        + _belts_along(feed_route[1:], body[0])
        + [("cutter", origin[0], origin[1], 0)]
        + _belts_along(hub_route, HUB_TILE)
        + [("trash", waste_out[0], waste_out[1], 0)]
    )


def plan_cutter_line(map_payload, bounds, source_shape, goal_slot=0, budget=None):
    """Plan miner -> belt -> cutter, one half to the hub and the offcut to a trash.

    ``source_shape`` is the raw shape to mine (``CuCuCuCu`` for a ``----CuCu``
    goal); the goal itself is never on the map, so planning has to target the
    material the cutter is fed with.
    """
    blocked = occupied_tiles(map_payload)
    patches = [p for p in shape_patches(map_payload, source_shape) if p not in blocked]
    if not patches:
        return None

    patches.sort(key=lambda p: abs(p[0] - HUB_INPUT[0]) + abs(p[1] - HUB_INPUT[1]))

    # Cutter sites near the hub keep the post-cut belt run short, which matters
    # because every tile of it eats into the placement budget.
    sites = [
        (x, y)
        for x in range(bounds["x"], bounds["x"] + bounds["w"] - 1)
        for y in range(bounds["y"] + 1, bounds["y"] + bounds["h"] - 1)
    ]
    sites.sort(key=lambda s: abs(s[0] - HUB_INPUT[0]) + abs(s[1] - HUB_INPUT[1]))

    for patch in patches[:MAX_CANDIDATES]:
        for site in sites[:MAX_CUTTER_SITES]:
            plan = _cutter_plan_at(site, patch, blocked, bounds, goal_slot)
            if plan and (budget is None or len(plan) <= budget):
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
        self.plan = self._plan(map_payload)
        self._queue = plan_to_actions(self.plan, self.env) if self.plan else []
        self._route = {(x, y) for _id, x, y, _rot in (self.plan or [])}
        return len(self._queue)

    def _plan(self, map_payload):
        """A cut goal needs a cutter; a raw goal is just a mining line."""
        sources = source_shapes(self.target_shape)
        if sources and {"cutter", "trash"} <= set(self.env.buildings):
            return plan_cutter_line(
                map_payload,
                self.env.bounds,
                next(iter(sources)),
                goal_slot=GOAL_OUTPUT_SLOT,
                budget=self.env.placement_budget,
            )
        return plan_mining_line(map_payload, self.env.bounds, self.target_shape)

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

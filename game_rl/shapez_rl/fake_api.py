import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HUB_BOUNDS = {"x": -2, "y": -2, "w": 4, "h": 4}
HUB_INPUT = (-3, 1)
TICKS_PER_DELIVERY = 300

CATALOGUE = {
    "belt": ["default"],
    "miner": ["chainable"],
    "cutter": ["default", "quad"],
    "rotater": ["default", "ccw", "rotate180"],
    "stacker": ["default"],
    "mixer": ["default"],
    "painter": ["default", "mirrored", "double", "quad"],
    "trash": ["default"],
    "balancer": ["default", "merger", "splitter"],
    "underground_belt": ["default", "tier2"],
    "storage": ["default"],
}


BUILDING_SIZE = {"cutter": (2, 1), "painter": (2, 1), "storage": (2, 2), "hub": (4, 4)}

NON_REMOVABLE = {"hub"}

SHAPE_KEYS = ("CuCuCuCu", "RuRuRuRu", "WuWuWuWu", "SuSuSuSu")
COLOR_KEYS = ("red", "green", "blue")


class FakeGame:
    """Deterministic world state keyed on a seed."""

    def __init__(self, seed=None):
        self.reset(seed)

    def reset(self, seed=None):
        self.seed = random.randrange(0, 2**32) if seed is None else int(seed)
        rng = random.Random(self.seed)

        self.game_time = 0.0
        self.ticks_run = 0
        self.stored_shapes = {}
        self.level = 1
        self.next_uid = 10000

        self.resources = {}
        for _ in range(rng.randint(14, 22)):
            cx, cy = rng.randint(-28, 28), rng.randint(-28, 28)
            if abs(cx) < 5 and abs(cy) < 5:
                continue
            kind = "shape" if rng.random() < 0.65 else "color"
            key = rng.choice(SHAPE_KEYS if kind == "shape" else COLOR_KEYS)
            for _ in range(rng.randint(3, 9)):
                x = cx + rng.randint(-2, 2)
                y = cy + rng.randint(-2, 2)
                if abs(x) < 5 and abs(y) < 5:
                    continue
                self.resources[(x, y)] = (kind, key)

        self.buildings = {}  # uid -> dict
        self.occupied = {}  # (x, y) -> uid
        self._place("hub", HUB_BOUNDS["x"], HUB_BOUNDS["y"], 0, "default")


    def _size(self, building_id, rotation):
        w, h = BUILDING_SIZE.get(building_id, (1, 1))
        if rotation in (90, 270) and (w, h) != (h, w):
            w, h = h, w
        return w, h

    def _remove(self, uid):
        entry = self.buildings.pop(uid, None)
        if entry is None:
            return
        for dx in range(entry["w"]):
            for dy in range(entry["h"]):
                self.occupied.pop((entry["x"] + dx, entry["y"] + dy), None)

    def _place(self, building_id, x, y, rotation, variant):
        """Place, replacing whatever removable buildings are in the way.

        Matches the real API: only non-removable buildings (the hub) block a
        placement. Placing onto a belt replaces it and issues a *new* uid.
        """
        w, h = self._size(building_id, rotation)
        cells = [(x + dx, y + dy) for dx in range(w) for dy in range(h)]

        blocking = {self.occupied[cell] for cell in cells if cell in self.occupied}
        if any(self.buildings[uid]["id"] in NON_REMOVABLE for uid in blocking):
            return None
        for uid in blocking:
            self._remove(uid)

        uid = self.next_uid
        self.next_uid += 1
        entry = {
            "uid": uid,
            "id": building_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "rotation": rotation,
            "variant": variant,
        }
        self.buildings[uid] = entry
        for cell in cells:
            self.occupied[cell] = uid
        return entry

    def place(self, building_id, x, y, rotation, variant):
        """Returns (entry, error_slug)."""
        if building_id not in CATALOGUE:
            return None, ("invalid-building-id", 400)
        legal = CATALOGUE[building_id]
        chosen = variant if variant is not None else "default"
        if chosen not in legal:
            return None, ("invalid-variant", 400)
        if rotation not in (0, 90, 180, 270):
            return None, ("invalid-rotation", 400)

        entry = self._place(building_id, x, y, rotation, chosen)
        if entry is None:
            return None, ("placement-blocked", 409)
        return entry, None

    def destroy_removable(self):
        removed = 0
        for uid, entry in list(self.buildings.items()):
            if entry["id"] in NON_REMOVABLE:
                continue
            self._remove(uid)
            removed += 1
        return removed

    def _faces(self, entry):
        """Tile a building ejects into. rotation is clockwise from up, y grows down."""
        delta = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}[entry["rotation"]]
        return (entry["x"] + delta[0], entry["y"] + delta[1])

    def _producing_miners(self):
        """Miners on a resource tile that eject into belts reaching the hub input."""
        producing = []
        for entry in self.buildings.values():
            if entry["id"] != "miner":
                continue
            cell = (entry["x"], entry["y"])
            if cell not in self.resources:
                continue

            target = self._faces(entry)
            if target == HUB_INPUT:
                producing.append(self.resources[cell])
                continue

            uid = self.occupied.get(target)
            if uid is None or self.buildings[uid]["id"] != "belt":
                continue
            if self._belt_reaches_hub(target):
                producing.append(self.resources[cell])
        return producing

    def _belt_reaches_hub(self, start):
        seen = {start}
        frontier = [start]
        while frontier:
            x, y = frontier.pop()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in seen:
                    continue
                if (nx, ny) == HUB_INPUT:
                    return True
                uid = self.occupied.get((nx, ny))
                if uid is not None and self.buildings[uid]["id"] == "belt":
                    seen.add((nx, ny))
                    frontier.append((nx, ny))
        return False

    def tick(self, ticks):
        self.ticks_run = ticks
        self.game_time += ticks / 60.0
        deliveries = ticks // TICKS_PER_DELIVERY
        if deliveries:
            for kind, key in self._producing_miners():
                if kind != "shape":
                    continue
                self.stored_shapes[key] = self.stored_shapes.get(key, 0) + deliveries

    def hub_goals(self):
        return {
            "level": self.level,
            "storedShapes": dict(self.stored_shapes),
            "upgradeLevels": {},
        }

    def compact_state(self, extra=None):
        body = {
            "state": "s10_gameRunning",
            "compact": True,
            "gameTime": self.game_time,
            "ticksRun": self.ticks_run,
            "mapSeed": self.seed,
            "hubGoals": self.hub_goals(),
        }
        body.update(extra or {})
        return body

    def full_state(self, extra=None):
        body = {
            "state": "s10_gameRunning",
            "gameTime": self.game_time,
            "ticksRun": self.ticks_run,
            "savegame": {
                "dump": {
                    "map": {"seed": self.seed},
                    "hubGoals": self.hub_goals(),
                    "entities": [{"uid": u} for u in self.buildings],
                }
            },
        }
        body.update(extra or {})
        return body

    def map_window(self, x, y, w, h, compact):
        resources, buildings = [], []
        for (rx, ry), (kind, key) in self.resources.items():
            if x <= rx < x + w and y <= ry < y + h:
                resources.append([rx, ry, kind, key] if compact
                                 else {"x": rx, "y": ry, "type": kind, "key": key})

        for entry in self.buildings.values():
            if not (x <= entry["x"] < x + w and y <= entry["y"] < y + h):
                continue
            if compact:
                buildings.append(
                    [entry["id"], entry["x"], entry["y"], entry["w"], entry["h"], entry["rotation"]]
                )
            else:
                buildings.append(
                    {
                        "uid": entry["uid"],
                        "id": entry["id"],
                        "x": entry["x"],
                        "y": entry["y"],
                        "rotation": entry["rotation"],
                        "variant": entry["variant"],
                        "bounds": {
                            "x": entry["x"], "y": entry["y"],
                            "w": entry["w"], "h": entry["h"],
                        },
                    }
                )

        return {
            "compact": compact,
            "state": "s10_gameRunning",
            "gameTime": self.game_time,
            "mapSeed": self.seed,
            "bounds": {"x": x, "y": y, "w": w, "h": h},
            "resources": resources,
            "buildings": buildings,
        }

    def catalogue(self):
        return {
            "rotations": [0, 90, 180, 270],
            "buildings": [
                {
                    "id": building_id,
                    "layer": "regular",
                    "unlocked": True,
                    "excluded": False,
                    "placeable": True,
                    "variants": [{"variant": v, "rotationVariants": [0]} for v in variants],
                }
                for building_id, variants in CATALOGUE.items()
            ],
        }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    @property
    def game(self):
        return self.server.game

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        with self.server.lock:
            if parsed.path == "/rl/gamestate":
                self._send(200, self.game.full_state())
            elif parsed.path == "/rl/buildings":
                self._send(200, self.game.catalogue())
            elif parsed.path == "/rl/map":
                def q(name, default):
                    return int(query.get(name, [default])[0])

                self._send(
                    200,
                    self.game.map_window(
                        q("x", -16), q("y", -16), q("w", 32), q("h", 32),
                        query.get("compact", ["0"])[0] == "1",
                    ),
                )
            else:
                self._send(404, {"error": "not-found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._body()

        with self.server.lock:
            if parsed.path == "/rl/reset":
                seed = body.get("seed")
                if seed is not None and (
                    not isinstance(seed, int) or isinstance(seed, bool)
                    or seed < 0 or seed > 0xFFFFFFFF
                ):
                    self._send(400, {"error": "invalid-seed"})
                    return
                self.game.reset(seed)
                self._send(200, self.game.full_state({"reset": True, "seed": self.game.seed}))

            elif parsed.path == "/rl/tick":
                ticks = body.get("ticks", 1)
                if not isinstance(ticks, int) or isinstance(ticks, bool) or not 0 <= ticks <= 100000:
                    self._send(400, {"error": "ticks-must-be-integer-0-to-100000"})
                    return
                self.game.tick(ticks)
                self._send(
                    200,
                    self.game.compact_state() if body.get("compact")
                    else self.game.full_state(),
                )

            elif parsed.path == "/rl/building":
                entry, error = self.game.place(
                    body.get("id"), body.get("x"), body.get("y"),
                    body.get("rotation", 0), body.get("variant"),
                )
                if error:
                    self._send(error[1], {"error": error[0]})
                    return
                extra = {
                    "placed": True,
                    "entityUid": entry["uid"],
                    "building": {
                        "id": entry["id"], "x": entry["x"], "y": entry["y"],
                        "rotation": entry["rotation"], "variant": entry["variant"],
                        "bounds": {
                            "x": entry["x"], "y": entry["y"],
                            "w": entry["w"], "h": entry["h"],
                        },
                    },
                }
                self._send(
                    200,
                    self.game.compact_state(extra) if body.get("compact")
                    else self.game.full_state(extra),
                )

            elif parsed.path == "/rl/destroy-removable-buildings":
                removed = self.game.destroy_removable()
                self._send(200, self.game.full_state({"destroyed": removed}))

            else:
                self._send(404, {"error": "not-found"})


class FakeShapezServer:
    """Runs FakeGame over HTTP on a free localhost port."""

    def __init__(self, seed=None):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.game = FakeGame(seed)
        self._server.lock = threading.Lock()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def game(self):
        return self._server.game

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.stop()

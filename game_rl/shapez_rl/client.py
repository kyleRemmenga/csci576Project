"""HTTP client for the shapez RL API."""

import http.client
import json
import urllib.parse


class RlApiError(RuntimeError):
    """The API returned a non-2xx response."""

    def __init__(self, status, error, path):
        super().__init__(f"{path} -> HTTP {status}: {error}")
        self.status = status
        self.error = error
        self.path = path


class RlApiUnavailable(RuntimeError):
    """The API could not be reached at all."""


class ShapezClient:
    """Talks to a single running headless shapez instance."""

    def __init__(self, base_url="http://127.0.0.1:17872", timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        parsed = urllib.parse.urlsplit(self.base_url)
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._conn = None

    def _connection(self):
        # A fresh HTTPConnection per request exhausts Windows' socket buffers
        # over long runs (WinError 10055), so keep one alive and reuse it.
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self.timeout
            )
        return self._conn

    def _request(self, path, method="GET", payload=None, timeout=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        last_error = None
        for attempt in range(2):
            try:
                conn = self._connection()
                conn.request(method, path, body=data, headers=headers)
                response = conn.getresponse()
                body = response.read()
                break
            except (http.client.HTTPException, OSError) as ex:
                self._conn = None
                last_error = ex
        else:
            raise RlApiUnavailable(f"{self.base_url}{path}: {last_error}") from last_error

        if response.status >= 400:
            text = body.decode("utf-8", "replace")
            try:
                error = json.loads(text).get("error", text)
            except json.JSONDecodeError:
                error = text
            raise RlApiError(response.status, error, path)

        return json.loads(body.decode("utf-8"))

    def reset(self, seed=None, goal_level=None):
        """Start a fresh episode, optionally on a chosen map seed and hub level."""
        payload = {"seed": seed}
        if goal_level is not None:
            # Sets the hub goal and unlocks every building earned up to that level.
            payload["goalLevel"] = int(goal_level)
        return self._request("/rl/reset", method="POST", payload=payload)

    def gamestate(self):
        """Full savegame dump. Debugging only - step loops want the compact forms."""
        return self._request("/rl/gamestate")

    def map_window(self, x, y, w, h, compact=True):
        params = {"x": x, "y": y, "w": w, "h": h}
        if compact:
            params["compact"] = 1
        return self._request(f"/rl/map?{urllib.parse.urlencode(params)}")

    def buildings(self):
        """Catalogue of buildings with their currently legal variants."""
        return self._request("/rl/buildings")

    def tick(self, ticks, compact=True):
        """Advance the simulation. 60 ticks == 1 second of game time."""
        return self._request(
            "/rl/tick", method="POST", payload={"ticks": ticks, "compact": compact}
        )

    def place_building(
        self, building_id, x, y, rotation=0, variant=None, rotation_variant=None, compact=True
    ):
        """Place a building."""
        payload = {"id": building_id, "x": x, "y": y, "rotation": rotation, "compact": compact}
        if variant is not None:
            payload["variant"] = variant
        if rotation_variant is not None:
            payload["rotationVariant"] = rotation_variant
        return self._request("/rl/building", method="POST", payload=payload)

    def destroy_removable_buildings(self):
        return self._request("/rl/destroy-removable-buildings", method="POST")

    def ping(self):
        try:
            self.gamestate()
            return True
        except (RlApiError, RlApiUnavailable):
            return False


def _hub_goals(payload):
    """hubGoals from either a compact or a full response."""
    if "hubGoals" in payload:
        return payload["hubGoals"]
    return payload["savegame"]["dump"]["hubGoals"]


def stored_shapes(payload):
    """Shape key -> count delivered to the hub."""
    return _hub_goals(payload).get("storedShapes", {})


def hub_level(payload):
    return _hub_goals(payload)["level"]


def map_seed(payload):
    if "mapSeed" in payload:
        return payload["mapSeed"]
    if "seed" in payload:
        return payload["seed"]
    return payload["savegame"]["dump"]["map"]["seed"]

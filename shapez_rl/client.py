"""HTTP client for the shapez RL API."""

import json
import urllib.error
import urllib.parse
import urllib.request


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

    def _request(self, path, method="GET", payload=None, timeout=None):
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            body = ex.read().decode("utf-8", "replace")
            try:
                error = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                error = body
            raise RlApiError(ex.code, error, path) from ex
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            raise RlApiUnavailable(f"{self.base_url}{path}: {ex}") from ex

    def reset(self, seed=None):
        """Start a fresh episode, optionally on a chosen map seed."""
        return self._request("/rl/reset", method="POST", payload={"seed": seed})

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
        """Place a building.

        rotation_variant is left unset on purpose: the game then computes the optimal
        one, which is what auto-curves belts into their neighbours.
        """
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

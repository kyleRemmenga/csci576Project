"""Turn ``/rl/map`` payloads into tensors."""

import numpy as np

DEFAULT_BUILDINGS = (
    "belt",
    "miner",
    "cutter",
    "rotater",
    "stacker",
    "mixer",
    "painter",
    "trash",
    "balancer",
    "underground_belt",
    "storage",
)

DEFAULT_SHAPE_KEYS = ("CuCuCuCu", "RuRuRuRu", "WuWuWuWu", "SuSuSuSu")
DEFAULT_COLOR_KEYS = ("red", "green", "blue")

ROTATIONS = (0, 90, 180, 270)


def iter_resources(payload):
    """Yield ``(x, y, kind, key)`` from either payload form."""
    compact = payload.get("compact", False)
    for resource in payload.get("resources", []):
        if compact:
            yield resource[0], resource[1], resource[2], resource[3]
        else:
            yield resource["x"], resource["y"], resource.get("type"), resource.get("key")

def iter_buildings(payload):
    """Yield ``(id, x, y, w, h, rotation)`` from either payload form."""
    compact = payload.get("compact", False)
    for building in payload.get("buildings", []):
        if compact:
            yield tuple(building[:6])
        else:
            bounds = building.get("bounds") or {}
            yield (
                building.get("id"),
                bounds.get("x", building.get("x")),
                bounds.get("y", building.get("y")),
                bounds.get("w", 1),
                bounds.get("h", 1),
                building.get("rotation", 0),
            )

class MapEncoder:
    """Encodes a map window into a fixed-size channel stack."""

    def __init__(
        self,
        bounds,
        buildings=DEFAULT_BUILDINGS,
        shape_keys=DEFAULT_SHAPE_KEYS,
        color_keys=DEFAULT_COLOR_KEYS,
    ):
        self.bounds = dict(bounds)
        self.buildings = tuple(buildings)
        self.shape_keys = tuple(shape_keys)
        self.color_keys = tuple(color_keys)

        self._building_index = {name: i for i, name in enumerate(self.buildings)}
        self._shape_index = {key: i for i, key in enumerate(self.shape_keys)}
        self._color_index = {key: i for i, key in enumerate(self.color_keys)}

        names = ["resource_shape_any", "resource_color_any"]
        names += [f"resource_shape_{k}" for k in self.shape_keys]
        names += [f"resource_color_{k}" for k in self.color_keys]
        names += [f"building_{b}" for b in self.buildings]
        names += ["building_other", "building_hub"]
        names += [f"rotation_{r}" for r in ROTATIONS]
        self.channel_names = tuple(names)
        self._channel = {name: i for i, name in enumerate(self.channel_names)}

    @property
    def height(self):
        return self.bounds["h"]

    @property
    def width(self):
        return self.bounds["w"]

    @property
    def num_channels(self):
        return len(self.channel_names)

    @property
    def shape(self):
        return (self.num_channels, self.height, self.width)

    def _to_indices(self, x, y):
        """World tile -> ``(row, col)``, or None if outside the window."""
        col = x - self.bounds["x"]
        row = y - self.bounds["y"]
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def _footprint(self, origin_x, origin_y, width, height):
        """Every in-window cell a building covers."""
        if origin_x is None or origin_y is None:
            return
        for x in range(origin_x, origin_x + width):
            for y in range(origin_y, origin_y + height):
                cell = self._to_indices(x, y)
                if cell is not None:
                    yield cell

    def encode(self, map_payload):
        """Map payload -> float32 ``(C, H, W)``."""
        grid = np.zeros(self.shape, dtype=np.float32)
        channel = self._channel

        for x, y, kind, key in iter_resources(map_payload):
            cell = self._to_indices(x, y)
            if cell is None:
                continue
            row, col = cell

            if kind == "shape":
                grid[channel["resource_shape_any"], row, col] = 1.0
                if key in self._shape_index:
                    grid[channel[f"resource_shape_{key}"], row, col] = 1.0
            elif kind == "color":
                grid[channel["resource_color_any"], row, col] = 1.0
                if key in self._color_index:
                    grid[channel[f"resource_color_{key}"], row, col] = 1.0

        for building_id, bx, by, bw, bh, rotation in iter_buildings(map_payload):
            if building_id in self._building_index:
                type_channel = channel[f"building_{building_id}"]
            elif building_id == "hub":
                type_channel = channel["building_hub"]
            else:
                type_channel = channel["building_other"]

            rotation_channel = channel.get(f"rotation_{rotation}")

            for row, col in self._footprint(bx, by, bw, bh):
                grid[type_channel, row, col] = 1.0
                if rotation_channel is not None:
                    grid[rotation_channel, row, col] = 1.0

        return grid

    def occupancy(self, map_payload):
        """Boolean ``(H, W)`` - True where a building already sits."""
        occupied = np.zeros((self.height, self.width), dtype=bool)
        for _id, bx, by, bw, bh, _rotation in iter_buildings(map_payload):
            for row, col in self._footprint(bx, by, bw, bh):
                occupied[row, col] = True
        return occupied

    def resource_mask(self, map_payload, kind="shape", key=None):
        """Boolean ``(H, W)`` for resource tiles, optionally filtered by key."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for x, y, resource_kind, resource_key in iter_resources(map_payload):
            if resource_kind != kind:
                continue
            if key is not None and resource_key != key:
                continue
            cell = self._to_indices(x, y)
            if cell is not None:
                mask[cell] = True
        return mask

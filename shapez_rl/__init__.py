"""RL environment for the shapez.io headless fork."""

from .client import (
    RlApiError,
    RlApiUnavailable,
    ShapezClient,
    hub_level,
    map_seed,
    stored_shapes,
)
from .encoding import DEFAULT_BUILDINGS, MapEncoder, iter_buildings, iter_resources
from .fake_api import FakeShapezServer

__all__ = [
    "DEFAULT_BUILDINGS",
    "FakeShapezServer",
    "MapEncoder",
    "RlApiError",
    "RlApiUnavailable",
    "ShapezClient",
    "hub_level",
    "iter_buildings",
    "iter_resources",
    "map_seed",
    "stored_shapes",
]


def __getattr__(name):
    if name == "ShapezBuildEnv":
        from .env import ShapezBuildEnv

        return ShapezBuildEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

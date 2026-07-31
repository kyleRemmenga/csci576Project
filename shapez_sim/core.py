from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Set, Tuple


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def delta(self) -> Tuple[int, int]:
        return _DELTAS[self]

    def opposite(self) -> "Direction":
        return Direction((int(self) + 2) % 4)


_DELTAS = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}


class BuildingType(Enum):
    HUB = "hub"
    EXTRACTOR = "extractor"
    BELT = "belt"
    CUTTER = "cutter"
    ROTATOR = "rotator"
    PAINTER = "painter"
    MIXER = "mixer"
    STACKER = "stacker"
    TRASH = "trash"
    SPLITTER = "splitter"


@dataclass(frozen=True)
class Item:
    code: str

    def __str__(self) -> str:
        return self.code


CIRCLE = Item("circle")
SQUARE = Item("square")
WINDMILL = Item("windmill")


@dataclass
class MechanicsConfig:
    enabled: Set[BuildingType] = field(
        default_factory=lambda: {BuildingType.EXTRACTOR, BuildingType.BELT}
    )
    extractor_interval: int = 3

    def is_enabled(self, building_type: BuildingType) -> bool:
        if building_type is BuildingType.HUB:
            return True
        return building_type in self.enabled

    def enable(self, *types: BuildingType) -> None:
        self.enabled.update(types)

    def disable(self, *types: BuildingType) -> None:
        self.enabled.difference_update(types)


@dataclass
class SimConfig:
    width: int = 24
    height: int = 24
    hub_size: int = 4
    resource_patch_size: int = 2
    resource_spacing: int = 1
    resource_hub_clearance: int = 2
    random_seed: Optional[int] = None
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .buildings import Building
from .core import Item


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._buildings: Dict[Tuple[int, int], Building] = {}
        self._resources: Dict[Tuple[int, int], Item] = {}

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def building_at(self, x: int, y: int) -> Optional[Building]:
        return self._buildings.get((x, y))

    def resource_at(self, x: int, y: int) -> Optional[Item]:
        return self._resources.get((x, y))

    def _set_resource(self, x: int, y: int, item: Item) -> None:
        if self.in_bounds(x, y):
            self._resources[(x, y)] = item

    def _add_resource_patch(
        self, x: int, y: int, width: int, height: int, item: Item
    ) -> None:
        for dy in range(height):
            for dx in range(width):
                self._set_resource(x + dx, y + dy, item)

    def resource_area_is_empty(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        resource_clearance: int = 0,
        building_clearance: int = 0,
    ) -> bool:
        for dy in range(height):
            for dx in range(width):
                tx, ty = x + dx, y + dy
                if not self.in_bounds(tx, ty):
                    return False

        for ty in range(max(0, y - resource_clearance), min(self.height, y + height + resource_clearance)):
            for tx in range(max(0, x - resource_clearance), min(self.width, x + width + resource_clearance)):
                if self.resource_at(tx, ty) is not None:
                    return False

        for ty in range(max(0, y - building_clearance), min(self.height, y + height + building_clearance)):
            for tx in range(max(0, x - building_clearance), min(self.width, x + width + building_clearance)):
                if self.building_at(tx, ty) is not None:
                    return False
        return True

    def can_place(self, building: Building) -> bool:
        for tx, ty in building.tiles():
            if not self.in_bounds(tx, ty):
                return False
            if (tx, ty) in self._buildings:
                return False
        return True

    def place(self, building: Building) -> None:
        for tile in building.tiles():
            self._buildings[tile] = building

    def remove(self, building: Building) -> None:
        for tile in building.tiles():
            if self._buildings.get(tile) is building:
                del self._buildings[tile]

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import List, Optional

from .buildings import Belt, Building, Extractor, Hub
from .core import CIRCLE, SQUARE, WINDMILL, Direction, Item, SimConfig
from .grid import Grid


class Simulation:
    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.grid = Grid(self.config.width, self.config.height)
        self.buildings: List[Building] = []
        self.tick_count = 0
        self._order: List[Building] = []
        self._order_dirty = True
        self.hub = self._place_centered_hub()
        self._generate_resources()

    def _place_centered_hub(self) -> Hub:
        size = self.config.hub_size
        x = (self.config.width - size) // 2
        y = (self.config.height - size) // 2
        hub = Hub(x, y, size)
        if not self.grid.can_place(hub):
            raise ValueError("hub does not fit on the configured grid")
        self.grid.place(hub)
        self.buildings.append(hub)
        self._order_dirty = True
        return hub

    def _generate_resources(self) -> None:
        size = self.config.resource_patch_size
        if size <= 0:
            raise ValueError("resource_patch_size must be positive")
        if self.config.resource_spacing < 0:
            raise ValueError("resource_spacing cannot be negative")
        if self.config.resource_hub_clearance < 0:
            raise ValueError("resource_hub_clearance cannot be negative")

        rng = random.Random(self.config.random_seed)
        candidates = [
            (x, y)
            for y in range(self.grid.height - size + 1)
            for x in range(self.grid.width - size + 1)
        ]
        rng.shuffle(candidates)

        for item in (CIRCLE, SQUARE, WINDMILL):
            for x, y in candidates:
                if self.grid.resource_area_is_empty(
                    x,
                    y,
                    size,
                    size,
                    resource_clearance=self.config.resource_spacing,
                    building_clearance=self.config.resource_hub_clearance,
                ):
                    self.grid._add_resource_patch(x, y, size, size, item)
                    break
            else:
                raise ValueError("grid does not have room for all resource patches")

    def place(self, building: Building) -> bool:
        if not isinstance(building, (Belt, Extractor)):
            return False
        if not self.config.mechanics.is_enabled(building.type):
            return False
        if not self.grid.can_place(building):
            return False
        if isinstance(building, Extractor):
            if self.grid.resource_at(building.x, building.y) is None:
                return False
        self.grid.place(building)
        self.buildings.append(building)
        self._order_dirty = True
        return True

    def remove(self, building: Building) -> bool:
        if building not in self.buildings:
            return False
        if building is self.hub:
            return False
        self.grid.remove(building)
        self.buildings.remove(building)
        self._order_dirty = True
        return True

    def try_transfer(self, source: Building, item: Item, direction: Direction) -> bool:
        dx, dy = direction.delta
        target = self.grid.building_at(source.x + dx, source.y + dy)
        if target is None or target is source:
            return False
        if not target.can_accept(item, direction):
            return False
        target.accept(item, direction)
        return True

    def _successor(self, building: Building) -> Optional[Building]:
        dx, dy = building.rotation.delta
        target = self.grid.building_at(building.x + dx, building.y + dy)
        if target is None or target is building:
            return None
        if not target.accepts_from(building.rotation):
            return None
        return target

    def _rebuild_order(self) -> None:
        successor = {}
        predecessors = defaultdict(list)
        for building in self.buildings:
            nxt = self._successor(building)
            successor[building] = nxt
            if nxt is not None:
                predecessors[nxt].append(building)

        out_degree = {b: (1 if successor[b] is not None else 0) for b in self.buildings}
        queue = deque(b for b in self.buildings if out_degree[b] == 0)

        order: List[Building] = []
        seen = set()
        while queue:
            building = queue.popleft()
            if building in seen:
                continue
            seen.add(building)
            order.append(building)
            for pred in predecessors[building]:
                out_degree[pred] -= 1
                if out_degree[pred] == 0:
                    queue.append(pred)

        for building in self.buildings:
            if building not in seen:
                order.append(building)

        self._order = order
        self._order_dirty = False

    def tick(self) -> None:
        if self._order_dirty:
            self._rebuild_order()
        for building in self._order:
            building.tick(self)
        self.tick_count += 1

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.tick()

    @property
    def delivered(self) -> int:
        return self.hub.total_delivered
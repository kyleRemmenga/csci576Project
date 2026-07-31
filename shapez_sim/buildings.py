from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .core import BuildingType, Direction, Item


class Building:
    type: Optional[BuildingType] = None
    size: Tuple[int, int] = (1, 1)

    def __init__(self, x: int, y: int, rotation: Direction = Direction.NORTH):
        self.x = x
        self.y = y
        self.rotation = Direction(rotation)

    def tiles(self) -> List[Tuple[int, int]]:
        w, h = self.size
        return [(self.x + dx, self.y + dy) for dy in range(h) for dx in range(w)]

    def output_tile(self) -> Tuple[int, int]:
        dx, dy = self.rotation.delta
        return self.x + dx, self.y + dy

    def accepts_from(self, travel: Direction) -> bool:
        return False

    def can_accept(self, item: Item, travel: Direction) -> bool:
        return False

    def accept(self, item: Item, travel: Direction) -> None:
        raise NotImplementedError

    def tick(self, sim) -> None:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.x}, {self.y}, {self.rotation.name})"


class Belt(Building):
    type = BuildingType.BELT

    def __init__(self, x: int, y: int, rotation: Direction = Direction.NORTH):
        super().__init__(x, y, rotation)
        self.item: Optional[Item] = None

    def accepts_from(self, travel: Direction) -> bool:
        return travel != self.rotation.opposite()

    def can_accept(self, item: Item, travel: Direction) -> bool:
        return self.item is None and self.accepts_from(travel)

    def accept(self, item: Item, travel: Direction) -> None:
        self.item = item

    def tick(self, sim) -> None:
        if self.item is None:
            return
        if sim.try_transfer(self, self.item, self.rotation):
            self.item = None


class Extractor(Building):
    type = BuildingType.EXTRACTOR

    def __init__(
        self,
        x: int,
        y: int,
        rotation: Direction = Direction.NORTH,
        interval: int = 3,
    ):
        super().__init__(x, y, rotation)
        self.interval = interval
        self.timer = 0
        self.buffer: Optional[Item] = None

    def tick(self, sim) -> None:
        if self.buffer is not None and sim.try_transfer(self, self.buffer, self.rotation):
            self.buffer = None

        self.timer += 1
        if self.buffer is None and self.timer >= self.interval:
            resource = sim.grid.resource_at(self.x, self.y)
            if resource is not None:
                self.buffer = resource
                self.timer = 0


class Hub(Building):
    type = BuildingType.HUB

    def __init__(self, x: int, y: int, size: int = 4):
        super().__init__(x, y, Direction.NORTH)
        self.size = (size, size)
        self.delivered: Dict[str, int] = {}
        self.total_delivered = 0

    def accepts_from(self, travel: Direction) -> bool:
        return True

    def can_accept(self, item: Item, travel: Direction) -> bool:
        return True

    def accept(self, item: Item, travel: Direction) -> None:
        self.delivered[item.code] = self.delivered.get(item.code, 0) + 1
        self.total_delivered += 1

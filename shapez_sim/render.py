from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from .buildings import Belt, Extractor, Hub
from .core import Direction

ARROWS = {
    Direction.NORTH: "^",
    Direction.EAST: ">",
    Direction.SOUTH: "v",
    Direction.WEST: "<",
}

LEGEND = "H hub   ^ > v < belt   o item on belt   . empty   lowercase resource   uppercase extractor"


def tile_char(sim, x: int, y: int) -> str:
    building = sim.grid.building_at(x, y)
    if isinstance(building, Hub):
        return "H"
    if isinstance(building, Extractor):
        resource = sim.grid.resource_at(x, y)
        return resource.code[0].upper() if resource is not None else "E"
    if isinstance(building, Belt):
        return "o" if building.item is not None else ARROWS[building.rotation]
    if building is not None:
        return "?"
    resource = sim.grid.resource_at(x, y)
    if resource is not None:
        return resource.code[0].lower()
    return "."


def render(sim, header: bool = True) -> str:
    lines: List[str] = []
    if header:
        lines.append(f"tick {sim.tick_count}   delivered {sim.delivered}")
    for y in range(sim.grid.height):
        lines.append("".join(tile_char(sim, x, y) for x in range(sim.grid.width)))
    return "\n".join(lines)


def show(sim, header: bool = True) -> None:
    print(render(sim, header))


class Recorder:
    def __init__(self, sim):
        self.sim = sim
        self.frames: List[str] = []

    def capture(self) -> None:
        self.frames.append(render(self.sim))

    def run(self, ticks: int, every: int = 1) -> None:
        self.capture()
        for i in range(ticks):
            self.sim.tick()
            if (i + 1) % every == 0:
                self.capture()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.frames, handle)

    @staticmethod
    def load(path: str) -> List[str]:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def replay(frames: List[str], delay: float = 0.15, clear: bool = True) -> None:
        for frame in frames:
            if clear:
                os.system("cls" if os.name == "nt" else "clear")
            print(frame)
            print(LEGEND)
            time.sleep(delay)

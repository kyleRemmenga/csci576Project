from shapez_sim import (
    CIRCLE,
    SQUARE,
    WINDMILL,
    Belt,
    BuildingType,
    Direction,
    Extractor,
    Hub,
    MechanicsConfig,
    SimConfig,
    Simulation,
    render,
)

E = Direction.EAST
W = Direction.WEST
N = Direction.NORTH


def make_line_sim():
    sim = Simulation(SimConfig(width=10, height=6, hub_size=2, random_seed=1))
    belts = []
    for x in range(1, 4):
        belt = Belt(x, 2, E)
        assert sim.place(belt)
        belts.append(belt)
    return sim, belts


def first_resource(sim):
    for y in range(sim.grid.height):
        for x in range(sim.grid.width):
            item = sim.grid.resource_at(x, y)
            if item is not None:
                return x, y, item
    raise AssertionError("simulation has no resource tiles")


def test_item_moves_one_tile_per_tick():
    sim, belts = make_line_sim()
    belts[0].item = CIRCLE

    sim.tick()
    assert belts[0].item is None
    assert belts[1].item is CIRCLE

    sim.tick()
    assert belts[1].item is None
    assert belts[2].item is CIRCLE


def test_full_chain_shifts_as_a_unit():
    sim, belts = make_line_sim()
    for belt in belts:
        belt.item = CIRCLE

    sim.tick()

    assert belts[0].item is None
    assert belts[1].item is CIRCLE
    assert belts[2].item is CIRCLE
    assert sim.delivered == 1


def test_blocked_chain_does_not_move():
    sim = Simulation(SimConfig(width=10, height=6, hub_size=2, random_seed=1))
    belts = []
    for x in range(3):
        belt = Belt(x, 0, E)
        assert sim.place(belt)
        belts.append(belt)
    for belt in belts:
        belt.item = CIRCLE

    sim.tick()

    assert all(belt.item is CIRCLE for belt in belts)


def test_belt_rejects_input_through_its_output_face():
    belt = Belt(3, 3, E)
    assert belt.accepts_from(E)
    assert belt.accepts_from(N)
    assert not belt.accepts_from(W)


def test_extractor_needs_a_resource_tile():
    sim = Simulation(SimConfig(width=10, height=10, hub_size=2, random_seed=1))
    empty = next(
        (x, y)
        for y in range(sim.grid.height)
        for x in range(sim.grid.width)
        if sim.grid.resource_at(x, y) is None and sim.grid.building_at(x, y) is None
    )
    assert not sim.place(Extractor(*empty, E))
    x, y, _ = first_resource(sim)
    assert sim.place(Extractor(x, y, E))


def test_cannot_place_on_occupied_tile():
    sim = Simulation(SimConfig(width=10, height=10, hub_size=2, random_seed=1))
    assert sim.place(Belt(0, 0, E))
    assert not sim.place(Belt(0, 0, N))


def test_disabled_building_is_rejected():
    config = SimConfig(
        width=10,
        height=10,
        hub_size=2,
        random_seed=1,
        mechanics=MechanicsConfig(enabled={BuildingType.BELT}),
    )
    sim = Simulation(config)
    x, y, _ = first_resource(sim)
    assert not sim.place(Extractor(x, y, E))
    assert sim.place(Belt(0, 0, E))


def test_extractor_produces_at_its_interval():
    sim = Simulation(SimConfig(width=10, height=6, hub_size=2, random_seed=1))
    x, y, item = first_resource(sim)
    extractor = Extractor(x, y, E, interval=3)
    assert sim.place(extractor)

    sim.run(2)
    assert extractor.buffer is None
    sim.tick()
    assert extractor.buffer is item


def test_hub_cannot_be_removed():
    sim = Simulation(SimConfig(width=12, height=12, hub_size=4, random_seed=1))
    hub = sim.hub

    assert (hub.x, hub.y) == (4, 4)
    assert not sim.remove(hub)
    assert sim.hub is hub
    assert sim.grid.building_at(hub.x, hub.y) is hub
    assert not sim.place(Hub(0, 0, 2))


def test_resources_are_generated_with_the_grid():
    sim = Simulation(SimConfig(width=12, height=12, hub_size=4, random_seed=1))
    resources = [
        sim.grid.resource_at(x, y)
        for y in range(sim.grid.height)
        for x in range(sim.grid.width)
    ]

    for item in (CIRCLE, SQUARE, WINDMILL):
        assert resources.count(item) == 4


def test_extractor_renders_resource_as_uppercase():
    sim = Simulation(SimConfig(width=10, height=10, hub_size=2, random_seed=1))
    x, y, item = first_resource(sim)
    assert sim.place(Extractor(x, y, E))

    rendered_rows = render(sim, header=False).splitlines()
    assert rendered_rows[y][x] == item.code[0].upper()


def test_default_grid_is_large_and_resources_are_separated():
    config = SimConfig(random_seed=1)
    sim = Simulation(config)
    resource_tiles = [
        (x, y, sim.grid.resource_at(x, y))
        for y in range(sim.grid.height)
        for x in range(sim.grid.width)
        if sim.grid.resource_at(x, y) is not None
    ]

    assert (sim.grid.width, sim.grid.height) == (24, 24)
    for x, y, item in resource_tiles:
        for other_x, other_y, other_item in resource_tiles:
            if item is not other_item:
                assert max(abs(x - other_x), abs(y - other_y)) > config.resource_spacing

        hub_distance = max(
            sim.hub.x - x,
            x - (sim.hub.x + sim.hub.size[0] - 1),
            sim.hub.y - y,
            y - (sim.hub.y + sim.hub.size[1] - 1),
        )
        assert hub_distance > config.resource_hub_clearance


def test_hub_accepts_from_every_side():
    sim = Simulation(SimConfig(width=12, height=12, hub_size=4, random_seed=1))
    hub = sim.hub

    left = Belt(3, 5, E)
    top = Belt(5, 3, Direction.SOUTH)
    right = Belt(8, 5, W)
    for belt in (left, top, right):
        assert sim.place(belt)
        belt.item = CIRCLE

    sim.tick()
    assert hub.total_delivered == 3


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}  {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
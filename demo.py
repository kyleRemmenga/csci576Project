from datetime import datetime
from pathlib import Path

from shapez_sim import Belt, Direction, Extractor, Recorder, SimConfig, Simulation, show

W = Direction.WEST


def build() -> Simulation:
    sim = Simulation(SimConfig(random_seed=7))

    # sim.place(Extractor(21, 10, W, interval=3))

    # for x in range(14, 21):
    #     sim.place(Belt(x, 10, W))

    return sim


def main() -> None:
    sim = build()

    show(sim)
    print()

    for _ in range(4):
        sim.run(5)
        show(sim)
        print()

    print(f"total delivered after {sim.tick_count} ticks: {sim.delivered}")
    print(f"by type: {sim.hub.delivered}")

    recorder = Recorder(build())
    recorder.run(60)
    output_dir = Path("game_runs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_path = output_dir / f"game_run_{timestamp}.json"
    recorder.save(str(output_path))
    print(f"\nsaved {len(recorder.frames)} frames to {output_path}")
    print("replay latest with: python replay.py")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

from shapez_sim import Recorder


def latest_game_run() -> Path:
    runs = list(Path("game_runs").glob("game_run_*.json"))
    if not runs:
        raise FileNotFoundError("no saved game runs found in game_runs")
    return max(runs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_game_run()
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    Recorder.replay(Recorder.load(str(path)), delay=delay)


if __name__ == "__main__":
    main()

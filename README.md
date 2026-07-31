# csci576Project

## Run the Demo

```powershell
python demo.py
```

The demo:

- Creates a 24x24 simulation using lucky number seed `7`
- Places the fixed 4x4 hub in the center
- Generates circle, square, and windmill resource patches.
- Prints the initial grid and four later snapshots at five-tick intervals
- Records 60 ticks and saves 61 frames
- Creates a timestamped JSON file under `game_runs/`

## Simulation Configuration

The default settings are defined in `shapez_sim/core.py` in the `SimConfig`
class:

| Setting | Default | What it controls |
| --- | --- | --- |
| `width` | `24` | Number of grid columns |
| `height` | `24` | Number of grid rows |
| `hub_size` | `4` | Width and height of the square hub |
| `resource_patch_size` | `2` | Width and height of each square resource patch |
| `resource_spacing` | `1` | Minimum empty space between different resource patches |
| `resource_hub_clearance` | `2` | Minimum empty space between resources and the hub |
| `random_seed` | `None` | Resource-layout seed; `None` creates a new layout each run, while the same number repeats the same layout |
| `mechanics` | `MechanicsConfig()` | Controls which building types are enabled and stores mechanics settings |

To change the defaults for every simulation, edit `SimConfig` in
`shapez_sim/core.py`. To change only one game run, pass settings when creating
the simulation. For example, change the `Simulation` line in `demo.py`:

```python
sim = Simulation(
	SimConfig(
		width=30,
		height=30,
		hub_size=4,
		resource_patch_size=3,
		resource_spacing=2,
		resource_hub_clearance=3,
		random_seed=7,
	)
)
```

The grid must be large enough to fit the centered hub and all three resource
patches with the configured spacing. Otherwise, simulation creation raises a
`ValueError`

## Replay a Game Run

Replay the newest file in `game_runs/`:

```powershell
python replay.py
```

Replay a specific recording:

```powershell
python replay.py "game_runs\game_run_2026-07-30_21-34-02-483467.json"
```

An optional second argument controls the delay between frames in seconds:

```powershell
python replay.py "game_runs\game_run_2026-07-30_21-34-02-483467.json" 0.1
```

The default delay is `0.15` seconds

## Grid Symbols

| Symbol | Meaning |
| --- | --- |
| `H` | Hub tile |
| `c`, `s`, `w` | Circle, square, or windmill resource |
| `C`, `S`, `W` | Extractor on that resource |
| `^`, `>`, `v`, `<` | Empty belt and its direction |
| `o` | Item currently on a belt |
| `.` | Empty grid tile |
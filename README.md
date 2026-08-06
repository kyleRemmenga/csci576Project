# shapez RL

Reinforcement learning agent for shapez. The agent places buildings on a tile grid,
then the factory runs and is scored on what it delivers to the hub.

## Setup

```bash
cd game_rl
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below are run from `game_rl/` with `PYTHONPATH=.` set.

## Train

```bash
PYTHONPATH=. .venv/bin/python train.py --timesteps 100000
```

Saves a checkpoint to `runs/ppo_shapez.zip`. Useful flags:

| flag          | default           | what it does                             |
| ------------- | ----------------- | ---------------------------------------- |
| `--timesteps` | 200000            | how long to train                        |
| `--budget`    | 24                | buildings placed per episode             |
| `--ticks`     | 3000              | how long the factory runs before scoring |
| `--n-envs`    | 4                 | parallel environments                    |
| `--save`      | `runs/ppo_shapez` | checkpoint path                          |

## Evaluate

```bash
PYTHONPATH=. .venv/bin/python evaluate.py --model runs/ppo_shapez
```

Scores the checkpoint against two baselines on identical maps:

```
  seed                  policy                  expert    random
  1000       0.0 (  0 del, 24 placed)      10.0 ( 12 planned)      0.0
  ...
mean policy 0.00   mean expert 9.00   mean random 0.00
```

- **expert** - a scripted policy that mines one patch and belts it to the hub. This
  is the number to beat.
- **random** - masked random play. This is the floor.

A policy sitting at the random floor has not learned anything, whatever its loss
curves looked like.

## Running against the real game

Start the game in one terminal (from the `shapez.io_rl` repo):

```bash
nix run .#rl
```

It serves the RL API on port 17872. Then point either script at it:

```bash
PYTHONPATH=. .venv/bin/python evaluate.py --model runs/ppo_shapez \
    --base-url http://127.0.0.1:17872 --episodes 3

PYTHONPATH=. .venv/bin/python train.py \
    --base-url http://127.0.0.1:17872 --n-envs 1 --timesteps 100000
```

**Use `--n-envs 1` against a real game.** Every environment would otherwise share
one game instance and overwrite each other's episodes. For parallel training, start
several games on different ports:

```bash
SHAPEZ_RL_API_PORT=17873 SHAPEZ_WEB_PORT=3006 nix run .#rl
```

## Model architecture

Fully convolutional, trained with `MaskablePPO` (sb3-contrib).

- **Observation** - a `(26, 32, 32)` stack of binary channels over a 32x32 tile
  window: resource types, one channel per building type, and belt rotations.
- **Trunk** - four 3x3 convs at dilations 1, 2, 4, 8. All stride 1, so the feature
  map stays one cell per tile. The dilations give a 31-tile receptive field, wide
  enough to see both ends of a belt run.
- **Action head** - a 1x1 conv producing one logit per
  `(building, tile, rotation)`, 45,056 actions in total. Because it is a conv, the
  same weights apply at every tile, so what the agent learns about placing a belt in
  one place transfers everywhere.
- **Value head** - global average pool, then a single linear layer.

Total: **128,749 parameters**. Flattening the feature map and using a linear action
head instead would need about 3 billion, and would throw away the fact that the tile
an action names is the tile the observation describes.

Illegal actions are masked out before sampling: `env.action_masks()` returns a
boolean over all 45,056 actions, currently encoding tile occupancy.

## Tests

```bash
PYTHONPATH=. .venv/bin/python tests/test_encoding.py
PYTHONPATH=. .venv/bin/python tests/test_env.py
PYTHONPATH=. .venv/bin/python tests/test_expert.py
PYTHONPATH=. .venv/bin/python tests/test_wrappers.py
PYTHONPATH=. .venv/bin/python tests/test_policy.py
```

# Arena Fighters

2D platform fighter where two RL agents train against each other via self-play PPO. Built on PettingZoo and Stable-Baselines3.

Two agents spawn on a multi-platform arena, shoot projectiles, swing melee attacks, dodge, duck, and jump. One neural network plays both sides (shared-weight self-play with observation mirroring). An opponent pool keeps old snapshots around so the agent doesn't overfit to its own latest strategy.

## Features

- **9-action combat**: move, jump, duck, shoot (forward, diagonal up, diagonal down), melee, idle
- **Tile-based physics**: gravity, platform collisions, bullet trajectories
- **Self-play training**: single network plays both agents via observation mirroring
- **Opponent pool**: 20 historical snapshots (80% latest, 20% random older) prevent strategy cycling
- **Custom CNN extractor**: processes a 6-channel 20x40 grid observation + 6-dim vector (HP, positions, cooldowns)
- **3 modes**: headless training, live ASCII watch, frame-by-frame replay
- **Milestone checkpoints**: auto-saves at 100K, 500K, 1M, 5M, 10M steps
- **Episode replay**: JSON-serialized frame logs for post-hoc analysis

## Prerequisites

- Python 3.12+
- NVIDIA GPU recommended (CPU works but slow)

## Installation

```bash
git clone https://github.com/jasperan/arena-fighters.git
cd arena-fighters
pip install -e ".[dev]"
```

## Usage

### Train

```bash
python scripts/train.py --mode train
python scripts/train.py --mode train --timesteps 5000000
python scripts/train.py --mode train --checkpoint-dir ./my_checkpoints
```

Training logs go to `./tb_logs/` (viewable with `tensorboard --logdir tb_logs`).

### Watch

```bash
# Random agents (no checkpoint)
python scripts/train.py --mode watch

# Trained agent
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_final

# Fixed number of rounds
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_1M --rounds 10
```

Renders ASCII in the terminal. `Ctrl+C` to stop.

### Replay

```bash
python scripts/train.py --mode replay --episode replays/episode_100.json
```

## Architecture

```
src/arena_fighters/
├── config.py      # All constants: grid size, HP, damage, rewards, PPO hyperparams, platform layout
├── env.py         # PettingZoo ParallelEnv with tile physics, combat, observation building
├── network.py     # Custom SB3 features extractor (CNN for 6x20x40 grid + MLP for 6-dim vector)
├── self_play.py   # Opponent pool manager, wraps env for SB3 single-agent training
└── replay.py      # Episode logger (JSON) and playback
scripts/
└── train.py       # CLI entrypoint with train/watch/replay modes
```

### Observation Space

Dict observation with two components:

| Key | Shape | Contents |
|-----|-------|----------|
| `grid` | `(6, 20, 40)` | Platforms, own position, opponent position, own bullets, opponent bullets, facing direction |
| `vector` | `(6,)` | Own HP, opponent HP, x position, y position, shoot cooldown, melee cooldown |

### Action Space

| Action | ID | Description |
|--------|---:|-------------|
| Idle | 0 | Do nothing |
| Move left | 1 | Walk left |
| Move right | 2 | Walk right |
| Jump | 3 | Jump upward |
| Duck | 4 | Crouch (avoids horizontal bullets) |
| Shoot forward | 5 | Fire horizontal projectile |
| Shoot diag up | 6 | Fire diagonal-up projectile |
| Shoot diag down | 7 | Fire diagonal-down projectile |
| Melee | 8 | Close-range attack (higher damage) |

## Testing

```bash
pytest tests/ -v
```

43 tests covering the environment, network, self-play wrapper, renderer, and replay system.

## License

MIT

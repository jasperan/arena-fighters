# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

2D platform fighter where two RL agents train via self-play PPO. PettingZoo environment with SB3 training loop.

## Commands

```bash
# Setup
conda activate arena-fighters
pip install -e ".[dev]"

# Run tests (43 tests across env, network, self-play, renderer, replay)
pytest tests/ -v

# Run single test
pytest tests/test_env.py::test_name -v

# Train (headless, default 10M steps)
python scripts/train.py --mode train
python scripts/train.py --mode train --timesteps 5000000
python scripts/train.py --mode train --checkpoint-dir ./my_checkpoints

# Watch live (ASCII terminal, Ctrl+C to stop)
python scripts/train.py --mode watch
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_final
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_1M --rounds 10

# Replay a saved episode
python scripts/train.py --mode replay --episode replays/episode_100.json

# TensorBoard
tensorboard --logdir tb_logs
```

## Architecture

All source lives in `src/arena_fighters/`:

- `config.py` -- all constants (grid, HP, damage, rewards, hyperparams, platform layout)
- `env.py` -- PettingZoo ParallelEnv with tile physics, combat, observation building
- `network.py` -- custom SB3 features extractor (CNN for 6x20x40 grid + MLP for 6-dim vector)
- `self_play.py` -- opponent pool manager, wraps env for SB3 single-agent training
- `renderer.py` -- ASCII terminal renderer with speed control
- `replay.py` -- episode logger (JSON) and playback
- `scripts/train.py` -- CLI entrypoint

## Key Design Decisions

- Shared-weight self-play: one network plays both sides via observation mirroring
- Opponent pool (max 20 snapshots): 80% latest, 20% random older
- Dict observation space: `{"grid": Box(6,20,40), "vector": Box(6,)}` with `MultiInputPolicy`
- Milestone checkpoints auto-saved at 100K, 500K, 1M, 5M, 10M steps into `checkpoints/`
- Episode frames logged as JSON to `replays/` for post-hoc analysis

## Observation & Action Space

Grid channels (6x20x40): platforms, own position, opponent position, own bullets, opponent bullets, facing direction.

Vector (6,): own HP, opponent HP, x pos, y pos, shoot cooldown, melee cooldown.

Actions (9): idle, move left, move right, jump, duck, shoot forward, shoot diag-up, shoot diag-down, melee.

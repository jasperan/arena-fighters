# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

2D platform fighter where two RL agents train via self-play PPO. PettingZoo environment with SB3 training loop.

## Commands

```bash
# Setup
conda activate arena-fighters
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run single test
pytest tests/test_env.py::test_name -v

# Train (headless)
python scripts/train.py --mode train

# Watch live
python scripts/train.py --mode watch

# Replay
python scripts/train.py --mode replay --episode 100
```

## Architecture

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
- Dict observation space: {"grid": Box(6,20,40), "vector": Box(6,)} with MultiInputPolicy

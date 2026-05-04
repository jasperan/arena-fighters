# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

2D platform fighter where two RL agents train via self-play PPO. PettingZoo environment with SB3 training loop.

## Commands

```bash
# Setup
conda activate arena-fighters
pip install -e ".[dev]"

# Run tests (env, network, self-play, evaluation, renderer, replay, training metadata)
pytest tests/ -v

# Run single test
pytest tests/test_env.py::test_name -v

# Train (headless, default 10M steps)
python scripts/train.py --mode train
python scripts/train.py --mode train --timesteps 5000000
python scripts/train.py --mode train --checkpoint-dir ./my_checkpoints
python scripts/train.py --mode train --randomize-maps --map-choices classic,flat,split,tower
python scripts/train.py --mode train --reward-preset anti_stall
python scripts/train.py --mode train --curriculum map_progression

# Watch live (ASCII terminal, Ctrl+C to stop)
python scripts/train.py --mode watch
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_final
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_1M --rounds 10
python scripts/train.py --mode watch --map tower --rounds 3

# Replay a saved episode
python scripts/train.py --mode replay --episode replays/episode_100.json
python scripts/train.py --mode analyze --episode replays/episode_100.json

# Evaluate a checkpoint or random policy against a baseline
python scripts/train.py --mode eval --checkpoint checkpoints/ppo_final --opponent scripted --rounds 100 --seed 123
python scripts/train.py --mode eval --opponent scripted --rounds 20 --seed 123
python scripts/train.py --mode eval --agent-policy idle --opponent idle --rounds 1 --reward-preset anti_stall
python scripts/train.py --mode eval --opponent evasive --rounds 20 --seed 123
python scripts/train.py --mode eval --checkpoint checkpoints/ppo_final --opponent scripted --rounds 100 --randomize-maps --map-choices classic,flat,split,tower
python scripts/train.py --mode eval --opponent scripted --rounds 20 --eval-output-dir evals --eval-label random-vs-scripted

# Compare saved eval summaries
python scripts/train.py --mode compare --before evals/baseline.json --after evals/anti_stall.json --eval-output-dir evals --eval-label anti-stall-comparison

# Gate saved eval summaries with default regression thresholds
python scripts/train.py --mode gate --before evals/baseline.json --after evals/anti_stall.json --eval-output-dir evals --eval-label anti-stall-gate

# Run a compact baseline suite
python scripts/train.py --mode suite --suite-opponents idle,scripted,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label baseline-suite
python scripts/train.py --mode suite --agent-policy idle --suite-opponents idle --suite-maps flat --rounds 1 --reward-preset anti_stall --eval-output-dir evals --eval-label idle-suite

# Rank checkpoints by baseline-suite score
python scripts/train.py --mode rank --checkpoint-dir checkpoints --suite-opponents idle,scripted,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label checkpoint-rank
python scripts/train.py --mode rank --rank-checkpoints checkpoints/ppo_1M.zip,checkpoints/ppo_final.zip --suite-opponents scripted --suite-maps classic --rounds 20 --rank-head-to-head
python scripts/train.py --mode rank_gate --rank-summary evals/checkpoint-rank.json --eval-output-dir evals --eval-label checkpoint-rank-gate
python scripts/train.py --mode promotion_audit --checkpoint-dir checkpoints --suite-opponents idle,scripted,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label checkpoint-promotion
python scripts/train.py --mode audit_summary --audit-summary evals/checkpoint-promotion.json
python scripts/train.py --mode analyze --episode replays/episode_0100.json --eval-output-dir evals --eval-label replay-episode-0100
python scripts/train.py --mode analyze --replay-dir replays --replay-samples-per-bucket 2 --eval-output-dir evals --eval-label replay-sample
python scripts/train.py --mode artifact_index --artifact-dir evals --recursive-artifacts --eval-output-dir evals --eval-label artifact-index
python scripts/train.py --mode strategy_report --artifact-dir evals --recursive-artifacts --eval-output-dir evals --eval-label bad-strategy-report
python scripts/train.py --mode long_run_manifest --run-id arena-manual-001 --timesteps 5000000 --eval-output-dir evals --eval-label arena-manual-001-plan
python scripts/train.py --mode long_run_check --promotion-audit-summary evals/promotion.json --strategy-report-summary evals/strategy-report.json --artifact-index-summary evals/artifact-index.json --long-run-required-maps classic,flat,split,tower --long-run-min-eval-episodes 240 --long-run-min-map-score 0.0 --eval-output-dir evals --eval-label long-run-check
python scripts/train.py --mode long_run_status --artifact-dir evals --eval-output-dir evals --eval-label long-run-status
python scripts/reward_shaping_smoke.py
python scripts/train_eval_smoke.py

# TensorBoard
tensorboard --logdir tb_logs
```

Treat Stable-Baselines3 checkpoints as trusted executable artifacts. Only use
`--checkpoint`, `--checkpoint-dir`, or `--rank-checkpoints` with local or
trusted checkpoints, and verify external checkpoint digests before loading them.

Generated long-run launchers persist `preflight.exitcode`, `train.exitcode`,
`promotion-audit.exitcode`, and `long-run-check.exitcode`; early preflight or
training failures still write a final artifact index before the launcher exits.
Long-run manifests record a compact git source snapshot with commit, branch,
dirty flag, and a `git status --short` sample for reproducibility.
Manifest generation writes both the full launcher and a `.preflight.sh` launcher
that runs only the tiny smoke path without entering expensive training.
Missing promotion, strategy-report, or artifact-index files are passed into the
verifier as placeholder paths so failures are recorded in `long_run_check`
instead of aborting during shell artifact resolution.
Generated launcher commands write stdout/stderr to `.out` sidecars that
artifact indexes summarize as command logs.
Use `long_run_status` to recursively inspect generated manifests and run
directories without starting training; it reports whether the latest launcher
has produced a passing long-run-check artifact and prints the next launcher
commands when its preflight or full run has not run and its source snapshot still
matches the current clean checkout. The status artifact also includes
`missing_evidence` for machine-readable blocker reporting and
`source_safe_to_launch`/`source_stale_reasons` for stale-manifest detection.

## Architecture

All source lives in `src/arena_fighters/`:

- `config.py` -- all constants (grid, HP, damage, rewards, hyperparams, map layouts)
- `env.py` -- PettingZoo ParallelEnv with tile physics, combat, observation building
- `evaluation.py` -- checkpoint/baseline matchups with aggregate metrics
- `network.py` -- custom SB3 features extractor (CNN for 6x20x40 grid + MLP for 6-dim vector)
- `self_play.py` -- opponent pool manager, wraps env for SB3 single-agent training
- `replay.py` -- episode logger (JSON), playback metadata, and replay analysis
- `scripts/train.py` -- CLI entrypoint
- `docs/experiments.md` -- repeatable eval/save/compare/gate/suite workflow

## Key Design Decisions

- Shared-weight self-play: one network plays both sides via observation mirroring
- Named maps: classic, flat, split, tower; `--randomize-maps` samples one on each reset
- Curriculum: `map_progression` stages from flat/default rewards to full map pool/anti-stall rewards through training callback updates
- Opponent pool (max 20 snapshots): sample frozen historical snapshots, 80% latest and 20% random older; reset info and training logger expose latest-vs-historical sampling telemetry
- Checkpoints get companion `.meta.json` files with map settings, reward config, and active curriculum stage
- Eval JSON includes average cumulative rewards and behavior diagnostics for idle rate, action spam, no-damage episodes, low-engagement episodes, and damage events
- Evaluation winner inference treats timeouts as draws and knockouts by terminal HP; shaped rewards do not create timeout wins
- Eval and suite configs include active curriculum metadata plus checkpoint metadata when a companion file exists
- Eval, suite, rank, comparison, gate, rank-gate, promotion-audit, audit-summary, artifact-index, strategy-report, long-run-check, and replay-analysis JSON include an `artifact` type/schema marker
- Env infos include per-step and cumulative combat event counters under `events` and `episode_events`
- Built-in eval opponents: random, idle, scripted, aggressive, evasive
- Eval summaries can be persisted with `--eval-output-dir`; `evals/` is ignored
- Compare mode reports metric deltas between two saved eval summaries
- Gate mode exits non-zero when default comparison guardrails fail
- Suite mode evaluates one policy/checkpoint against multiple opponents and maps
- Rank mode discovers checkpoints and sorts them by baseline-suite score with no-damage and low-engagement penalties
- Rank mode can include head-to-head checkpoint standings and Elo-style ratings with `--rank-head-to-head`
- Rank gate checks the top-ranked checkpoint against baseline and optional head-to-head promotion thresholds before selection
- Promotion audit runs rank plus rank gate in one command and saves rank/gate/audit artifacts when `--eval-output-dir` is set; add `--audit-include-nested` to embed full rank/gate JSON in the audit artifact
- Audit summary reads saved promotion-audit JSON and prints a compact candidate/pass-fail/failure/path summary
- Artifact index scans saved JSON, `.exitcode`, `.sh`, and `.out` artifacts and emits a lightweight manifest with type counts, compact summaries, command-log tails, and links between related artifacts
- Strategy report scans saved eval/rank/audit artifacts, flags all-draw, no-damage, low-engagement, idle, or action-spam behavior, and ranks weakest map-opponent matchups
- Long-run check validates promotion-audit, strategy-report, and artifact-index artifacts against documented promotion criteria
- `scripts/reward_shaping_smoke.py` runs deterministic idle-vs-idle reward-shaping eval/compare/suite/report/index artifacts in `/tmp`
- `scripts/train_eval_smoke.py` runs tiny train-to-checkpoint-to-suite/promotion/report/index wiring checks in `/tmp`
- Reward presets: `default` and `anti_stall`; anti-stall adds a no-damage timeout penalty, and eval behavior diagnostics should be used to judge reward-shaping experiments
- `docs/long_run_protocol.md` contains the conservative real-compute training and promotion workflow; do not run it by default
- Dict observation space: `{"grid": Box(6,20,40), "vector": Box(6,)}` with `MultiInputPolicy`
- Milestone checkpoints auto-saved at 100K, 500K, 1M, 5M, 10M steps into `checkpoints/`
- Episode frames logged as JSON to `replays/` with top-level map and event metadata for post-hoc analysis

## Observation & Action Space

Grid channels (6x20x40): platforms, own position, opponent position, own bullets, opponent bullets, facing direction.

Vector (6,): own HP, opponent HP, shoot cooldown, melee cooldown, vertical velocity, ducking state.

Actions (9): idle, move left, move right, jump, duck, shoot forward, shoot diag-up, shoot diag-down, melee.

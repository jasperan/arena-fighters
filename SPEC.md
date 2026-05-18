# Autonomous Improvement Loop Spec

## Goal

Create a re-runnable `/goal` contract that lets Codex improve this repository
autonomously in bounded, reviewable slices. Each run should research the current
state, update a ranked backlog, implement one to three high-value changes, and
leave verification evidence plus the next recommended slice.

## Project Context

Arena Fighters is a Python 3.12 RL project for a two-agent platform fighter
using PettingZoo-style environments and Stable-Baselines3 PPO. The current
repo has strong tests and long-run artifact tooling, with the largest surfaces
in `scripts/train.py`, `src/arena_fighters/evaluation.py`, replay handling, and
the training/evaluation smoke scripts.

## Improvement Lanes

- Maintainability: continue reducing `scripts/train.py` by moving cohesive
  behavior behind tested module boundaries.
- Security and robustness: harden local artifact, replay, checkpoint metadata,
  and manifest ingestion without weakening existing trust checks.
- Reproducibility: make dependency audits and CI validation runnable from a
  fresh checkout.
- Experiment quality: improve cheap smoke diagnostics, artifact summaries, and
  promotion gates before spending real compute.
- Documentation: keep long-run protocol and README examples aligned with CLI
  behavior.

## Non-Goals

- Do not run expensive real training as part of ordinary improvement loops.
- Do not rewrite the environment, model architecture, or reward system without
  first adding focused regression coverage and cheap evaluation evidence.
- Do not add dependencies unless a small, justified slice cannot be completed
  with the standard library or current project dependencies.
- Do not commit or push unless the launched prompt explicitly asks for it.

## Done When

- `docs/autonomous_improvement_backlog.md` exists and reflects the current
  ranked improvement queue.
- `GOAL.md` contains a measurable, finite `/goal` contract for autonomous
  improvement runs.
- The first selected slice is implemented with regression tests.
- Relevant focused tests pass.
- Full `pytest`, `python -m compileall scripts src tests`, and
  `git diff --check` pass before completion.
- The final report identifies changed files, verification evidence, and the
  next recommended autonomous slice.

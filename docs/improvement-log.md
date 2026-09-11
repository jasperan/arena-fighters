# Improvement Log

Cycle-by-cycle record of strategy and training improvements. A cycle is a
**GAIN** when at least one is true and evidenced by artifacts: a new built-in
opponent archetype, a trained-policy gain over the prior best, a config-derived
behavior change, or a new replay behavior cluster. Everything else is a
no-gain cycle. The loop stops after 3 consecutive no-gain cycles.

## Cycle 1 — 2026-09-11 — make self-play actually self-play

**Lanes:** trained policy gains (critical infrastructure), config-derived behaviors.

**Changes**

1. `fix(self_play)`: `SelfPlayWrapper` now keeps the caller-supplied opponent
   pool instead of silently substituting a private empty one (`opponent_pool or
   OpponentPool()` was falsy for empty pools). Before: 0 opponent samples in a
   real 6144-step PPO run; after: 53. The opponent pool had never functioned
   during training.
2. `fix(env)`: double knockouts now pay symmetric draw rewards to both agents.
   Before: agent_0 −9.5 vs agent_1 +10.5 mean reward per episode (+20.0 gap).
   After: 0.000 gap on all maps.
3. `fix(config)`: `classic` map made mirror-symmetric (low platform `(35,38)`,
   centered mid platform `(16,23)`). Before: mirrored action streams diverged
   on classic at tick 12. After: exact mirror invariance on all maps.

**Evidence**

- `uv run pytest -q` → 298 passed (8 new regression tests).
- `uv run python scripts/train.py --mode train --timesteps 20000 …` → checkpoint
  + sidecar SHA-256 verified + trust manifest.
- 4-case checkpoint-trust probe (refuse / valid / tampered / override) all correct.
- Pool sampling probe: `latest_samples 0 → 53` over an identical 6144-step learn.
- Reward symmetry probe: `+20.0 → 0.000` mean gap across 240 scripted episodes.
- Mirror invariance probe: `first_mirror_break t=12 → None` on all four maps.
- `uv run python scripts/smoke_suite.py` → exit 0, all three smoke artifacts pass.

**Verdict: GAIN** (trained-policy lane: self-play opponents are live for the
first time; config lane: symmetric double-KO rewards + symmetric classic map).

**Next candidate**

1. Visuals: PNG/GIF frame renderer for episodes so trained behavior can be
   inspected directly, then iterate on the aesthetics/layout of the arena.
2. Run a real long-enough training run (≥110k steps) and verify snapshot
   telemetry shows latest + historical opponent samples; evaluate via suite/rank.
3. Add new scripted opponent archetypes (e.g., zoner that keeps distance,
   camper that holds high platforms) and measure them against the current pool.

## Unresolved / carried forward

- Long-run promotion artifacts unproven at real scale (no long run has ever
  completed in this repo before this session).
- Audit LOW items from `AUDIT-2026-09-10.md` remain open (assert guard,
  redaction scope, broad artifact `except`, env `lru_cache`, no CI/lint config,
  `scripts/train.py` monolith).
- No CI workflow; tests run locally only.

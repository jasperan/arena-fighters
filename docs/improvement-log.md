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

## Cycle 2 — 2026-09-11 — reviewable match visuals

**Lane:** tooling for all strategy lanes (vision-reviewable evidence).

**Changes**

1. `feat(render)`: new `arena_fighters.render` (Pillow) plus
   `scripts/render_episode.py`. Renders any env state or saved replay frame to
   PNG: gradient arena with vignette, faint cell grid, solid platform runs with
   exposed top rims, fighter silhouettes (highlight, legs, facing chevron,
   weapon stub), glow-comet bullets with trails, and an HP/tick/score HUD. The
   CLI writes PNG frames, an animated GIF, and a sampled contact sheet, and
   accepts built-in policies, trusted checkpoints, and `--replay` inputs.
2. Auto-crops empty sky above the highest platform (platforms used to occupy
   only the bottom half of the frame).

**Evidence**

- `uv run pytest -q` → 305 passed at the time (12 renderer tests).
- Contact sheets reviewed directly; frame-by-frame inspection caught and fixed
  a real defect (an `lru_cache` on the background image made successive frames
  draw over earlier frames — regression test added).
- `scripts/render_episode.py --agent-policy zoner --opponent aggressive` shows
  the zoner kiting and winning with full HP at tick 18.

**Verdict: GAIN** (enables fast visual review of every future strategy change).

## Cycle 3 — 2026-09-11 — two new scripted archetypes

**Lane:** new scripted archetypes (+ config-derived behavior).

**Changes**

1. `ZonerPolicy`: holds a firing lane, retreats inside `preferred_min`, only
   melees when pinned against a wall.
2. `CamperPolicy`: takes and holds an elevated platform and shoots from it.
   Required understanding the jump physics: jumps rise in decreasing velocity
   steps (h + h-1 + ... + 1) and treat solid tiles as ceilings, so a platform
   cannot be mounted by jumping from directly underneath. The camper walks
   beside the target span, jumps, steers onto it while airborne, latches the
   climb target for the duration of the jump, turns to face the opponent by
   stepping within the platform, and holds station instead of chasing.

**Evidence** — 25 rounds per pairing across classic/flat/split/tower:

| matchup | zoner win rate (as agent_0) | notes |
|---|---|---|
| vs scripted | 1.00 | wins without taking damage |
| vs aggressive | 1.00 | kites the rush |
| vs idle | 1.00 | |
| vs random | 0.72 | |
| vs camper | 0.52 | |
| vs evasive | 0.00 | 100% draws (evasive denies every lane) |

Camper reaches and holds elevation on classic/split/tower (189-196 of 200
ticks) with 38-39 shots per episode; ground-only on flat. Tests cover policy
construction, zoner spacing/firing, a full zoner-vs-scripted no-damage win,
and camper leaving the ground floor.

**Verdict: GAIN** (zoner is the strongest scripted baseline in the repo; camper
adds the first platform-control archetype).

## Cycle 4 — 2026-09-11 — anti-stall training run (in progress)

**Lane:** trained policy gains.

The first trained checkpoint in repo history (`checkpoints/ppo_final`, 150k
steps, default rewards, classic only) collapsed to **duck on 100% of ticks**
across all eight suite matchups, winning 0 of 40 episodes (draws vs idle and
evasive, losses vs scripted and aggressive). Holding duck blocks horizontal
bullets, and no-damage draws cost only -1 under the default preset, so ducking
is a local optimum worth escaping.

Cycle 4 runs 1M steps with `--reward-preset anti_stall` on classic+flat, and
will be evaluated with the extended suite (now including zoner and camper) and
ranked against the 150k checkpoint. Status: training in progress at the time of
writing; results appended below when complete.

## Unresolved / carried forward

- Long-run promotion artifacts unproven at real scale (no long run has ever
  completed in this repo before this session).
- Audit LOW items from `AUDIT-2026-09-10.md` remain open (assert guard,
  redaction scope, broad artifact `except`, env `lru_cache`, no CI/lint config,
  `scripts/train.py` monolith).
- No CI workflow; tests run locally only.

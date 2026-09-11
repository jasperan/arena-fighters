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

**Evidence** — 25 rounds per pairing across classic/flat/split/tower.
Note (added in cycle 7): these pairings are all deterministic policies and the
environment had no spawn variety at the time, so every seeded round replayed
the same episode; read the numbers as the outcome map of the fixed opening,
not as sampled win probabilities. The conclusions (zoner dominant, camper
elevation-dependent, evasive denying every lane) reproduce with jittered
spawns.

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

## Cycle 4 — 2026-09-11 — anti-stall training run (complete)

**Lane:** trained policy gains.

The first trained checkpoint in repo history (`checkpoints/ppo_final`, 150k
steps, default rewards, classic only) collapsed to **duck on 100% of ticks**
in every matchup, winning 0 of 84 episodes and dealing 0 damage; holding duck
blocks horizontal bullets and a no-damage draw costs only -1, so ducking was a
local optimum the default preset never punished.

Cycle 4 ran 1M steps with `--reward-preset anti_stall` on classic+flat
(`checkpoints/antistall-1m/`, seed 21, 254-330 fps). Both checkpoints were
re-evaluated under identical settings (7 opponents x 2 maps x 6 rounds) after
the redo fixes:

| checkpoint | mean win rate | losses | action mix | damage | draws |
|---|---|---|---|---|---|
| `ppo_final` 150k, default rewards | **0.000** | 4 matchups | duck 1.00 | 0 | 10 |
| `antistall-1m/ppo_final` | **0.845** | 0.17 (random@flat) | duck 0.07-0.60, shoot 0.33-0.99 | 60/matchup | 1 (evasive) |

* The anti-stall policy kills every scripted archetype in 10-33 ticks; only
  `evasive` remains a 500-tick stalemate, which is baseline-specific (zoner is
  also 0.00 wins vs evasive).
* League telemetry confirms the cycle-1 pool fix in a real run: pool 10,
  24515 latest / 5429 historical samples, historical rate 18.1% (target 20%),
  13 checkpoints saved.
* Head-to-head between the run's own checkpoints (100K vs 500K vs 1M, both
  sides, both maps, 32 episodes each) is **100% draws with zero damage**, and
  the action distribution contains essentially no left/right movement. The
  policy is a stationary trench gunner: it out-trades anything that walks into
  its fire lane and cannot beat a copy of itself. The rendered contact sheet
  (`renders/antistall-1m-vs-scripted/`) shows the pattern directly.

**Verdict: GAIN** (0.000 -> 0.845 mean win rate; the reward preset escapes the
duck collapse), with a clearly identified next target: the policy never learned
positioning, so mirror matches stall.

## Cycle 5 — 2026-09-11 — redo pass: three further defects fixed

**Lane:** correctness foundation for every other lane.

An independent re-verification from first principles (probes written against
the specification, not the implementation) found three more defects, all fixed
and regression-tested; details and evidence in
`docs/verification-2026-09-11.md` ("Redo pass"):

1. **D4 bullets skipped tiles between ticks** -- a stationary opponent at an
   even offset from the shooter was literally unhittable, and shots crossed the
   far edge of platforms. Hit rate 5/9 offsets -> 9/9 on all four maps.
2. **D5 tower platforms could be jumped through from below** -- the destination
   row was free while the platform sat in between; `classic` blocked the same
   jump, so physics was map-dependent.
3. **D6 the self-play mirror corrupted bullet geometry** -- the transform
   flipped the grid and then swapped own/opponent channels, cancelling the
   reflection for positions while double-transforming bullets; 9398 of 53428
   enumerated geometries blinded the frozen opponent to a bullet within three
   tiles, and its movement actions were never un-mirrored.

Re-verification after the fixes: 359 tests pass; 30/30 independent probe
assertions; round-robin tournament over all seven built-ins re-run; smoke suite
3/3 green. The cycle-4 checkpoints predate these fixes and are labelled as
such.

**Verdict: GAIN** (removes three silent gameplay/training defects; the bullet
sweep also removes the stationary "safe" offsets that made trench play cheap).

## Cycle 6 — 2026-09-11 — four-map anti-stall run (in progress)

**Lane:** trained policy gains across the full map pool.

With physics and mirroring corrected, the next run repeats the successful
cycle-4 recipe over all four maps (`--randomize-maps --map-choices
classic,flat,split,tower --reward-preset anti_stall --timesteps 1000000`,
checkpoint dir `checkpoints/antistall-4map-1m`, seed 33). Goal: a policy that
keeps the 0.845 win rate while generalizing across platform geometry, which is
also the prerequisite for the next improvement: a mixed training league that
includes the scripted archetypes (zoner, camper, evasive) instead of relying on
self-snapshots alone, so positioning and anti-camper play are under pressure.

Status: launched; results appended below when complete.

## Cycle 7 — 2026-09-11 — opponent diversity and spawn variety

**Lane:** trained policy gains (new strategy pressure) + evaluation rigor.

Two changes, both driven by measurements from the redo pass and cycle 4:

1. **Mixed training league.** Training only ever faced frozen snapshots of
   itself, so it converged on a stationary trench gunner (checkpoint vs
   checkpoint = 100% draws, no movement actions). `--scripted-opponents
   zoner,camper,evasive --scripted-opponent-prob 0.35` now puts a built-in
   archetype on the opponent side for that fraction of episodes. Scripted
   policies read the environment directly, so they receive raw observations and
   their actions are applied unchanged; frozen snapshots keep the mirrored
   observation path. Telemetry: reset info, TensorBoard
   (`self_play/scripted_opponent_*`), `[Snapshot]` log line, and checkpoint
   metadata. Verified end-to-end with a 40k-step run: **308 scripted episodes
   over 20 rollouts (~35%, the configured rate), split camper 103 / evasive
   108 / zoner 97**, with pool sampling still active.
2. **Symmetric spawn jitter.** Every seeded episode previously started from
   the same columns with identical physics, so a deterministic pairing replayed
   one episode no matter how many rounds were requested: win rates in suite,
   rank, and tournament artifacts were deterministic scans over (map, opponent)
   pairs with no statistical content. `reset(seed)` now shifts both spawns by a
   shared offset in `[-2, 2]` (default `spawn_jitter=2`), keeping the opening
   mirror symmetric, and unseeded auto-resets during training draw a fresh
   offset each episode. Measured: scripted vs aggressive now yields 3 distinct
   outcomes over 6 seeds (ticks 9/10) instead of one repeated episode.

Re-measured under jittered spawns (7 opponents x 2 maps x 6 rounds): the
cycle-4 policy still scores **0.845** (robust to varied openings) while the
150k default-reward checkpoint stays at **0.000**.

**Verdict: GAIN** (opponent diversity attacks the positioning gap; spawn
variety fixes the statistics of every future evaluation).

## Cycle 8 — 2026-09-11 — movement diagnostics in evaluation and strategy report

**Lane:** training/eval tooling for strategy quality.

The existing degeneracy checks covered stuck actions (dominant-action rate) and
non-engagement (no-damage/low-engagement rates), but nothing observed movement.
The cycle-4 policy therefore scored 0.845 while never leaving its spawn, and the
tooling had no opinion about it.

`run_episode` now records `stand_still_rate` (fraction of ticks whose column did
not change) and `travel_distance`; `evaluate_matchup` aggregates both into the
behavior block at match and per-map level, and the strategy report raises
`agent_0_stand_still_rate_above_threshold` with a new tunable
(`--strategy-max-stand-still-rate`, default 0.95) that flows into the long-run
manifest alongside the other thresholds.

Measured: scripted-vs-scripted 0.22 stand-still with 8 tiles of travel per
episode; the cycle-4 checkpoint 0.83 with 2 tiles; a hand-written never-moving
fighter 1.00 with 0 tiles. 366 tests pass, smoke suite 3/3.

**Verdict: GAIN** (the loop can now see the failure mode it just produced).

## Unresolved / carried forward

- The trained policy has no positioning behaviour: checkpoint-vs-checkpoint is
  a pure stalemate and the action distribution has almost no movement. Next
  lever is opponent diversity during training (mixed league with scripted
  archetypes) plus spawn/position variety.
- `evasive` denies every lane to every policy tried so far (zoner, camper and
  both checkpoints all draw); either it is a valid ceiling on the current
  action set or the reward preset needs an anti-turtle term -- worth a cycle.
- Cycle-4 checkpoints were trained under the pre-fix physics/mirror; their
  results are valid reward-preset evidence but superseded as gameplay
  baselines.
- Long-run promotion artifacts unproven at real scale (no long run has ever
  completed in this repo before this session).
- Audit LOW items from `AUDIT-2026-09-10.md` remain open (assert guard,
  redaction scope, broad artifact `except`, env `lru_cache`, no CI/lint config,
  `scripts/train.py` monolith).
- No CI workflow; tests run locally only.

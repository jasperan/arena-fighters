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

## Cycle 6 — 2026-09-11 — four-map anti-stall run: NO GAIN

**Lane:** trained policy gains across the full map pool.

Ran 1M steps on all four maps (`checkpoints/antistall-4map-1m`, seed 33) with
the cycle-4 recipe. Result: **the run collapsed**. Evaluated with the extended
suite over all four maps (7 opponents x 4 maps x 4 rounds = 112 episodes,
jittered spawns):

* final checkpoint: **0.000 mean win rate**, stand-still rate 1.000, action
  distribution **duck 1.00** -- it loses to scripted and camper, draws (500-tick,
  no damage) against zoner and idle, and never deals damage.
* five checkpoints sampled across the run (100K, snap_300, 500K, snap_450,
  final) all score **0.00**, so the degeneracy is not just a late-training
  artifact.
* the same evaluation run on the cycle-4 (classic+flat) policy scores **0.812
  across all four maps** -- the two-map policy generalizes to split and tower
  better than the model trained on them.

Monitoring lesson: the action telemetry added in cycle 4b reported a *healthy*
sampled policy at the end of this run (entropy 0.837, dominant action `shoot`
0.383) while the deterministic policy is 100% duck. Sampled-action entropy does
not detect a degenerate greedy policy; suite/rank evaluation of the actual
checkpoint does, which is what the promotion tooling exists for.

**Verdict: NO GAIN** (1 of 3 allowed before the loop stops). The four-map
recipe is retired; map variety is being reintroduced through a curriculum
rather than from step 0.

*Corrected in cycle 14:* the 0.000 figure is the greedy mode; with stochastic
sampling the same checkpoint scores 0.609 across all four maps. It is still a
no-gain (0.609 < 0.729 cycle-4 / 0.844 cycle-9 on the comparable metric), but it
is greedy-degenerate rather than collapsed.

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

*Corrected in cycle 14:* in stochastic sampling this run is the strongest policy
produced so far -- **0.844** mean win rate, with the evasive stalemate solved
(0.00 -> 1.00), versus 0.729 for cycle 4. It had been graded NO GAIN from a
greedy-mode suite that reported 0.798 against cycle 4's 0.845.

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

## Cycle 9 — 2026-09-11 — mixed-league run (GAIN)

**Lane:** trained policy gains under the fixed foundations.

Cycle 6 showed that four-map randomization from step 0 collapses, so the next
run returns to the proven two-map recipe and adds the cycle-7 pressure instead:
anti-stall rewards, classic+flat, symmetric spawn jitter, and a mixed league of
zoner/camper/evasive at `--scripted-opponent-prob 0.35`
(`checkpoints/mixed-league-1m`, seed 55). The expectation is a policy that keeps
the cycle-4 win rate while finally learning to reposition, which the stand-still
diagnostic (cycle 8) can now measure directly.

**Result: GAIN (re-graded in cycle 14).** This entry was left unfinished when
the loop moved on; measured over 8 opponents x classic+flat x 6 rounds it is
stochastic **0.844** mean win rate (greedy 0.798) with the evasive stalemate
solved (0.00 -> 1.00) -- the best policy of the whole loop. Its 500K snapshot
scored **0.969** (win 0.958) on the six-opponent rank suite, above the final
checkpoint's 0.917. Artifacts: `evals/*cycle9-stochastic.json`,
`evals/*cycle9-rank-stochastic.json`, `evals/*cycle9-rank-gate.json`.

## Cycle 10 — 2026-09-11 — rusher archetype: punishing turtling

**Lane:** new scripted archetypes.

Cycle 9 was a no-gain because the mixed league did not actually pressure the
turtling strategy: every existing archetype fights at range, and ducking blocks
horizontal bullets, so standing still and trading fire stayed close to optimal
(stand-still rate stayed 1.000 across 1M steps).

The `rusher` archetype closes the gap with a **duck-march** -- a duck covers two
ticks (`duck_duration`), so alternating duck and move keeps the body low on
every tick while still advancing a tile every other tick -- then finishes with
melee, which ducking does not block. Outbound fire is blocked the whole way;
the counter-play is a diagonal shot, since ducking only stops `dy == 0`
bullets. It also steps out from under a platform before jumping when the
opponent is above, reusing the ceiling lesson from the camper work.

**Evidence**

* Head-to-head against the strongest trained policy: the cycle-4 1M checkpoint
  (`checkpoints/antistall-1m/ppo_final`) **loses 0-6 on classic and 0-6 on
  flat** to the rusher (`evals/*rusher-vs-1m.json`), while still beating
  scripted 1.00 and drawing evasive -- the rusher is a genuine counter to the
  policy that every recipe so far has converged to.
* Round robin over all eight built-ins, both sides, all four maps
  (`/tmp/reverify/tournament-rusher.json`): rusher **win 0.33 / loss 0.01 /
  draw 0.66** -- it almost never loses. It beats idle 1.00, splits 0.50/0.50
  with zoner, and forces mutual knockouts (20-tick draws) against
  scripted/aggressive/camper. Evasive still outruns it.
* Regression tests: melee on contact, duck-march advance under cover, a
  stationary duck/shooter losing on four seeds, and a fast win over idle.

**Verdict: GAIN** (new archetype + a league counter to the recurring degenerate
strategy which no previous opponent punished).

## Cycle 11 — 2026-09-11 — mixed league with the rusher (NO GAIN)

**Lane:** trained policy gains.

Next run repeats the cycle-9 recipe (anti-stall, classic+flat, spawn jitter)
and adds the rusher to the mixed league:
`--scripted-opponents rusher,zoner,camper,evasive --scripted-opponent-prob 0.4`
(`checkpoints/mixed-league-rusher-1m`, seed 61). Success is measured with the
cycle-8 diagnostic: a stand-still rate well below 1.000 together with a win
rate near the cycle-4 level would mean the policy finally learned to reposition
instead of trading from its spawn.

**Result: NO GAIN.** Greedy **0.000** / stochastic **0.719** (8 opponents,
classic+flat, 6 rounds) against cycle 9's 0.798 / 0.844. Partial gain: first wins
against the rusher (0.00 -> 0.17). Regression: camper on classic 1.00 -> 0.17.
Stand-still 1.000 in both modes. Artifacts: `evals/*cycle11-greedy.json`,
`evals/*cycle11-stochastic.json`.

## Cycle 12 — 2026-09-11 — positional replay analysis: the trench pattern in numbers

**Lane:** novel replay behaviors.

Action histograms could not distinguish holding spawn from repositioning, so
replay frames are now summarized positionally (`summarize_spatial_behavior`,
embedded in every `analyze_replay` artifact): lateral range, travel distance,
`stand_still_rate` (same definition as the evaluation diagnostic),
`spawn_camp_rate` (frames within two columns of the start), `elevated_rate`,
and a per-column occupancy histogram.

**Discovery** — replay directories of the three runs, first 40 episodes each,
agent_0:

| run | stand still | travel/ep | spawn camp | elevated | median lateral range |
|---|---|---|---|---|---|
| `antistall-1m` (cycle 4) | 0.901 | 4.0 | 0.696 | 0.105 | 3 |
| `antistall-4map-1m` (cycle 6, collapsed) | 0.968 | 2.4 | 0.904 | 0.071 | 1 |
| `mixed-league-1m` (cycle 9) | 0.873 | 3.5 | 0.746 | 0.361 | 2 |
| `mixed-league-rusher-1m` (cycle 11, 7 replays, mid-run) | **0.750** | **9.6** | **0.369** | **0.485** | **4** |

The gradient quantifies the degeneracy that the win-rate tables hid: the
collapsed four-map policy is the most spawn-bound (0.968 stand-still, 90% of
frames within two columns of spawn, median lateral range 1 tile), while the run
exposed to the rusher is already moving 2.4x further per episode and camping
37% of frames instead of 75-90%. Cycle 11's success criterion (movement under
pressure) is therefore measurable from replays as well as from evaluation.

**Verdict: GAIN** (new replay lane: a positional metric that separates policies
the action histograms call identical).

## Cycle 13 — 2026-09-11 — entropy bonus as an explicit training knob

**Lane:** config-derived behaviors (training recipes).

Cycle 6 showed sampled-action entropy (0.837) does not detect a degenerate
greedy policy: the four-map run logged a healthy-looking distribution while
every deterministic evaluation came out 100% duck. The entropy bonus was
implicitly zero and not settable from the CLI, so the obvious lever for that
failure mode could only be changed by editing the dataclass.

`TrainingConfig.ent_coef` (default 0.0) is now plumbed to PPO and exposed as
`--ent-coef`, validated as non-negative, and recorded in checkpoint metadata so
runs remain comparable. Verified end-to-end with a 4,096-step run
(`--ent-coef 0.02`): the flag reaches PPO and `ent_coef: 0.02` appears in
`ppo_final.meta.json`. Tests cover the default, the override, and the metadata
field. 375 tests pass.

**Verdict: GAIN** (a recorded, testable knob for the collapse mode that cost
cycles 6 and 9, and an input for the planned ablation).

## Cycle 14 — 2026-09-11 — dual-mode evaluation: greedy is a bad promotion metric

**Lane:** training/eval tooling (correct measurement).

Cycles 6, 9 and 11 were judged with `--mode suite`, which evaluates checkpoints
with `deterministic=True`. Re-running the same checkpoints with `--stochastic`
(the flag already existed; the loop had never used it) overturns two of those
verdicts. Same opponents, maps and rounds, stochastic sampling:

| run (recipe) | greedy mean | **stochastic mean** | notes |
|---|---|---|---|
| cycle 4 — 2 maps, anti-stall | 0.845 | **0.729** | loses to evasive 0.00 |
| cycle 6 — 4 maps, anti-stall | 0.000 | **0.609** | not a collapse; mediocre, greedy-degenerate |
| cycle 9 — 2 maps, mixed league | 0.798 | **0.844** | beats evasive 1.00 |
| cycle 11 — 2 maps, mixed league + rusher | 0.000 | **0.719** | first wins vs rusher (0.17); camper-classic 0.17 |

Consequences:

* **Cycle 9 is re-graded from NO GAIN to GAIN**: in the mode where these
  policies actually behave, the mixed league raised the mean from 0.729 to
  0.844 and solved the evasive stalemate (0.00 -> 1.00), which no other recipe
  has done.
* **Cycle 11 stands as NO GAIN** (0.719 < 0.844) with a partial gain (rusher
  0.00 -> 0.17) and a new weakness (camper on classic 1.00 -> 0.17).
* **Cycle 6 stands as NO GAIN** but was mischaracterised: the four-map policy is
  greedy-degenerate, not useless (0.609 stochastic across all four maps).
* Deterministic evaluation of these PPO checkpoints measures the mode of a
  distribution the policy does not actually act from, and the mode can be the
  worst action in the distribution. Promotion decisions need both modes, or
  stochastic alone.
* Tooling gap found and fixed: suite artifacts did not record which mode they
  used. `suite_config.policy_sampling` is now `"greedy"` or `"stochastic"` for
  every suite written, verified with a fresh 2-round run.
* Still true in every mode: `stand_still_rate` is ~1.000 for all four
  checkpoints (travel 0.0-0.02 tiles/episode). The positioning problem is not a
  sampling artifact -- it is real and unsolved.

**Verdict: GAIN** (measurement correction worth 0.845 of misjudged policy
quality, plus provenance in every suite artifact).

## Cycle 15 — 2026-09-11 — entropy-regularised mixed-league run (NO GAIN)

**Lane:** trained policy gains.

Cycle 14 established that the best recipe so far is cycle 9's (anti-stall,
classic+flat, spawn jitter, mixed league of zoner/camper/evasive at 0.35), whose
stochastic score is 0.844, and that every checkpoint since cycle 4 has a
degenerate *greedy* mode (0.000-0.798) even when sampling plays well. Cycle 13
added the entropy coefficient that exists to counter exactly that.

This run repeats the cycle-9 recipe and adds `--ent-coef 0.01`
(`checkpoints/entcoef-mixed-1m`, seed 71). Success criteria, both measured:
stochastic mean win rate >= 0.844 (holding the best score) **and** a greedy mean
win rate well above 0.000 (the mode is no longer degenerate). Stand-still and
replay-spatial metrics are reported alongside, since positioning remains the
open problem.

**Result: NO GAIN.** Greedy **0.000** / stochastic **0.708** against the 0.844
best; stand-still 0.998, travel 0.28 tiles per episode. The entropy bonus did
not fix the degenerate greedy mode (0.000 with and without `--ent-coef 0.01`),
so the mechanic (cycle 17) was tried instead. Artifacts:
`evals/*cycle15-greedy.json`, `evals/*cycle15-stochastic.json`.

## Cycle 16 — 2026-09-11 — engagement preset: paying for passivity

**Lane:** config-derived behaviors.

Every recipe so far ends at `stand_still_rate` 1.000 with 0.0-0.02 tiles of
travel per episode, including the runs that score 0.84: the environment pays
nothing for closing distance or holding ground, and ducking blocks horizontal
fire, so trading from spawn is the cheapest policy that survives.

`RewardConfig` gains two shaping fields -- `far_distance` (free radius, default
0) and `far_penalty_per_tile` (default 0.0) -- applied symmetrically each tick
as `-far_penalty_per_tile * max(0, |dx| - far_distance)`. A new
`engagement` preset layers them on the anti-stall values (free radius 6 tiles,
0.005 per excess tile, so trading at 20 tiles costs ~0.07 per tick while a
closed position costs nothing). Unlike a stand-still penalty it cannot be gamed
by jittering in place: only actually closing the gap stops the bleed.

Tests cover the free radius, the scaling with distance, symmetry between
agents, and that presets without shaping are unaffected. 377 tests pass.

**Smoke test (120k steps, same league/ent-coef as the control)** — the
mechanism works but the behavioural effect is *not* demonstrated yet, and one
measurement came out opposite to the intent:

| checkpoint | stochastic win vs scripted | vs zoner | mean distance vs scripted | fraction > 6 tiles | stand-still |
|---|---|---|---|---|---|
| engagement 120k | 0.50 | 0.83 | **18.35** | **0.890** | 0.916 |
| anti-stall 100k (control) | 0.50 | 0.67 | 14.98 | 0.757 | 0.890 |

At 120k both policies are still almost stationary, so the distance is dominated
by what the opponent does rather than by shaping, and six-round win rates carry
no weight. The preset is a verified capability (tests: free radius, scaling,
symmetry, no effect when disabled) but its usefulness is unproven; a full 1M
comparison with the distance/far-fraction metrics is the test, and the 0.005
per-tile magnitude may simply be too small next to a ±12 win/lose swing.

**Verdict: GAIN** (capability + honest negative early signal; the real test is
deferred to cycle 17, not claimed here).

## Cycle 17 — 2026-09-11 — duck cooldown: capping how long a turtle can hide

**Lane:** config-derived behaviors (mechanics).

Four training runs in a row ended in the same place: hold position, duck the
incoming fire, fire back (`stand_still_rate` 0.998-1.000, travel 0.0-0.3 tiles
per episode). Ducking is what makes that work -- a duck covers two ticks and
blocks every `dy == 0` bullet, so alternate ducks give *continuous* cover while
standing still.

`AgentConfig.duck_cooldown` (default **0**, so existing behaviour and every
shipped checkpoint are untouched) adds recovery ticks after a duck during which
DUCK is refused. With `duck_duration=2` and `duck_cooldown=2`, a dedicated
ducker covers two of every four ticks. Measured functionally -- blocking a
bullet versus taking it -- a turtle blocks 12/12 incoming shots at cooldown 0
and **6/12** at cooldown 2. Exposed as `--duck-cooldown`, recorded in checkpoint
metadata; tests cover the halved cover, unchanged default, agent symmetry and
refused back-to-back ducks. 380 tests pass.

**Interaction found before trusting it:** the cooldown also disarms the
league's anti-turtle archetype. The rusher's duck-march assumes cover can be
renewed every other tick, and at `duck_cooldown=2` it goes from **4-0 wins to
0-4 losses** against a stationary gunner and stalls against
scripted/aggressive/zoner. A hop-based cooldown-aware rewrite produced 500-tick
stalemates instead and was reverted rather than shipped half-working; the
archetype documents that it assumes cooldown 0, and the gap is tracked above.

**Verdict: GAIN** (removes the attractor every previous recipe collapsed into,
with the trade-off measured rather than assumed).

## Cycle 18 — 2026-09-11 — duck-cooldown training run (NO GAIN)

**Lane:** trained policy gains.

1M steps with the best-known recipe plus the mechanic change: anti-stall
rewards, classic+flat, spawn jitter, mixed league (zoner/camper/evasive at
0.35), `--ent-coef 0.01`, `--duck-cooldown 2`
(`checkpoints/duckcd-mixed-1m`, seed 83). Success criteria, both modes
reported: stochastic mean win rate >= 0.844 (beat cycle 9) **or** a
stand-still rate clearly below 0.99 at comparable win rate. Because the rusher
is weak under the new mechanic it is deliberately not in this league; the
cooldown itself already halves a turtle's cover, so a duck-locking policy eats
half the incoming horizontal fire whether or not a specialist punisher is
present.

**Early read at the 100K milestone (10% of the run), stochastic sampling,
7 opponents x classic+flat:**

| run | mean win rate | stand-still | travel/episode |
|---|---|---|---|
| **cycle 18 (cooldown 2) @100K** | 0.829 | **0.699** | **6.41** |
| cycle 9 (no cooldown) @1M | 0.844 | 1.000 | 0.00 |
| cycle 15 (no cooldown, ent-coef) @1M | 0.708 | 0.998 | 0.28 |

The cooldown is the first change that moves the policy off its spawn: movement
goes up by 20-60x (6.41 tiles per episode, stand-still 0.699 versus 0.998-1.000)
while the win rate is already in cycle-9 territory at a tenth of the training
budget. That is the behaviour cycles 7-16 failed to produce, and it arrives with
the mechanic rather than with a reward tweak. The final verdict still rests on
the completed run; this is recorded as a partial result, not a claimed gain.

**Result: NO GAIN.** Completed 1M steps: greedy **0.750** / stochastic **0.781**
(8 opponents, classic+flat, 6 rounds) against cycle 9's 0.798 / 0.844;
stand-still 1.000, travel 0.01 tiles per episode. The 100K movement spike
(stand-still 0.699, 6.41 tiles) had faded by 1M. Artifacts:
`evals/*cycle18-greedy.json`, `evals/*cycle18-stochastic.json`,
`evals/*duckcd-100k-stoch.json`.

## Cycle 19 — 2026-09-11 — strong engagement shaping run (NO GAIN)

**Lane:** trained policy gains (config-derived shaping).

**Cycle 18 verdict first: NO GAIN.** The completed 1M run with a duck cooldown
scored greedy 0.750 / stochastic 0.781 with stand-still 1.000 and 0.01 tiles of
travel per episode -- both modes below cycle 9 (0.798 / 0.844), so the cooldown
cost a little top-end strength without producing lasting movement. The 100K
movement spike (stand-still 0.699, travel 6.41) turned out to be a transient.
No-gain count: **2 of 3** (cycles 15 and 18).

Cycle 19 is the pre-registered next attempt, testing the other half of the
problem: if standing still is cheap because opponents come to you, make passive
distance expensive. It repeats the cycle-9 recipe (anti-stall, classic+flat,
spawn jitter, mixed league zoner/camper/evasive at 0.35), keeps the duck
cooldown (it removed pure duck-lock even though it lost a little score), and
raises the passive-distance penalty fourfold with a tighter free radius:
`--far-distance 4 --far-penalty-per-tile 0.02` (holding at 20 tiles then costs
0.32 per tick against a 12-point win). `--far-distance` and
`--far-penalty-per-tile` are new CLI knobs so the magnitude is testable rather
than hard-coded.

**Pre-registered success criteria (any one is a gain):** stochastic mean win
rate >= 0.844 (matching the best), or greedy mean win rate >= 0.85, or a
stand-still rate <= 0.90 with a mean win rate >= 0.80. Anything else is the
third consecutive no-gain and the loop stops, per the objective's stop rule.

Status: launched; results appended below when complete.

**Result (cycle 19): NO GAIN.** None of the three pre-registered criteria
were met:

| criterion | required | measured |
|---|---|---|
| stochastic mean win rate | >= 0.844 | 0.771 |
| greedy mean win rate | >= 0.85 | 0.750 |
| stand-still rate (with mean >= 0.80) | <= 0.90 | 0.993 |

The fourfold passive-distance penalty (0.02 per excess tile, free radius 4) did
not change where the policy fights: travel rose only to 0.36 tiles per episode
against 0.01 for the same recipe without the penalty, and the score stayed
below cycle 9. Paying for passivity is not sufficient either.

# Loop status (corrected after audit, cycles 1-19)

**Stop rule NOT satisfied — the loop continues.** An earlier version of this
section claimed three consecutive no-gain cycles (15, 18, 19). That was wrong:
cycles **16 and 17 are gains** and sit between 15 and 18. The verdict sequence is

    1-5 gain, 6 no-gain, 7-10 gain, 11 no-gain, 12-14 gain,
    15 no-gain, 16-17 gain, 18 no-gain, 19 no-gain

so the longest run of consecutive no-gain cycles is **two (18, 19)**, one short
of the objective's stop condition. The loop therefore resumes with cycle 20
rather than stopping, and every further cycle must run the full promotion
evaluation (suite + rank + strategy_report) and have its result appended to its
own entry.

**What was verified:** `docs/verification-2026-09-11.md` records the
first-principles pass (its "Redo pass" section covers six defects, D1-D6, each
with a reproducing probe and a regression test), and `/tmp/reverify/*.py` holds
the independent probes: 30/30 environment/plumbing assertions, a mirror-contract
test that enumerates 53,428 bullet geometries, a physics probe, a greedy-vs-
stochastic probe, and an eight-archetype round robin. `uv run pytest -q` is at
382 passing; the smoke suite is 3/3.

**Lane coverage across the 19 cycles:**

* new scripted archetypes: zoner, camper, rusher (cycles 3, 10) plus the
  rusher's measured counter-play against the best trained policy (6-0)
* trained-policy gains: cycle 4 (0.000 -> 0.845 after the anti-stall preset
  escaped a 100% duck collapse), cycle 9 (re-graded in cycle 14: 0.844 with the
  evasive stalemate solved), cycle 5 (physics/mirror fixes that made training
  sound at all)
* config-derived behaviors: anti-stall preset, mixed league, spawn jitter,
  entropy coefficient, engagement preset, duck cooldown (cycles 7, 13, 16, 17)
* novel replay behaviors: positional replay analysis that exposed the
  trench-gunner gradient across four runs (cycle 12)
* measurement/tooling: action-entropy telemetry, stand-still diagnostics,
  dual-mode evaluation with `policy_sampling` provenance, movement/gate/report
  verification (cycles 8, 14)

**Best artifacts:** `checkpoints/antistall-1m/` (cycle 4) and
`checkpoints/mixed-league-1m/` (cycle 9, best measured policy: 0.844 stochastic
over eight opponents and two maps, 0.958 for its 500K snapshot over six
opponents); both are trust-manifested. `evals/` holds the suite, rank, gate,
strategy-report and league-health artifacts behind every number quoted here.

**The one unsolved problem**, stated plainly: every recipe ends with
`stand_still_rate` ~1.000. Six levers were tried against it -- mixed league
(7/9/11), spawn jitter (7), entropy bonus (15), passive-distance shaping (16,
19), capped duck cover (17/18) -- and the only real movement seen anywhere was a
transient at 100K steps of the duck-cooldown run (stand-still 0.699, 6.41 tiles
of travel per episode) that had faded by 1M. The lead worth following next is
that transient: it shows the policy *can* move under pressure, which points at
schedules (strong mechanic early, relaxed later) rather than static shaping, and
at a cooldown-aware melee threat to keep that pressure alive.

## Cycle 20 — 2026-09-11 — duck-cooldown schedule: pressure early, released later

**Lane:** config-derived behaviors (schedule).

Evidence from cycle 18: with a duck cooldown the policy genuinely repositioned
early (100K: stand-still 0.699, 6.41 tiles of travel per episode) and then
converged back to a stationary gunner by 1M (stand-still 1.000, travel 0.01).
The mechanic creates the behaviour while the policy is still forming; it does not
survive a long run. So the cooldown is applied as a *schedule*: keep it until a
configured step, then release it, letting the policy keep the habits it built
without paying the cooldown's cost for another 700K steps.

Implementation: `TrainingConfig.duck_cooldown_until`, the CLI
`--duck-cooldown-until`, `SelfPlayCallback._apply_duck_cooldown_schedule`
(idempotent, logs a `[Schedule]` line), plus `set_duck_cooldown` on both the env
and the training wrapper. Recorded in checkpoint metadata as
`duck_cooldown_until`. Tests: the schedule fires once at the threshold and not
before (`test_duck_cooldown_schedule_relaxes_at_the_configured_step`), and the
setter changes blocked-shot behaviour mid-episode, including the two ticks of
residual timer. 383 tests pass.

Run: 1M steps, `--duck-cooldown 3 --duck-cooldown-until 300000` on the cycle-9
recipe (anti-stall, classic+flat, spawn jitter, mixed league
zoner/camper/evasive at 0.35, `--ent-coef 0.01`), checkpoint dir
`checkpoints/cooldown-sched-1m`, seed 101. The stronger cooldown (3, not 2) is
used during the formative phase where cycle 18 showed movement.

**Pre-registered success criteria (any one is a gain):** stochastic mean win
rate >= 0.844 (matching the best), or a stand-still rate <= 0.90 with mean win
rate >= 0.80 (movement at the top of the range). Anything else is a no-gain and
makes it two consecutive (19, 20).

Status: launched; results appended below when complete.

## Unresolved / carried forward

- The trained policy has no positioning behaviour: stand_still_rate stays
  0.998-1.000 with 0.0-0.3 tiles of travel per episode in every mode. The mixed
  league (cycles 7/9/11), the entropy bonus (cycle 15) and the engagement preset
  (cycle 16) have not fixed it; cycle 17 attacks the mechanic that makes it
  viable (unlimited duck cover) and cycle 18 tests that.
- The rusher archetype assumes `duck_cooldown == 0`: under a cooldown its
  duck-march loses its approach timing and it goes from 4-0 wins to 0-4 losses
  against a stationary gunner. A cooldown-aware melee threat is needed if
  cooldown runs are to keep anti-turtle pressure in the league.
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

# Evaluation Experiment Playbook

Use this workflow for short, repeatable checks before and after changing rewards,
maps, policies, or training code. It avoids long training runs by default and
keeps comparable JSON artifacts in `evals/`.

## 1. Run A Baseline Eval

```bash
python scripts/train.py --mode eval \
  --opponent scripted \
  --rounds 20 \
  --seed 123 \
  --eval-output-dir evals \
  --eval-label baseline-scripted
```

For checkpoint evals, add `--checkpoint checkpoints/ppo_final` and pass a
trusted SHA-256 allowlist such as
`--trusted-checkpoint-manifest checkpoints/checkpoint-trust-manifest.json`.
If a companion checkpoint metadata file exists, eval output records it under
`eval_config.checkpoint_metadata`. New checkpoint sidecars also record file size
and SHA-256 digest so verifier runs can catch stale or replaced candidates.
Only load Stable-Baselines3 checkpoints from local or trusted sources. External
checkpoint `.zip` files should be treated as executable serialized artifacts and
verified by provenance and digest before use. Checkpoint-loading modes require
an explicit trust manifest before deserialization; sidecar project metadata is
integrity evidence, not a trust source. Use `--allow-unverified-checkpoints`
only for known-local legacy artifacts.

Saved eval, suite, rank, comparison, gate, rank-gate, promotion-audit,
audit-summary, artifact-index, strategy-report, long-run-check,
checkpoint-trust-manifest, and replay-analysis JSON include an `artifact` block
with `artifact_type` and `schema_version` so comparison and gate tools can
detect artifact shape changes.

## 2. Run A Changed Eval

Use the same seed, rounds, opponent, and map settings when comparing a code or
config change.

```bash
python scripts/train.py --mode eval \
  --opponent scripted \
  --rounds 20 \
  --seed 123 \
  --reward-preset anti_stall \
  --eval-output-dir evals \
  --eval-label anti-stall-scripted
```

## 3. Compare Saved Summaries

```bash
python scripts/train.py --mode compare \
  --before evals/baseline-scripted.json \
  --after evals/anti-stall-scripted.json \
  --eval-output-dir evals \
  --eval-label anti-stall-comparison
```

The comparison reports deltas for win/draw rates, episode length, average
rewards, action distributions, damage, behavior diagnostics, and per-map
metrics, including map-specific action-distribution, damage, and behavior
deltas. Evaluation per-map summaries include action counts, normalized action
distributions, damage totals, event totals, and behavior diagnostics so
map-specific failures can be isolated.
`--eval-output-dir` persists the comparison artifact for later review.

## 4. Gate Regressions

```bash
python scripts/train.py --mode gate \
  --before evals/baseline-scripted.json \
  --after evals/anti-stall-scripted.json \
  --eval-output-dir evals \
  --eval-label anti-stall-gate
```

Gate mode exits non-zero when default guardrails fail:

- agent 0 win-rate delta below `-0.05`
- draw-rate delta above `0.05`
- agent 0 idle-rate delta above `0.05`
- no-damage episodes increase
- low-engagement episodes increase

The same guardrails apply to per-map deltas when the comparison artifact
contains them, so a regression on one map cannot be hidden by aggregate
performance.

When `--eval-output-dir` is set, gate mode writes the gate artifact before
exiting, including on a failing gate.

## 5. Run A Baseline Suite

Use suite mode when checking broad behavior across opponent archetypes and maps.

```bash
python scripts/train.py --mode suite \
  --suite-opponents idle,scripted,aggressive,evasive \
  --suite-maps classic,flat,split,tower \
  --rounds 5 \
  --seed 123 \
  --eval-output-dir evals \
  --eval-label baseline-suite
```

Suite output contains one nested summary per map and opponent, plus a compact
overview with total matchups, total episodes, and mean agent 0 win rate. The
top-level suite config records active curriculum metadata and checkpoint
metadata when available. Use `--agent-policy` with no checkpoint to run a
built-in agent-under-test policy across a suite.

## 6. Rank Checkpoints

Use rank mode when comparing several saved checkpoints against the same
opponents and maps.

```bash
python scripts/train.py --mode rank \
  --checkpoint-dir checkpoints \
  --suite-opponents idle,scripted,aggressive,evasive \
  --suite-maps classic,flat,split,tower \
  --rounds 5 \
  --seed 123 \
  --rank-head-to-head \
  --eval-output-dir evals \
  --eval-label checkpoint-rank
```

Rank output sorts checkpoints by a baseline-suite score across the requested
opponents and maps. The default formula is
`mean(win_rate_agent_0 + 0.5 * draw_rate - 0.25 * no_damage_rate - 0.25 * low_engagement_rate)`.
Use `--rank-draw-weight`, `--rank-no-damage-penalty`, and
`--rank-low-engagement-penalty` to tune it. The output also includes checkpoint
metadata, per-matchup scores, per-map mean scores, worst-map fields, and nested
suite summaries. Use
`--rank-checkpoints path_a.zip,path_b.zip` to compare an explicit list. Add
`--rank-head-to-head` to include checkpoint-vs-checkpoint forward/reverse
matchups and standings in the same artifact. Head-to-head standings include
Elo-style ratings; tune the starting rating and update size with
`--rank-initial-elo` and `--rank-elo-k`.

## 7. Gate A Rank Artifact

Use rank gate before promoting a checkpoint selected by rank mode.

```bash
python scripts/train.py --mode rank_gate \
  --rank-summary evals/checkpoint-rank.json \
  --eval-output-dir evals \
  --eval-label checkpoint-rank-gate
```

The default gate checks the top-ranked checkpoint for:

- score at least `0.1`
- mean win rate at least `0.0`
- draw rate at most `0.9`
- no-damage rate at most `0.75`
- low-engagement rate at most `0.5`

Tune these with `--rank-min-score`, `--rank-min-win-rate`,
`--rank-max-draw-rate`, `--rank-max-no-damage-rate`, and
`--rank-max-low-engagement-rate`. Add `--rank-min-map-score` when a promotion
candidate must clear a score floor on every evaluated map, not just on aggregate
rank score.
For rank artifacts with a `head_to_head` section, optionally add
`--rank-min-head-to-head-elo` or `--rank-min-head-to-head-score`. When
`--eval-output-dir` is set, rank gate writes the gate artifact before exiting,
including on a failing promotion gate.

## 8. Run A Promotion Audit

Use promotion audit when selecting a checkpoint candidate. It runs rank and rank
gate in one command and writes the rank, gate, and audit bundle artifacts before
returning.

```bash
python scripts/train.py --mode promotion_audit \
  --checkpoint-dir checkpoints \
  --suite-opponents idle,scripted,aggressive,evasive \
  --suite-maps classic,flat,split,tower \
  --rounds 5 \
  --seed 123 \
  --rank-head-to-head \
  --eval-output-dir evals \
  --eval-label checkpoint-promotion
```

The saved promotion-audit artifact records whether the promotion passed, the
selected candidate, any rank-gate failures, rank labels, thresholds, and paths
to the saved rank and rank-gate artifacts. Add `--audit-include-nested` if a
single audit artifact should also embed the full rank and rank-gate JSON. A
failing audit exits non-zero after the artifacts are written.

## 9. Skim A Promotion Audit

Use audit summary when reviewing previous autonomous runs without opening a full
promotion-audit JSON file.

```bash
python scripts/train.py --mode audit_summary \
  --audit-summary evals/checkpoint-promotion.json \
  --eval-output-dir evals \
  --eval-label checkpoint-promotion-skim
```

The summary validates the input artifact type and prints candidate metrics,
pass/fail, failures, ranking labels, thresholds, and rank/gate artifact paths.

## 10. Index Saved Artifacts

Use artifact index after several autonomous loops to build a lightweight
manifest of an ignored eval directory.

```bash
python scripts/train.py --mode artifact_index \
  --artifact-dir evals \
  --recursive-artifacts \
  --eval-output-dir evals \
  --eval-label artifact-index
```

The index records artifact type counts, schema metadata, compact per-file
summaries, and links such as compare before/after paths, rank-gate rank
summaries, promotion-audit rank/gate paths, strategy-report issue counts, and
long-run-check failed required checks. Rank summaries also include the top
checkpoint's worst map score and invalid map-score count for quick triage. It
avoids duplicating the full nested JSON in rank and suite artifacts.

## 11. Detect Bad Strategies

Use strategy report after eval, suite, rank, or promotion-audit runs to identify
stalled or degenerate behavior automatically.

```bash
python scripts/train.py --mode strategy_report \
  --artifact-dir evals \
  --recursive-artifacts \
  --eval-output-dir evals \
  --eval-label bad-strategy-report
```

The report flags all-draw behavior, no-damage episodes or replay analyses, low
engagement, high agent 0 idle rate, high agent 0 dominant-action rate, saved
long-run-status artifacts whose checkpoint metadata still lacks required
historical-opponent samples, reward-shaping smoke regressions, and failed
smoke-suite, self-play-sampling-smoke, or long-run-artifact-smoke
health/strategy signals.
Tune thresholds with
`--strategy-max-draw-rate`, `--strategy-max-no-damage-rate`,
`--strategy-max-low-engagement-rate`, `--strategy-max-idle-rate`, and
`--strategy-max-dominant-action-rate`. It also reports the weakest suite/rank
map-opponent matchups by score, capped by `--strategy-max-weaknesses`.

## 12. Curriculum Planning

`map_progression` updates the training map pool and reward preset through the
self-play callback.
During training, reset info includes opponent-pool telemetry so smoke runs or
custom callbacks can confirm whether the league is sampling latest and
historical frozen snapshots. The training logger also records
`self_play/opponent_pool_size`, `self_play/latest_opponent_samples`,
`self_play/historical_opponent_samples`, `self_play/historical_sample_rate`,
`self_play/latest_opponent_snapshot_id`, `self_play/last_opponent_snapshot_id`,
and `self_play/last_sample_was_historical`. Reset info also includes active
snapshot ids and per-snapshot sample counts under `opponent_pool`, which helps
detect silent collapse to only the latest policy. Use `--opponent-pool-seed` to
make latest-vs-historical sampling reproducible for a training run. Checkpoint
metadata records the opponent-pool config plus latest opponent-pool stats, and
generated real-run manifests require historical opponent sampling evidence
before promotion. Long-run status also summarizes checkpoint opponent-pool
metadata and flags `checkpoint_historical_opponent_samples` when a run directory
has not produced the required historical sampling evidence yet.

Current stages:

- `flat_intro`: `flat` from step `0`
- `classic_duel`: `flat`, `classic` from step `250000`
- `mixed_routes`: `classic`, `split` with `anti_stall` from step `1000000`
- `full_map_pool`: `classic`, `flat`, `split`, `tower` with `anti_stall` from step `2500000`

Run a curriculum smoke train with a small timestep budget:

```bash
python scripts/train.py --mode train \
  --timesteps 128 \
  --curriculum map_progression \
  --checkpoint-dir /tmp/arena-curriculum-smoke
```

Use short smoke runs for wiring checks only. Real curriculum quality still needs
longer training and the eval suite/gate workflow above.
The anti-stall reward preset increases draw and idle penalties, adds an extra
penalty for no-damage timeout draws, and increases damage shaping. Use eval,
promotion-audit, and strategy-report artifacts to confirm that no-damage and
low-engagement rates move in the intended direction.
Average cumulative rewards are included in eval and per-map metrics, so reward-only
changes can be checked before behavior changes appear in win/draw rates.
Use `--agent-policy` for deterministic reward-shaping smokes when no checkpoint
is needed:

```bash
python scripts/train.py --mode eval \
  --agent-policy idle \
  --opponent idle \
  --rounds 1 \
  --reward-preset anti_stall \
  --eval-output-dir evals \
  --eval-label idle-anti-stall
```

For a repeatable bundle that also runs compare, suite, strategy report, and
artifact index, use:

```bash
python scripts/reward_shaping_smoke.py
python scripts/reward_shaping_smoke.py --summary-output /tmp/arena-reward-smoke-summary.json
```

By default the script writes artifacts and command logs to a timestamped system
temp directory and prints a compact JSON summary with reward, draw-rate,
idle-rate, dominant-action, no-damage, low-engagement, and damage-event deltas
plus strategy issue and indexed artifact counts. Use `--summary-output` when an
autonomous run should archive the reward-shaping smoke as an indexable
`reward_shaping_smoke` JSON artifact. Strategy reports scan saved
`reward_shaping_smoke` summaries and flag regressions when anti-stall idle
rewards do not decrease or draw rate increases. The nested strategy issue count
is retained as diagnostic context, but expected idle/no-training strategy issues
do not fail the smoke by themselves.
Use `--command-timeout-seconds` to bound each child command.

For a no-training self-play league sampling check, use:

```bash
python scripts/self_play_sampling_smoke.py
python scripts/self_play_sampling_smoke.py --summary-output /tmp/arena-self-play-sampling-summary.json
```

This seeds an opponent pool, loads frozen snapshots through the training
wrapper's reset path, and verifies that historical opponent snapshots are
sampled while map-pool resets cover multiple maps. Strategy reports scan saved
`self_play_sampling_smoke` artifacts and flag failed historical-sampling checks.

For a short training-to-evaluation wiring check, use:

```bash
python scripts/train_eval_smoke.py --opponent-pool-seed 123
```

This runs a tiny curriculum training job, forces sampled replay capture,
evaluates the final checkpoint with suite mode, runs replay analysis plus a
relaxed promotion audit, then writes strategy-report and artifact-index
artifacts. It also runs a diagnostic `long_run_check` and reports that
verifier's artifact, pass/fail result, exit code, and failed checks without
making expected weak-policy failures abort the smoke. Unexpected verifier
failures still fail the smoke. By default it exercises the same baseline
coverage set as the long-run manifest: `idle,scripted,aggressive,evasive` opponents across
`classic,flat,split,tower` maps. The relaxed promotion audit allows smoke-sized
draw, no-damage, and low-engagement outcomes so the command validates artifact
plumbing instead of policy quality. Treat it as a smoke check only; real policy
quality still needs longer training. Use `--opponent-pool-seed` to make the
smoke's self-play opponent sampling reproducible and record the checkpoint
opponent-pool config in the smoke summary.
Use `--command-timeout-seconds` to bound each child command.

For actual compute runs, use `docs/long_run_protocol.md`. To create the full
command bundle without executing expensive training, run:

```bash
python scripts/train.py --mode long_run_manifest \
  --run-id arena-manual-001 \
  --timesteps 5000000 \
  --eval-output-dir evals \
  --eval-label arena-manual-001-plan
```

The generated launcher embeds the no-training self-play sampling smoke and then
the tiny train/eval/verifier smoke before the expensive training command. The
standalone smoke above is still a useful quick check before generating or
inspecting a long-run launcher.
For a no-training check of only the manifest, status, league-health, and
artifact-index plumbing, run:

```bash
python scripts/long_run_artifact_smoke.py
python scripts/long_run_artifact_smoke.py --summary-output /tmp/arena-long-run-artifact-summary.json
```

Use `--summary-output` when an autonomous run should archive the no-training
manifest/status/health/index smoke as an indexable `long_run_artifact_smoke`
JSON artifact. The summary records explicit validation checks so expected
no-training long-run blockers do not become strategy blockers by themselves. It
also reports `self_play_sampling_preflight_state` and fails if indexed
status/health artifacts report a failed self-play preflight. Use
`--command-timeout-seconds` to bound each child command.

To run the cheap smoke bundle in compute-cost order, use:

```bash
python scripts/smoke_suite.py
python scripts/smoke_suite.py --summary-output /tmp/arena-smoke-summary.json
python scripts/smoke_suite.py --include-train-eval
```

The default smoke suite avoids training and includes reward shaping, self-play
sampling, and long-run artifact plumbing checks; `--include-train-eval` opts into
the tiny train/eval smoke. Use `--summary-output` when an autonomous run should
archive the combined smoke result as an indexable `smoke_suite` JSON artifact
that strategy reports can scan. The suite also tells child no-training smokes to
write their own indexable summary artifacts and records those paths in
`summary_paths`. Use `--command-timeout-seconds` to bound each child smoke
command in unattended runs.

Manifest generation also writes a `.preflight.sh` launcher so that exact smoke
can be run safely without entering the expensive training path.
The manifest records a compact git source snapshot so long-run artifacts can be
linked back to the code state that produced them.
The launcher records preflight and training exit-code sidecars and writes a
final artifact index before exiting if either early step fails, so failed
compute attempts still leave a small diagnostic bundle. Missing promotion,
strategy-report, or artifact-index files are converted into verifier input
failures rather than shell `ls` failures during artifact resolution.
Generated commands also write stdout/stderr to `.out` files in the run eval
directory, and artifact indexes summarize those command-log tails.
After the verifier runs, generated launchers save long-run status and league
health artifacts before the final artifact index, so each run ends with a compact
triage bundle even when promotion fails.
League health blocks on candidate, replay, historical-opponent, and smoke-scoped
strategy issues, so failed smoke summaries or smoke artifacts mixed into a run
directory cannot look healthy.
When a diagnostic manifest uses `--timesteps 10000` or less, the launcher pins
`--replay-save-interval 1` so replay analysis is covered even in tiny runs.
To see whether the latest generated plan has actually produced run evidence,
use:

```bash
python scripts/train.py --mode long_run_status --artifact-dir evals
python scripts/train.py --mode league_health --artifact-dir evals --eval-output-dir evals --eval-label league-health
```

The status output reports the latest manifest, whether the run directory exists,
passing long-run-check counts, and the next launcher command when the plan has
not been executed and the manifest source snapshot still matches the current
clean checkout. It also reports `missing_evidence` entries for absent exit-code
sidecars, usable checkpoint files, valid training replay files, or latest-run
verifier artifacts, and includes `source_safe_to_launch` plus
`source_stale_reasons` before a long run is started. The latest manifest status
also summarizes checkpoint opponent-pool metadata and flags missing
historical-opponent sample evidence.
League health mode rolls the latest strategy report, long-run status,
rank/head-to-head standings, promotion audit, and long-run check into one
promotion-health artifact with `health.blockers`, `health.warnings`, opponent-pool
readiness, self-play sampling smoke status, replay strategy blockers, long-run
status blockers, strategy-report weakest maps, direct rank-derived per-map score
signals, malformed candidate map-score checks, and head-to-head candidate Elo.
When the latest status includes an eval directory, those source artifacts are
scoped to that run so unrelated older artifacts do not make a blocked run look
healthy.

Single eval output records the current curriculum under
`eval_config.curriculum`; suite output records it under
`suite_config.curriculum`. This makes saved artifacts easier to compare after
curriculum changes.

## 13. Analyze A Replay

```bash
python scripts/train.py --mode analyze --episode replays/episode_100.json
python scripts/train.py --mode analyze \
  --episode replays/episode_100.json \
  --eval-output-dir evals \
  --eval-label replay-episode-100
python scripts/train.py --mode analyze \
  --replay-dir replays \
  --replay-samples-per-bucket 2 \
  --eval-output-dir evals \
  --eval-label replay-sample
```

Replay analysis reports winner, length, map, terminal HP, event totals, and
simple flags such as no damage, no projectile hits, no melee hits, no melee
attempts, no shots fired, and no attacks. With `--eval-output-dir`, it also
saves a `replay_analysis` artifact that can be scanned by `artifact_index` and
required by `long_run_check`.
Training writes sampled episode replay JSON files to `--replay-dir` every
`Config.training.replay_save_interval` episodes, which keeps long runs
inspectable without saving every rollout.
Replay frames include the actions that produced each post-step state, so replay
analysis can report action counts, action distributions, idle rate, and
dominant-action rate from real training episodes.
Directory analysis samples representative agent 0 wins, agent 1 wins, draws,
combat episodes, no-damage episodes, no-attack episodes, idle-heavy episodes,
and dominant-action episodes when those buckets are present, then writes a
`replay_analysis_batch` manifest alongside the selected analyses.

## Interpreting Behavior Diagnostics

- `avg_idle_rate`: high values indicate passive or stuck policies.
- `avg_dominant_action_rate`: high values indicate action spam or collapse.
- `no_damage_episodes`: episodes with no successful combat.
- `low_engagement_episodes`: no-damage episodes that reach max ticks.
- `damage_events`: count of successful damaging interactions.
- `event_totals`: explicit environment counters for shots fired, melee attempts,
  melee hits, projectile hits, damage dealt, and damage taken.

Winner metrics use terminal outcome: timeout truncations are draws, and
knockouts are decided by terminal HP. Shaped rewards can still differ inside a
draw, but they do not turn timeout episodes into wins.

For reward-shaping changes, prefer improvements that reduce idle rate,
no-damage episodes, and low-engagement episodes without lowering win rate across
the same maps and opponent suite.

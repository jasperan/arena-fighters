# Arena Fighters

2D platform fighter where two RL agents train against each other via self-play PPO. Built on PettingZoo and Stable-Baselines3.

Two agents spawn on a multi-platform arena, shoot projectiles, swing melee attacks, dodge, duck, and jump. One neural network plays both sides (shared-weight self-play with observation mirroring). An opponent pool keeps old snapshots around so the agent doesn't overfit to its own latest strategy.

## Features

- **9-action combat**: move, jump, duck, shoot (forward, diagonal up, diagonal down), melee, idle
- **Tile-based physics**: gravity, platform collisions, bullet trajectories
- **Self-play training**: single network plays both agents via observation mirroring
- **Named arena maps**: classic, flat, split, and tower layouts with reset-time randomization
- **Curriculum training**: named map-progression stages update the active training map pool and reward preset over time
- **Opponent pool**: samples frozen historical snapshots (80% latest, 20% random older) and logs latest-vs-historical plus per-snapshot telemetry to prevent silent strategy cycling
- **Evaluation harness**: checkpoint-vs-baseline matchups with win rate, draw rate, episode length, action counts, damage metrics, and behavior diagnostics
- **Checkpoint metadata**: saved checkpoints get companion `.meta.json` files with map, reward, curriculum state, opponent-pool stats, size, and SHA-256 digest
- **Combat event counters**: env infos expose shots fired, melee attempts/hits, projectile hits, and damage totals
- **Eval comparison**: compare saved evaluation JSON files to report metric deltas across experiments
- **Eval gate**: fail saved-eval comparisons when key metrics regress beyond default thresholds
- **Baseline suites**: evaluate one policy against multiple opponents and maps in one JSON artifact
- **Checkpoint ranking**: discover checkpoints and rank them by baseline-suite score
- **Head-to-head rank matchups**: optionally compare checkpoints against each other in forward/reverse pairings
- **Rank gate**: fail checkpoint rank artifacts that do not meet promotion thresholds
- **Promotion audit**: run rank plus rank gate in one command and save the rank, gate, and audit artifacts
- **Audit summary**: skim saved promotion-audit JSON into candidate, pass/fail, failures, and artifact paths
- **Artifact index**: scan ignored eval directories and create a lightweight manifest of JSON artifact types, summaries, and links
- **Strategy report**: scan saved artifacts for no-damage, low-engagement, all-draw, idle, or action-spam behavior
- **Long-run manifest**: emit a reproducible real-compute command bundle and launcher script without executing training
- **Long-run check**: validate promotion-audit, strategy-report, and artifact-index outputs against documented promotion criteria
- **Built-in baselines**: random, idle, scripted, aggressive, and evasive evaluation opponents
- **Reward presets**: default and anti-stall reward profiles for evaluation-driven training experiments
- **Custom CNN extractor**: processes a 6-channel 20x40 grid observation + 6-dim vector (HP, cooldowns, velocity, ducking)
- **16 modes**: headless training, live ASCII watch, frame-by-frame replay, replay analysis, JSON evaluation, eval comparison, eval gating, baseline suites, checkpoint ranking, rank gating, promotion audit, audit summary, artifact indexing, strategy reporting, long-run manifesting, long-run checking
- **Milestone checkpoints**: auto-saves at 100K, 500K, 1M, 5M, 10M, 50M, and 100M steps
- **Episode replay**: JSON-serialized frame logs with map and event metadata for post-hoc analysis

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
python scripts/train.py --mode train --randomize-maps --map-choices classic,flat,split,tower
python scripts/train.py --mode train --reward-preset anti_stall
python scripts/train.py --mode train --curriculum map_progression
```

Training logs go to `./tb_logs/` (viewable with `tensorboard --logdir tb_logs`).
Saved checkpoints also get companion `.meta.json` files recording map settings,
reward config, active curriculum stage, opponent-pool stats, file size, and
SHA-256 digest. Eval mode can find this metadata even when you pass the Stable
Baselines `.zip` checkpoint path.
Training also writes sampled episode replays to `--replay-dir` every
`Config.training.replay_save_interval` episodes so long-run artifact analysis can
inspect real training behavior without saving every rollout.
The `map_progression` curriculum updates the training map pool and reward preset
by timestep: `flat` first, then `flat/classic`, then `classic/split` with
`anti_stall`, then the full map pool with `anti_stall`.
The `anti_stall` reward preset increases draw and idle penalties, adds an extra no-damage timeout penalty, and increases damage shaping; use eval behavior diagnostics to confirm that idle rate, no-damage episodes, and low-engagement episodes move in the intended direction.

### Watch

```bash
# Random agents (no checkpoint)
python scripts/train.py --mode watch

# Trained agent
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_final

# Fixed number of rounds
python scripts/train.py --mode watch --checkpoint checkpoints/ppo_1M --rounds 10

# Specific map
python scripts/train.py --mode watch --map tower --rounds 3
```

Renders ASCII in the terminal. `Ctrl+C` to stop.

### Replay

```bash
python scripts/train.py --mode replay --episode replays/episode_100.json
python scripts/train.py --mode analyze --episode replays/episode_100.json
python scripts/train.py --mode analyze --replay-dir replays --replay-samples-per-bucket 2 --eval-output-dir evals --eval-label replay-sample
```

Training replays are sampled during `--mode train`; use `--replay-dir` to choose
where those episode JSON files are written. Replay analysis flags no-damage,
no-hit, no-shot, no-melee-attempt, and no-attack episodes for later strategy
reports.

### Evaluate

```bash
# Checkpoint against a scripted baseline
python scripts/train.py --mode eval --checkpoint checkpoints/ppo_final --opponent scripted --rounds 100 --seed 123

# Random policy smoke test against the scripted baseline
python scripts/train.py --mode eval --opponent scripted --rounds 20 --seed 123

# Deterministic built-in policy smoke test
python scripts/train.py --mode eval --agent-policy idle --opponent idle --rounds 1 --reward-preset anti_stall

# Try different built-in baselines
python scripts/train.py --mode eval --opponent evasive --rounds 20 --seed 123

# Report performance across randomized maps
python scripts/train.py --mode eval --checkpoint checkpoints/ppo_final --opponent scripted --rounds 100 --randomize-maps --map-choices classic,flat,split,tower

# Persist timestamped metrics for later comparison
python scripts/train.py --mode eval --opponent scripted --rounds 20 --eval-output-dir evals --eval-label random-vs-scripted
```

Evaluation prints JSON with win rates, draw rate, average episode length, average cumulative rewards, action counts, normalized action distributions, event totals, damage dealt, behavior diagnostics, and per-map metrics. Per-map summaries include the same action, damage, event, and behavior breakdowns so map-specific failures are easier to isolate. Timeout episodes are counted as draws even when shaped rewards differ; knockouts use terminal HP. The eval config records map settings, reward preset, active curriculum metadata, and checkpoint metadata when a companion `.meta.json` file exists. Built-in opponents are `random`, `idle`, `scripted`, `aggressive`, and `evasive`; use `--agent-policy` to choose a built-in agent 0 policy when no checkpoint is supplied. Behavior diagnostics include idle rate, dominant-action rate, no-damage episodes, low-engagement episodes, and damage-event counts. Use `--eval-output-dir` to save timestamped JSON summaries under an ignored artifact directory such as `evals/`.
Saved eval, suite, rank, comparison, gate, rank-gate, promotion-audit,
audit-summary, artifact-index, strategy-report, long-run-manifest,
long-run-check, and replay-analysis/replay-analysis-batch JSON include an
`artifact` block with `artifact_type` and `schema_version`.

Only load Stable-Baselines3 checkpoints that were produced locally or obtained
from trusted sources. Checkpoints are serialized model artifacts, not inert data;
verify external checkpoint provenance and digests before using `--checkpoint`,
`--checkpoint-dir`, or `--rank-checkpoints`.

### Compare

```bash
python scripts/train.py --mode compare --before evals/baseline.json --after evals/anti_stall.json --eval-output-dir evals --eval-label anti-stall-comparison
```

Comparison prints JSON deltas for win/draw rates, episode length, average rewards, action distributions, damage, behavior diagnostics, and per-map metrics, including map-specific action-distribution, damage, and behavior deltas. Add `--eval-output-dir` to persist the comparison artifact.

### Gate

```bash
python scripts/train.py --mode gate --before evals/baseline.json --after evals/anti_stall.json --eval-output-dir evals --eval-label anti-stall-gate
```

Gate mode exits non-zero when a comparison regresses default guardrail metrics: agent 0 win rate, draw rate, agent 0 idle rate, no-damage episodes, or low-engagement episodes. The same guardrails are applied to per-map deltas when present, so a single-map regression is not hidden by aggregate performance. When `--eval-output-dir` is set, the gate artifact is written before a failing gate exits.

### Suite

```bash
python scripts/train.py --mode suite --suite-opponents idle,scripted,aggressive,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label baseline-suite
python scripts/train.py --mode suite --agent-policy idle --suite-opponents idle --suite-maps flat --rounds 1 --reward-preset anti_stall --eval-output-dir evals --eval-label idle-suite
```

Suite mode evaluates one checkpoint or built-in agent policy against multiple built-in opponents across selected maps and prints one combined JSON summary. Use `--agent-policy` when no checkpoint is supplied. The suite config includes the same curriculum and checkpoint metadata fields used by single evals.

### Rank

```bash
python scripts/train.py --mode rank --checkpoint-dir checkpoints --suite-opponents idle,scripted,aggressive,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label checkpoint-rank
python scripts/train.py --mode rank --rank-checkpoints checkpoints/ppo_1M.zip,checkpoints/ppo_final.zip --suite-opponents scripted --suite-maps classic --rounds 20 --rank-head-to-head
```

Rank mode discovers checkpoints from `--checkpoint-dir` or uses
`--rank-checkpoints`, runs the baseline suite for each checkpoint, and sorts
them by a score that rewards wins, gives configurable partial credit for draws,
and penalizes no-damage or low-engagement episodes. The default formula is
`mean(win_rate_agent_0 + 0.5 * draw_rate - 0.25 * no_damage_rate - 0.25 * low_engagement_rate)`.
Tune the weights with `--rank-draw-weight`, `--rank-no-damage-penalty`, and
`--rank-low-engagement-penalty`. The output includes checkpoint metadata,
per-matchup scores, and full nested suite summaries. Add
`--rank-head-to-head` to include pairwise checkpoint-vs-checkpoint matchups with
both forward and reverse sides plus standings. Head-to-head standings include
Elo-style ratings; tune them with `--rank-initial-elo` and `--rank-elo-k`.

### Rank Gate

```bash
python scripts/train.py --mode rank_gate --rank-summary evals/checkpoint-rank.json --eval-output-dir evals --eval-label checkpoint-rank-gate
```

Rank gate checks the top-ranked checkpoint against promotion thresholds:
minimum score, minimum mean win rate, maximum draw rate, maximum no-damage rate,
and maximum low-engagement rate. Tune the defaults with `--rank-min-score`,
`--rank-min-win-rate`, `--rank-max-draw-rate`, `--rank-max-no-damage-rate`, and
`--rank-max-low-engagement-rate`. For artifacts created with
`--rank-head-to-head`, optionally require pairwise strength with
`--rank-min-head-to-head-elo` or `--rank-min-head-to-head-score`. When
`--eval-output-dir` is set, the rank-gate artifact is written before a failing
promotion gate exits.

### Promotion Audit

```bash
python scripts/train.py --mode promotion_audit --checkpoint-dir checkpoints --suite-opponents idle,scripted,aggressive,evasive --suite-maps classic,flat --rounds 5 --eval-output-dir evals --eval-label checkpoint-promotion
```

Promotion audit runs the same baseline ranking and rank-gate checks in one
command. With `--eval-output-dir`, it saves three artifacts: the rank summary,
the rank-gate decision, and a compact promotion-audit summary that records the
selected candidate, failures if any, rank labels, thresholds, and paths to the
saved rank/gate artifacts. Add `--audit-include-nested` when you want the audit
artifact to embed the full rank and rank-gate JSON. It exits non-zero when the
rank gate fails, after writing the audit files.

### Audit Summary

```bash
python scripts/train.py --mode audit_summary --audit-summary evals/checkpoint-promotion.json
python scripts/train.py --mode audit_summary --audit-summary evals/checkpoint-promotion.json --eval-output-dir evals --eval-label checkpoint-promotion-skim
```

Audit summary reads a saved promotion-audit JSON artifact and prints a compact
skim with pass/fail, candidate metrics, failed thresholds, ranking labels, and
paths to saved rank/gate artifacts. It validates that the input is a
promotion-audit artifact before summarizing.

### Artifact Index

```bash
python scripts/train.py --mode artifact_index --artifact-dir evals
python scripts/train.py --mode artifact_index --artifact-dir evals --recursive-artifacts --eval-output-dir evals --eval-label artifact-index
```

Artifact index scans saved JSON artifacts and prints a lightweight manifest with
artifact type counts, per-file schema metadata, compact summaries, and links
between artifacts such as compare before/after paths, rank-gate rank summaries,
promotion-audit rank/gate paths, strategy-report issue counts, and
long-run-check failed required checks. It also indexes `.exitcode`, `.sh`, and
`.out` sidecars from generated long-run launchers with compact summaries. It
does not copy nested rank or suite JSON into a new shape; it points at the source
artifacts.

### Strategy Report

```bash
python scripts/train.py --mode strategy_report --artifact-dir evals --recursive-artifacts --eval-output-dir evals --eval-label bad-strategy-report
```

Strategy report scans saved eval, suite, rank, rank-gate, promotion-audit,
audit-summary, and replay-analysis artifacts and flags likely bad strategies:
all-draw behavior, no-damage episodes/replays, low engagement, high agent 0
idle rate, or high dominant action rate. Tune thresholds with
`--strategy-max-draw-rate`,
`--strategy-max-no-damage-rate`, `--strategy-max-low-engagement-rate`,
`--strategy-max-idle-rate`, and `--strategy-max-dominant-action-rate`. The
report also ranks the weakest suite/rank map-opponent matchups by score so the
next curriculum, reward, or training pass can target the most fragile maps.
Limit that list with `--strategy-max-weaknesses`.

### Long-Run Check

```bash
python scripts/train.py --mode long_run_manifest \
  --run-id arena-manual-001 \
  --timesteps 5000000 \
  --eval-output-dir evals \
  --eval-label arena-manual-001-plan
```

Long-run manifest mode writes a `long_run_manifest` JSON artifact and executable
shell launchers without executing training: one full launcher containing the
train, promotion-audit, sampled-replay, strategy-report, artifact-index, and
long-run-check command sequence, plus a `.preflight.sh` launcher that runs only
the tiny train/eval/verifier smoke. The full launcher starts with that same
preflight before the expensive training step, so environment or verifier
failures surface before real compute is spent. Add
`--replay-save-interval N` when generating a diagnostic launcher that should
capture training replays more frequently than the default config. Rank-gate and
strategy-report thresholds passed to the manifest command are pinned into the
generated promotion-audit and strategy-report commands so the launcher remains
reproducible if defaults change later. The launcher copies itself into the run's
eval directory as `long-run-launcher.sh` before training starts, and the manifest
records a compact git source snapshot with commit, branch, dirty flag, and a
sample of `git status --short`. The launcher captures preflight, training,
promotion-audit, and `long_run_check` exit codes.
Each generated command writes stdout/stderr to a named `.out` file in the run
eval directory so the final artifact index keeps lightweight command-log tails.
If the preflight or training step fails, it writes a final artifact index before
exiting with that failing status. If promotion-audit or `long_run_check` fails
after artifacts were written, it still continues into the diagnostic verifier
and final artifact index before exiting with the verifier status. The captured
exit codes are persisted as `preflight.exitcode`, `train.exitcode`,
`promotion-audit.exitcode`, and `long-run-check.exitcode` inside the run's eval
directory for later inspection. If promotion-audit crashes before writing a
promotion artifact, the launcher passes a missing-artifact placeholder into
`long_run_check` so the final verifier records an `input_artifacts_loadable`
failure instead of stopping at shell artifact resolution.
Run IDs are restricted to letters, numbers, dots, underscores, and hyphens so
generated checkpoint, replay, and eval paths stay under their configured roots.
Diagnostic manifests at `10000` timesteps or below automatically pin
`--replay-save-interval 1` so replay analysis is exercised; larger runs use the
config default unless an interval is specified. Larger runs also require
head-to-head checkpoint standings in the generated `long_run_check`; tiny
diagnostic manifests leave that requirement off because they may only produce a
single final checkpoint. For real manifests, the generated verifier also
requires enough actual head-to-head episodes for one forward/reverse checkpoint
pair across every suite map, with per-map coverage enforced separately.

```bash
python scripts/train.py --mode long_run_check \
  --promotion-audit-summary evals/run/promotion.json \
  --strategy-report-summary evals/run/strategy-report.json \
  --artifact-index-summary evals/run/artifact-index.json \
  --long-run-required-maps classic,flat,split,tower \
  --long-run-min-eval-episodes 320 \
  --long-run-min-map-episodes 80 \
  --long-run-min-map-score 0.0 \
  --long-run-require-replay-analysis \
  --long-run-min-replay-combat-maps 4 \
  --long-run-min-opponent-historical-samples 1 \
  --long-run-require-candidate-checkpoint \
  --long-run-require-candidate-metadata \
  --long-run-require-candidate-integrity \
  --long-run-require-head-to-head \
  --long-run-min-head-to-head-episodes 160 \
  --long-run-min-head-to-head-map-episodes 40 \
  --long-run-required-curriculum-stage full_map_pool \
  --long-run-required-reward-preset anti_stall \
  --eval-output-dir evals/run \
  --eval-label long-run-check
```

Use long-run status to audit existing plans and run directories without
launching training:

```bash
python scripts/train.py --mode long_run_status --artifact-dir evals
python scripts/train.py --mode long_run_status --artifact-dir evals --eval-output-dir evals --eval-label long-run-status
```

Status mode recursively scans `--artifact-dir`, reports the latest
`long_run_manifest`, whether its full and preflight launchers and expected run
directories exist, how many `long_run_check` artifacts are under that run,
whether any passing check is present, and the next `bash ...` commands when the
latest preflight or full launcher has not been executed yet and the manifest
source snapshot still matches the current clean checkout. It also emits a
machine-readable `missing_evidence` list such as `train_exitcode`,
`real_training_replay_files`, or `latest_run_long_run_check`. The latest
manifest summary includes `source_safe_to_launch` and `source_stale_reasons` so
stale or dirty-source launchers can be regenerated before spending compute. It
is an audit aid only; a passing status still needs the underlying verifier
artifacts to be inspected before promoting a checkpoint.

Long-run check validates saved promotion-audit, strategy-report, and
artifact-index outputs against the documented promotion criteria. It exits
non-zero when required criteria fail, after writing its JSON result if
`--eval-output-dir` is set. Missing or malformed input artifacts are reported as
a failed `input_artifacts_loadable` check instead of a Python traceback. When
input paths are provided, it also verifies that the artifact index contains the
same promotion-audit and strategy-report files, which catches stale or
mismatched run directories. Use
`--long-run-required-maps` to require the full map set,
`--long-run-min-eval-episodes` to reject smoke-sized rank artifacts by counting
the promoted candidate's actual nested baseline-suite episode summaries, and
`--long-run-min-map-episodes` to require enough candidate evidence on each map.
Use `--long-run-min-map-score` to reject candidates that collapse on one map.
When replay analysis is required, `--long-run-min-replay-combat-maps` rejects
runs whose sampled replay combat evidence only appears on too few required maps.
Generated long-run manifests default this threshold to the full required-map
count, so the four-map protocol expects combat replay evidence on all four maps.
Real-run manifests also require candidate checkpoint metadata to show at least
one historical-opponent sample, which catches self-play runs that never exercised
the frozen opponent pool.
Use `--long-run-require-candidate-checkpoint` to reject stale promotion audits
whose selected checkpoint path is no longer present. Use
`--long-run-require-head-to-head` to reject bundles where rank mode skipped
checkpoint-vs-checkpoint standings, which prevents a single-checkpoint run from
passing as empirical historical-opponent evidence. Use
`--long-run-min-head-to-head-episodes` to require actual nested head-to-head
episode summaries instead of accepting config-only standings. Use
`--long-run-min-head-to-head-map-episodes` to make that historical comparison
cover each required map instead of concentrating all evidence on one map.
Use `--long-run-require-candidate-metadata` to require the companion checkpoint
metadata sidecar with map, reward, and curriculum context. When required maps
are configured, the metadata must include those maps. Use
`--long-run-require-candidate-integrity` to require that the sidecar's SHA-256
and size metadata still match the selected checkpoint file.
Use `--long-run-required-curriculum-stage` and
`--long-run-required-reward-preset` to ensure the candidate reached the expected
curriculum stage and reward preset.

Replay evidence can be saved into the same artifact directory:

```bash
python scripts/train.py --mode analyze \
  --episode replays/episode_0100.json \
  --eval-output-dir evals/run \
  --eval-label replay-episode-0100
```

Saved replay-analysis artifacts are indexed by `artifact_index` and can satisfy
`long_run_check --long-run-require-replay-analysis` when they show combat.
When `analyze` is run with `--replay-dir` instead of `--episode`, it samples
representative agent 0 wins, agent 1 wins, draws, combat episodes, per-map
combat episodes, no-damage episodes, and no-attack episodes, then writes individual
`replay_analysis` artifacts plus a `replay_analysis_batch` manifest.

### Reward Shaping Smoke

```bash
python scripts/reward_shaping_smoke.py
python scripts/reward_shaping_smoke.py --output-dir /tmp/arena-reward-smoke --rounds 1 --map flat
```

The smoke script runs deterministic idle-vs-idle default and anti-stall evals,
compares reward deltas, runs an anti-stall suite, runs a strategy report, and
builds an artifact index. It writes all generated JSON and command output to a
timestamped `/tmp` directory by default.

### Train/Eval Smoke

```bash
python scripts/train_eval_smoke.py
python scripts/train_eval_smoke.py --output-dir /tmp/arena-train-eval-smoke --timesteps 128 --rounds 1
```

The train/eval smoke runs a tiny curriculum training job, forces sampled replay
capture, evaluates the produced `ppo_final.zip` checkpoint with suite mode, runs
replay analysis plus a relaxed promotion audit, then writes a strategy report
and artifact index. It also runs a diagnostic `long_run_check` and records that
verifier's artifact and exit code without failing the smoke when expected weak
policy checks fail; unexpected verifier failures still fail the smoke. By
default it uses the long-run baseline coverage set: `idle,scripted,aggressive,evasive`
opponents across `classic,flat,split,tower` maps. The relaxed audit allows
smoke-sized draw, no-damage, and low-engagement outcomes so the command checks
artifact plumbing rather than policy quality. It is only a wiring check; it
does not prove learning quality.

## Architecture

```
src/arena_fighters/
├── config.py      # All constants: grid size, HP, damage, rewards, PPO hyperparams, map layouts
├── env.py         # PettingZoo ParallelEnv with tile physics, combat, observation building
├── evaluation.py  # Evaluation policies and matchup metrics
├── network.py     # Custom SB3 features extractor (CNN for 6x20x40 grid + MLP for 6-dim vector)
├── self_play.py   # Opponent pool manager, wraps env for SB3 single-agent training
└── replay.py      # Episode logger (JSON) and playback
scripts/
└── train.py       # CLI entrypoint with train/watch/replay/analyze/eval/compare/gate/suite/rank/rank_gate/promotion_audit/audit_summary/artifact_index/strategy_report/long_run_manifest/long_run_check/long_run_status modes
```

### Observation Space

Dict observation with two components:

| Key | Shape | Contents |
|-----|-------|----------|
| `grid` | `(6, 20, 40)` | Platforms, own position, opponent position, own bullets, opponent bullets, facing direction |
| `vector` | `(6,)` | Own HP, opponent HP, shoot cooldown, melee cooldown, vertical velocity, ducking state |

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

The test suite covers the environment, network, self-play wrapper, evaluation harness, renderer, replay system, and training metadata.

For repeatable evaluation workflows, see `docs/experiments.md`.
For real compute runs, see `docs/long_run_protocol.md`.

## License

MIT

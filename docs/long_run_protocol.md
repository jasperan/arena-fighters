# Long-Run Training Protocol

Use this when you are ready to spend real compute. Do not run these commands as
part of a normal smoke check.

## Preflight: Train/Eval Smoke

Before spending real compute, run the tiny end-to-end smoke once:

```bash
python scripts/train_eval_smoke.py \
  --output-dir /tmp/arena-train-eval-smoke \
  --timesteps 128 \
  --rounds 1 \
  --opponent-pool-seed 123
```

This trains a tiny curriculum policy, writes a final checkpoint and metadata,
captures sampled training replays, runs suite evaluation, runs replay analysis,
runs a relaxed promotion audit, writes a strategy report, and builds an artifact
index. It also runs a diagnostic `long_run_check` and reports that verifier's
artifact, pass/fail result, exit code, and failed checks without making expected
weak-policy failures abort the smoke. Unexpected verifier failures still fail
the smoke. Its default suite coverage matches the long-run baseline set:
`idle,scripted,aggressive,evasive` opponents across
`classic,flat,split,tower` maps. The relaxed audit allows smoke-sized
draw/no-damage/low-engagement outcomes so the preflight validates artifact
plumbing instead of policy quality. Treat any strategy issues and failing smoke
long-run checks as expected diagnostics; real policy quality still requires the
long-run flow below. Use `--opponent-pool-seed` when you need reproducible
latest-vs-historical self-play sampling in the smoke; the summary records the
resulting checkpoint opponent-pool config.

## 0. Generate A Run Manifest

```bash
python scripts/train.py --mode long_run_manifest \
  --run-id arena-manual-001 \
  --timesteps 5000000 \
  --eval-output-dir evals \
  --eval-label arena-manual-001-plan
```

This writes a `long_run_manifest` JSON artifact, an executable full shell
launcher, and a `.preflight.sh` launcher with only the tiny smoke command. It
does not run training; inspect the launchers before spending compute. The full
launcher starts with the same tiny train/eval/verifier smoke preflight in a
sibling directory next to `$EVAL_DIR` before the expensive training command, so
a stale environment or broken verifier path fails before real compute is spent
without contaminating the real run's recursive artifact scans. Add
`--replay-save-interval N` for
diagnostic runs that should capture training replays more frequently than the
default config. Add `--opponent-pool-seed N` when generating the manifest to
seed both the embedded smoke preflight and the real self-play training command.
Rank-gate thresholds passed while generating the manifest are pinned into the
generated promotion-audit command for reproducibility; strategy-report
thresholds are pinned the same way.
The generated launcher copies itself into `$EVAL_DIR/long-run-launcher.sh`
before training starts so the final run bundle keeps the exact script that
produced it. The manifest also records a compact git source snapshot with
commit, branch, dirty flag, and a sample of `git status --short`. It writes
`preflight.exitcode` and `train.exitcode`; if either step fails, the launcher
writes a final artifact index before exiting with the failing status. If
promotion-audit exits before writing a promotion JSON, the launcher uses a
missing-artifact placeholder so `long_run_check` can report a structured
`input_artifacts_loadable` failure and the final artifact index can still be
written. After `long_run_check`, the launcher writes `long_run_status` and
`league_health` artifacts before the final artifact index, so the run bundle
keeps a compact evidence and blocker summary. Each generated command writes
stdout/stderr to a named `.out` file in `$EVAL_DIR` so failed runs keep
command-log tails in the final artifact index.
League health treats smoke-scoped strategy issues as blockers, which prevents
preflight or reward-smoke artifacts that were accidentally mixed into a real run
directory from being ignored in the final triage summary.
Run IDs are restricted to letters, numbers, dots, underscores, and hyphens so
generated checkpoint, replay, and eval paths stay under their configured roots.
For diagnostic manifests with `--timesteps` at or below `10000`, the generated
launcher automatically pins `--replay-save-interval 1` so the replay-analysis
path is exercised; larger runs use the config default unless you pass
`--replay-save-interval` explicitly.

## 1. Create Run Directories

```bash
RUN_ID=arena-$(date -u +%Y%m%dT%H%M%SZ)
CHECKPOINT_DIR=checkpoints/$RUN_ID
EVAL_DIR=evals/$RUN_ID
REPLAY_DIR=replays/$RUN_ID
mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR" "$REPLAY_DIR"
```

`checkpoints/`, `evals/`, and `replays/` are ignored by git. Keep them unless
you explicitly decide to archive or delete them.

## 2. Run The Embedded Preflight

```bash
python scripts/train_eval_smoke.py \
  --output-dir "${EVAL_DIR}-preflight-smoke" \
  --timesteps 128 \
  --rounds 1 \
  --suite-opponents idle,scripted,aggressive,evasive \
  --suite-maps classic,flat,split,tower \
  --opponent-pool-seed 123
```

This is the same smoke described above, embedded in the generated launcher. It
should complete before the expensive training command starts. It may report a
diagnostic `long_run_check` quality failure for the tiny policy, but unexpected
verifier or artifact failures abort the launcher.

## 3. Train With Curriculum

```bash
python scripts/train.py --mode train \
  --timesteps 5000000 \
  --curriculum map_progression \
  --randomize-maps \
  --map-choices classic,flat,split,tower \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --replay-dir "$REPLAY_DIR"
```

This uses the map-progression curriculum and switches to anti-stall rewards in
later stages. Increase timesteps only after the eval artifacts show useful
behavior rather than low-engagement draw farming.
Training writes checkpoint metadata plus `checkpoint-trust-manifest.json`; the
launcher also writes a run-scoped trust manifest under `$EVAL_DIR` before any
mode deserializes checkpoints.

## 4. Run Promotion Audit

```bash
python scripts/train.py --mode promotion_audit \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --trusted-checkpoint-manifest "$EVAL_DIR/checkpoint-trust-manifest.json" \
  --suite-opponents idle,scripted,aggressive,evasive \
  --suite-maps classic,flat,split,tower \
  --rounds 20 \
  --rank-head-to-head \
  --eval-output-dir "$EVAL_DIR" \
  --eval-label promotion
```

The promotion audit saves rank, rank-gate, and compact promotion-audit artifacts.
Use stricter rank-gate thresholds only after the baseline suite is stable enough
that failures are informative.

## 5. Inspect Behavior

```bash
python scripts/train.py --mode analyze \
  --replay-dir "$REPLAY_DIR" \
  --replay-samples-per-bucket 2 \
  --eval-output-dir "$EVAL_DIR" \
  --eval-label replay-sample

python scripts/train.py --mode strategy_report \
  --artifact-dir "$EVAL_DIR" \
  --recursive-artifacts \
  --eval-output-dir "$EVAL_DIR" \
  --eval-label strategy-report

python scripts/train.py --mode artifact_index \
  --artifact-dir "$EVAL_DIR" \
  --recursive-artifacts \
  --eval-output-dir "$EVAL_DIR" \
  --eval-label artifact-index
```

The strategy report should be treated as a blocker when it flags no-damage,
low-engagement, all-draw, idle, dominant-action behavior, or missing
historical-opponent sampling evidence for the candidate.
The training command in the generated launcher writes sampled episode replays to
`$REPLAY_DIR` every `Config.training.replay_save_interval` episodes.
Sampled replay analysis selects representative agent 0 wins, agent 1 wins,
draws, combat episodes, per-map combat episodes, no-damage episodes, and
no-attack episodes when present. Those `replay_analysis` artifacts give
`long_run_check --long-run-require-replay-analysis` machine-readable combat
evidence instead of relying on manual replay inspection.
Artifact indexing scans saved JSON artifacts plus generated `.exitcode`, `.sh`,
and `.out` sidecars, so final indexes surface verifier summaries, saved command
exit codes, archived launcher scripts, and command-log tails.

## 6. Skim The Audit

```bash
python scripts/train.py --mode audit_summary \
  --audit-summary "$EVAL_DIR"/*_promotion.json \
  --eval-output-dir "$EVAL_DIR" \
  --eval-label promotion-summary
```

Use the summary artifact for quick run comparison, then inspect the full rank
artifact before selecting a checkpoint.

## 7. Validate Promotion Criteria

```bash
python scripts/train.py --mode long_run_check \
  --promotion-audit-summary "$EVAL_DIR"/*_promotion.json \
  --strategy-report-summary "$EVAL_DIR"/*_strategy-report.json \
  --artifact-index-summary "$EVAL_DIR"/*_artifact-index.json \
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
  --eval-output-dir "$EVAL_DIR" \
  --eval-label long-run-check
```

The check exits non-zero when required criteria fail, after saving a
`long_run_check` artifact. Missing or malformed input artifacts are reported as
a failed `input_artifacts_loadable` check in that artifact instead of a Python
traceback. The generated launcher captures the `long_run_check` exit code, writes
a final artifact index that includes the `long_run_check` artifact, then exits
with the original verifier status. It also captures promotion-audit failure so a
rank-gate rejection can still proceed to strategy reporting, long-run checking,
and final artifact indexing when promotion-audit artifacts were written.
Captured exit codes are persisted as `preflight.exitcode`, `train.exitcode`,
`promotion-audit.exitcode`, and `long-run-check.exitcode` in the run eval
directory. Missing promotion, strategy-report, or artifact-index inputs are
passed through as placeholder paths so the verifier records a loadability
failure instead of the shell stopping during `ls` resolution. Use
`--long-run-min-maps` to tighten or relax map coverage expectations. The check
also confirms that the artifact index
contains the exact promotion-audit and strategy-report files passed on the
command line, so rerun artifact indexing if you regenerate either input. Use
`--long-run-required-maps` to require a specific map set,
`--long-run-min-eval-episodes` to reject smoke-sized rank artifacts. This check
counts the promoted candidate's actual nested baseline-suite matchup summaries
from the rank artifact instead of trusting `rank_config` or other checkpoints'
episodes; the command above expects at least
`20 rounds * 4 maps * 4 opponents = 320` candidate baseline evaluation episodes
before considering any head-to-head evidence. Use
`--long-run-min-map-episodes` to require enough promoted-candidate baseline
episodes on every evaluated map; the command above expects
`20 rounds * 4 opponents = 80` episodes per map.
Use `--long-run-min-map-score` to reject candidates whose mean score falls below
the threshold on any evaluated map.
When replay analysis is required, use `--long-run-min-replay-combat-maps` to
reject runs whose sampled combat replay evidence covers too few distinct
required maps. Generated long-run manifests default this threshold to the full
required-map count, so the standard four-map run expects combat replay evidence
on all four maps. Replay-required checks also fail when the strategy report
contains replay-level no-damage, no-attack, idle-heavy, or dominant-action
collapse issues.
Generated real-run manifests also require at least one historical-opponent sample
recorded in the selected checkpoint metadata, so a run cannot pass promotion if
it never exercised frozen opponent snapshots.
Use `--long-run-require-candidate-checkpoint` to reject stale promotion audits
whose selected checkpoint path is missing.
Use `--long-run-require-candidate-metadata` to require the checkpoint metadata
sidecar that records map, reward, and curriculum context. When required maps
are configured, the metadata must include those maps.
Use `--long-run-require-candidate-integrity` to require the sidecar's SHA-256
and size metadata to match the selected checkpoint file, which catches stale
metadata or checkpoint replacement before promotion.
Use `--long-run-require-head-to-head` to require checkpoint-vs-checkpoint
standings from rank mode. Generated real-run launchers enable this automatically
so a bundle cannot pass without evidence against historical snapshots; tiny
diagnostic launchers leave it off because they may only create one checkpoint.
Use `--long-run-min-head-to-head-episodes` to require actual nested
head-to-head episode summaries. The command above expects one forward/reverse
checkpoint pair across all four maps:
`20 rounds * 4 maps * 2 sides = 160` head-to-head episodes.
Use `--long-run-min-head-to-head-map-episodes` to require that evidence on each
required map. The command above expects `20 rounds * 2 sides = 40` head-to-head
episodes per required map.
Use `--long-run-required-curriculum-stage` and
`--long-run-required-reward-preset` to require that the candidate reached the
expected curriculum stage and reward preset.

## 8. Inspect Long-Run Status

```bash
python scripts/train.py --mode long_run_status \
  --artifact-dir evals \
  --eval-output-dir evals \
  --eval-label long-run-status

python scripts/train.py --mode league_health \
  --artifact-dir evals \
  --eval-output-dir evals \
  --eval-label league-health
```

Status mode recursively scans generated manifests, launcher sidecars, and
`long_run_check` artifacts without starting training. It reports the latest
manifest, whether its full/preflight launchers and run directories exist, how
many verifier artifacts are under that run, whether any latest-run verifier
passed, and the next preflight/full launcher commands when the latest plan has
not been executed and the manifest source snapshot still matches the current
clean checkout. It also emits a `missing_evidence` list for automation and human
triage, including missing exit-code sidecars, usable checkpoint files, valid
training replay files, or latest-run verifier artifacts. The latest manifest
summary also scans checkpoint metadata sidecars for opponent-pool
historical-sample evidence, and `missing_evidence` includes
`checkpoint_historical_opponent_samples` when a real-run manifest requires that
evidence but no checkpoint metadata satisfies it.
Treat this as a quick triage artifact, not as promotion proof by itself.

League health mode then combines the latest strategy report, long-run status,
rank/head-to-head standings, promotion audit, and long-run check into a compact
triage artifact. When the latest status points at a run eval directory, league
health reads strategy, rank, promotion, and verifier artifacts from that run
instead of mixing in older runs. Its `health.blockers` list highlights candidate
strategy issues, blocked long-run status, failed promotion audits, missing
historical-opponent sampling, and failed long-run checks in one place, while
`health.warnings` records missing source artifacts.

## 9. Promotion Criteria

Promote a checkpoint only when:

- rank gate passes with the selected thresholds
- strategy report has no candidate-level draw-rate, no-damage, low-engagement, idle, or dominant-action issues
- per-map metrics are not concentrated on a single easy map
- head-to-head standing is not worse than the older checkpoints being replaced
- sampled replay combat covers enough distinct required maps
- replay analysis from sampled episodes shows actual combat interactions
- selected candidate checkpoint metadata shows historical opponent sampling
- selected candidate checkpoint still exists
- selected candidate checkpoint metadata sidecar still exists
- selected candidate checkpoint metadata digest still matches the checkpoint file
- selected candidate metadata includes the required map set
- selected candidate metadata reached the required curriculum stage and reward preset

If these fail, prefer a narrow reward/curriculum change and repeat the same eval
suite before changing policy architecture or PPO hyperparameters.

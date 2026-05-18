# Autonomous Improvement Backlog

This backlog is the starting queue for re-runnable `/goal` improvement passes.
Each run should re-rank these items from current repo evidence before editing.

## Highest Leverage Lanes

1. Harden artifact and replay ingestion.
   - Why: long-run tooling reads ignored JSON artifacts and replay files from
     local run directories.
   - First slice: reject oversized or malformed JSON at loader boundaries and
     avoid following symlinked artifacts during recursive scans.
   - Verification: focused replay/evaluation/training-callback tests, full
     pytest, compileall, and diff whitespace checks.

2. Continue extracting `scripts/train.py` into cohesive modules.
   - Why: the CLI owns checkpoint trust, replay analysis, artifact indexing,
     strategy reporting, long-run status, manifest generation, and dispatch in
     one large file.
   - Candidate modules: checkpoint trust, artifact index, strategy report,
     long-run check/status, and manifest generation.
   - Constraint: move behavior behind existing tests before adding new APIs.

3. Make dependency and CI reproducibility explicit.
   - Why: security auditing currently depends on the active environment rather
     than a checked-in lock or constraints artifact.
   - Candidate slices: add a constraints workflow, CI smoke workflow, and
     documented audit command that can run from a fresh checkout.

4. Improve experiment ergonomics without spending real training compute.
   - Why: this project has good artifact plumbing, but autonomous runs need
     cheap diagnostics that identify bad reward/map/policy changes quickly.
   - Candidate slices: faster deterministic eval smoke presets, clearer
     promotion-audit summaries, and stronger artifact schema regression tests.

5. Keep docs synchronized with CLI behavior.
   - Why: README and protocol docs are extensive; drift is likely as modes and
     flags evolve.
   - Candidate slices: add doc tests or generated CLI help snapshots for the
     most important long-run commands.

## Current First Slice

Selected: harden artifact and replay ingestion.

Reason: it addresses concrete low-severity security-review residuals with a
narrow patch and tests, without changing valid training/evaluation behavior.

Status: implemented in this pass.

Evidence added:
- `load_eval_summary` rejects oversized summaries and non-object JSON.
- `load_replay` rejects oversized replay files, non-object JSON, and malformed
  `frames` values.
- Recursive strategy reports use the same symlink/root guard as artifact
  indexing.
- Focused regression tests cover the loader and symlink boundaries.

Validation:
- Focused replay/evaluation/training-callback tests passed.
- Full `python -m pytest` passed.
- `python -m compileall -q scripts src tests` passed.
- `git diff --check` passed.
- `pip check` passed.
- `pip-audit` found no known vulnerable dependencies; the local editable
  package itself is skipped because it is not published on PyPI.
- Codex review initially found one replay loader edge case (`frames: null`);
  it was fixed, retested, and the final rerun reported no actionable findings.
- Secret-pattern scan found only the existing redaction code and test fixtures.

## Next Recommended Slice

Extract checkpoint trust helpers from `scripts/train.py` into a cohesive module
under `src/arena_fighters/`, keeping all existing trust-manifest tests green.
This is the next best maintainability slice because checkpoint trust is a
security-sensitive boundary currently embedded in the large CLI file.

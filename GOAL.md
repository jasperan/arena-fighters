<goal>
Run a bounded autonomous improvement pass for the Arena Fighters repo. Research
the current checkout, update the ranked improvement backlog, implement one to
three high-value changes, verify them, and leave the next recommended slice for
the following run.
</goal>

<context>
Read these first:
- `/home/ubuntu/AGENTS.md`
- `README.md`
- `docs/long_run_protocol.md`
- `docs/experiments.md`
- `docs/autonomous_improvement_backlog.md`
- `pyproject.toml`

Inspect these code surfaces before choosing a slice:
- `scripts/train.py`
- `src/arena_fighters/evaluation.py`
- `src/arena_fighters/replay.py`
- `src/arena_fighters/env.py`
- `src/arena_fighters/self_play.py`
- `tests/test_training_callback.py`
- `tests/test_evaluation.py`
- `tests/test_replay.py`

Useful discovery commands:
- `git status --short --branch`
- `rg --files src scripts tests docs`
- `rg -n "TODO|FIXME|temporary|workaround|fallback|except Exception|pass$" src scripts tests docs README.md`
- `wc -l scripts/train.py src/arena_fighters/*.py tests/*.py`
</context>

<constraints>
Keep each run finite and reviewable. Prefer one well-verified improvement over
many shallow edits.

Do not run expensive real training. Use smoke tests and deterministic unit
tests unless the launch prompt explicitly authorizes long compute.

Do not add dependencies unless the selected slice clearly requires one and the
reason is documented.

Preserve checkpoint trust behavior: Stable-Baselines3 checkpoint loading must
remain gated by explicit trusted manifests or the existing legacy override.

Prefer deletion, extraction of cohesive existing behavior, and boundary
hardening over speculative abstractions.

Do not commit or push unless the launch prompt explicitly includes commit/push
instructions.
</constraints>

<done_when>
- `docs/autonomous_improvement_backlog.md` is updated with current evidence,
  completed slice notes, and the next recommended slice.
- At least one bounded improvement from the backlog is implemented, or a
  blocker is documented with exact evidence and no risky partial changes are
  left behind.
- New or updated regression tests cover the selected behavior.
- Focused tests for the touched surface pass.
- `python -m pytest` passes.
- `python -m compileall scripts src tests` passes.
- `git diff --check` passes.
- If security-sensitive code changed, a security review pass is summarized with
  severity-ranked findings or a clean result.
- If a non-trivial code change was made, Codex/code review output is summarized
  with accepted/rejected findings or a clean result when the local helper is
  available.
</done_when>

<workflow>
1. Check `git status --short --branch` and preserve unrelated user changes.
2. Read the context files and run discovery commands. Re-rank
   `docs/autonomous_improvement_backlog.md` from current evidence.
3. Choose one to three bounded slices. Prefer the highest-value item that can
   be covered with focused tests and no expensive training.
4. Before editing, run or add the smallest regression tests that lock the
   selected behavior.
5. Implement the slice with small, reversible patches. Keep behavior-compatible
   refactors separate from semantic fixes.
6. Re-run focused tests after each meaningful pass.
7. Run full verification: `python -m pytest`, `python -m compileall scripts src
   tests`, and `git diff --check`.
8. Run review/security checks when applicable and fix accepted findings with
   focused tests.
9. Update `docs/autonomous_improvement_backlog.md` with completed work,
   residual risks, and the next recommended slice.
</workflow>

<verification_loop>
Use the active project venv when one exists; otherwise create or install the
normal dev environment from `pyproject.toml`.

Focused checks should run before broad checks. For this project, common focused
commands include:
- `python -m pytest tests/test_replay.py tests/test_evaluation.py`
- `python -m pytest tests/test_training_callback.py`
- `python -m pytest tests/test_self_play.py tests/test_config.py`

Broad checks:
- `python -m pytest`
- `python -m compileall scripts src tests`
- `git diff --check`

Security/dependency checks when relevant and available:
- `pip check`
- `pip-audit`
</verification_loop>

<execution_rules>
- Check git status before edits.
- Preserve unrelated user changes.
- Prefer `rg` over `grep` when available.
- Use the runtime's patch/edit tool for manual edits when available.
- Read context files before implementation.
- Batch independent file reads in parallel when the runtime supports it.
- Run focused tests before broad tests.
- Do not paper over failures.
- Do not widen scope.
- Keep the final answer concise.
</execution_rules>

<output_contract>
Final response must include:
- selected slice and why it ranked highest
- changed files
- verification commands and pass/fail results
- review/security findings accepted or rejected
- residual risks
- next recommended autonomous slice

If commit/push was explicitly requested in the launch prompt, use the Lore
Commit Protocol from `/home/ubuntu/AGENTS.md` and report the commit hash plus
push target.
</output_contract>

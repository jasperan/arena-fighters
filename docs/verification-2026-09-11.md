# Verification Report — 2026-09-11

Independent first-principles verification of the arena-fighters repo, performed
before any improvement work. Every claim below was checked by running the code,
not by reading tests. Commands are reproducible from the repo root.

## 1. Claim ledger

| # | Claim | Command | Result | Confidence |
|---|-------|---------|--------|------------|
| 1 | Test suite passes | `uv run pytest -q` | 290 passed pre-fix, 298 post-fix | high |
| 2 | Sources compile | `uv run python -m compileall -q scripts src tests` | pass | high |
| 3 | No whitespace damage | `git diff --check` | pass | high |
| 4 | Cheap smoke bundle passes | `uv run python scripts/smoke_suite.py` | exit 0; reward-shaping, self-play-sampling, long-run-artifact all `passed: true` | high |
| 5 | Training runs end-to-end on GPU | `uv run python scripts/train.py --mode train --timesteps 20000 --checkpoint-dir /tmp/arena-verify/checkpoints --replay-dir /tmp/arena-verify/replays --seed 7` | `ppo_final.zip` + `.meta.json` + trust manifest + `replays/episode_0100.json` + TB logs | high |
| 6 | Checkpoint sidecar digest is real | recompute `hashlib.sha256` of `ppo_final.zip` vs `.meta.json` | match (`af4027…1bff`) | high |
| 7 | Checkpoint trust gate blocks untrusted loads | 4-case probe (no manifest / valid manifest / tampered SHA / override) | refuse=exit 1, valid=load, tampered=SHA mismatch exit 1, override=load | high |
| 8 | Suite mode emits a valid artifact | `--mode suite … --eval-output-dir /tmp/arena-verify/evals` | `artifact_type: suite, schema_version: 1` | high |
| 9 | Observation space shape | direct `env.reset()` probe | grid `(6,20,40)`, vector `(6,)` | high |
| 10 | Agent-1 observation mirroring is exact | `mirror_obs` involution + channel/HP checks under random play | exact | high |
| 11 | Timeout is a draw, not a shaped win | 5-tick episode + `infer_winner` with unequal rewards | truncations only; winner `draw` | high |
| 12 | Opponent pool sampling is 80/20 | 6000 samples, 8 snapshots | historical rate 0.1973 | high |
| 13 | Pool telemetry + eviction | stats keys, 25 adds into max-size-20 pool | size 20, oldest id 5, latest 24 | high |
| 14 | Reward presets match docs | `reward_config_for_preset` | anti_stall exact; `anti-stall` alias works | high |
| 15 | All 4 maps reset cleanly | reset probe per map | 4/4 ok with `events` info | high |
| 16 | Audit static-analysis claims still hold | `uvx ruff@0.16.7 …` F401/F541/F811/F841, ERA/PGH/PLW/FURB | all checks passed | high |
| 17 | Audit "no dead code / no secrets" claims | `uvx vulture@2.14 --min-confidence 60`, secret grep | same known SB3-callback/test-helper hits; only redaction test fixtures | medium |

## 2. Defects found (all fixed in cycle 1, see `docs/improvement-log.md`)

### D1 (CRITICAL) — self-play never swapped opponents

`SelfPlayWrapper.__init__` used `self.opponent_pool = opponent_pool or OpponentPool()`.
Because `OpponentPool` defines `__len__`, an **empty** pool is falsy, so the
wrapper silently discarded the caller's pool and created a private one. The
training loop's callback added snapshots to the caller's pool; the wrapper
sampled from its private, forever-empty pool and early-returned. Result: for an
entire training run the opponent stayed the frozen initial policy — the
opponent pool, historical sampling, and league behavior were all inoperative.

Evidence (pre-fix, 6144-step real PPO learn + 3 manual resets): pool size 1,
`latest_samples = 0`, `historical_samples = 0`, `last_sample_kind = None`.
Post-fix: `latest_samples = 53` over the same run; `wrapper.opponent_pool is pool`
is true. Regression test: `tests/test_self_play.py::test_self_play_wrapper_keeps_empty_caller_pool`.

Why tests missed it: `tests/test_integration.py` and the wrapper tests pass
pre-populated pools, which are truthy, so the swapped object went unnoticed.

### D2 (HIGH) — double knockout paid agent_1 a win over agent_0

The termination loop processed only the first dead agent in iteration order
(`agent_0` first), so a simultaneous KO always gave `agent_0` the lose reward
and `agent_1` the win reward. Measured over 200 scripted-vs-scripted episodes
per map: 200/200 double KOs, mean reward `agent_0 = -9.5`, `agent_1 = +10.5`
(**+20.0 reward gap per episode**). This is contradictory signal for a
shared-weight policy that plays both sides through mirrored observations, and
it disagrees with `infer_winner`, which already scored double KOs as draws.

Post-fix: gap `+0.000` on all four maps; both agents receive the draw reward.
Regression tests: `test_double_knockout_is_symmetric_draw`,
`test_single_knockout_pays_win_and_lose_rewards`.

### D3 (HIGH) — `classic` map was not mirror-symmetric

The shared-weight design plays agent_1 through mirrored observations, which is
only sound if mirroring maps the arena onto itself. `flat`, `split`, and
`tower` are symmetric; `classic` (the default map, inherited from the original
scaffold layout) was not: low platform `(31,34)` instead of `(35,38)` and
off-center mid platform `(14,21)`.

Evidence: mirror-symmetric action streams from mirrored spawns stayed exact on
`flat`/`split`/`tower` but diverged on `classic` at tick 12 (agent_1 lost HP,
agent_0 did not) — a pure geometry artifact. Post-fix: no divergence on any
map. Regression tests: `test_all_map_layouts_are_mirror_symmetric`,
`test_mirrored_action_stream_stays_mirror_symmetric[map_name=…]`.

## 3. Triage notes (checked, not defects)

- Timeout/knockout rewards include the per-step idle penalty when an agent
  idles; apparent "off-by-0.001" differences are correct behavior.
- `info["events"]` is per-step and reset each tick; `info["episode_events"]` is
  cumulative. Both were present and consistent.
- `scripts/train_eval_smoke.py` and the long-run artifact smoke were exercised
  through `scripts/smoke_suite.py` (exit 0).

## 4. Residual risks

- No long-duration training run had ever been executed before this pass; the
  first 150k-step run was started during this verification (see cycle log).
  Long-run promotion artifacts remain unproven at scale.
- Audit LOW items still open: `assert`-based empty-pool guard
  (`self_play.py`), redaction coverage only for `command_log` artifacts,
  broad `except Exception` in artifact scanning, `lru_cache` on env methods,
  no CI/lint config, `scripts/train.py` monolith.
- No dependency vulnerability scan in this pass (previous audit's `pip-audit`
  result cannot be reproduced offline against the torch-cu124 index).
- `/tmp` evidence paths are ephemeral; every claim above is reproducible with
  the listed commands.

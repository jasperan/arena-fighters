import json

from scripts.train import build_long_run_manifest
from scripts.train_eval_smoke import (
    ALLOWED_SMOKE_LONG_RUN_FAILURES,
    DEFAULT_SMOKE_SUITE_MAPS,
    DEFAULT_SMOKE_SUITE_OPPONENTS,
    build_long_run_check_command,
    build_parser,
    build_promotion_audit_command,
    build_train_eval_summary,
    validate_smoke_long_run_check,
)


def test_default_smoke_suite_matches_long_run_coverage():
    args = build_parser().parse_args([])

    assert args.suite_opponents == DEFAULT_SMOKE_SUITE_OPPONENTS
    assert args.suite_maps == DEFAULT_SMOKE_SUITE_MAPS
    assert args.suite_opponents == "idle,scripted,evasive"
    assert args.suite_maps == "classic,flat,split,tower"


def test_default_smoke_suite_matches_long_run_manifest_defaults():
    manifest = build_long_run_manifest(run_id="smoke-default-contract")
    manifest_config = manifest["manifest_config"]

    assert manifest_config["suite_opponents"] == DEFAULT_SMOKE_SUITE_OPPONENTS
    assert manifest_config["suite_maps"] == DEFAULT_SMOKE_SUITE_MAPS


def test_promotion_audit_smoke_command_relaxes_expected_tiny_run_failures(tmp_path):
    command = build_promotion_audit_command(
        ["python", "scripts/train.py"],
        tmp_path / "checkpoints" / "ppo_final.zip",
        "idle",
        "flat",
        1,
        tmp_path / "evals",
    )

    assert "--rank-min-score" in command
    assert command[command.index("--rank-min-score") + 1] == "-1"
    assert "--rank-max-draw-rate" in command
    assert command[command.index("--rank-max-draw-rate") + 1] == "1"
    assert "--rank-max-no-damage-rate" in command
    assert command[command.index("--rank-max-no-damage-rate") + 1] == "1"
    assert "--rank-max-low-engagement-rate" in command
    assert command[command.index("--rank-max-low-engagement-rate") + 1] == "1"


def test_long_run_check_smoke_command_uses_suite_coverage_thresholds(tmp_path):
    command = build_long_run_check_command(
        ["python", "scripts/train.py"],
        tmp_path / "evals" / "promotion.json",
        tmp_path / "evals" / "strategy.json",
        tmp_path / "evals" / "index.json",
        "idle,scripted,evasive",
        "classic,flat,split,tower",
        2,
        tmp_path / "evals",
    )

    assert "--long-run-min-maps" in command
    assert command[command.index("--long-run-min-maps") + 1] == "4"
    assert "--long-run-required-maps" not in command
    assert "--long-run-min-eval-episodes" in command
    assert command[command.index("--long-run-min-eval-episodes") + 1] == "24"
    assert "--long-run-min-map-episodes" in command
    assert command[command.index("--long-run-min-map-episodes") + 1] == "6"
    assert "--long-run-require-candidate-checkpoint" in command
    assert "--long-run-require-candidate-metadata" in command


def test_validate_smoke_long_run_check_allows_expected_strategy_failure(tmp_path):
    path = tmp_path / "long-run-check.json"
    path.write_text(
        json.dumps(
            {
                "passed": False,
                "checks": [
                    {
                        "id": next(iter(ALLOWED_SMOKE_LONG_RUN_FAILURES)),
                        "required": True,
                        "passed": False,
                    }
                ],
            }
        )
        + "\n"
    )

    validate_smoke_long_run_check(path, 1)


def test_validate_smoke_long_run_check_rejects_unexpected_failure(tmp_path):
    path = tmp_path / "long-run-check.json"
    path.write_text(
        json.dumps(
            {
                "passed": False,
                "checks": [
                    {
                        "id": "artifact_index_has_required_artifacts",
                        "required": True,
                        "passed": False,
                    }
                ],
            }
        )
        + "\n"
    )

    try:
        validate_smoke_long_run_check(path, 1)
    except RuntimeError as exc:
        assert "artifact_index_has_required_artifacts" in str(exc)
    else:
        raise AssertionError("Expected unexpected long_run_check failure to raise")


def test_build_train_eval_summary_reads_expected_artifacts(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    eval_dir = tmp_path / "evals"
    replay_dir = tmp_path / "replays"
    checkpoint_dir.mkdir()
    eval_dir.mkdir()
    replay_dir.mkdir()
    (checkpoint_dir / "ppo_final.zip").touch()
    (checkpoint_dir / "ppo_final.meta.json").write_text("{}\n")
    (replay_dir / "episode_0001.json").write_text("{}\n")
    (eval_dir / "20260504T000000Z_train-smoke-suite.json").write_text(
        json.dumps(
            {
                "suite_config": {
                    "opponents": ["idle", "scripted"],
                    "maps": ["flat", "tower"],
                },
                "overview": {
                    "total_matchups": 4,
                    "total_episodes": 4,
                },
            }
        )
        + "\n"
    )
    (eval_dir / "20260504T000001Z_train-smoke-promotion.json").write_text("{}\n")
    (eval_dir / "20260504T000001Z_train-smoke-long-run-check.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": [
                    {"id": "promotion_audit_passed", "required": True, "passed": True},
                    {
                        "id": "no_candidate_bad_strategy_issues",
                        "required": True,
                        "passed": False,
                    },
                    {
                        "id": "head_to_head_candidate_not_worse",
                        "required": False,
                        "passed": False,
                    },
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "long-run-check.exitcode").write_text("1\n")
    (eval_dir / "20260504T000001Z_train-smoke-replay-batch.json").write_text(
        "{}\n"
    )
    (eval_dir / "20260504T000001Z_train-smoke-strategy.json").write_text(
        json.dumps({"issue_count": 3}) + "\n"
    )
    (eval_dir / "20260504T000002Z_train-smoke-index.json").write_text(
        json.dumps(
            {
                "artifact_counts": {"replay_analysis": 1},
                "index_config": {"artifact_count": 2},
            }
        )
        + "\n"
    )

    summary = build_train_eval_summary(tmp_path)

    assert summary == {
        "output_dir": str(tmp_path),
        "checkpoint": str(checkpoint_dir / "ppo_final.zip"),
        "checkpoint_exists": True,
        "metadata": str(checkpoint_dir / "ppo_final.meta.json"),
        "metadata_exists": True,
        "replay_count": 1,
        "replay_analysis_batch": str(
            eval_dir / "20260504T000001Z_train-smoke-replay-batch.json"
        ),
        "replay_analysis_count": 1,
        "suite": str(eval_dir / "20260504T000000Z_train-smoke-suite.json"),
        "suite_opponents": ["idle", "scripted"],
        "suite_maps": ["flat", "tower"],
        "suite_total_matchups": 4,
        "suite_total_episodes": 4,
        "promotion_audit": str(
            eval_dir / "20260504T000001Z_train-smoke-promotion.json"
        ),
        "long_run_check": str(
            eval_dir / "20260504T000001Z_train-smoke-long-run-check.json"
        ),
        "long_run_check_passed": False,
        "long_run_check_exit_code": 1,
        "long_run_check_failed_checks": ["no_candidate_bad_strategy_issues"],
        "strategy_report": str(
            eval_dir / "20260504T000001Z_train-smoke-strategy.json"
        ),
        "strategy_issue_count": 3,
        "artifact_index": str(eval_dir / "20260504T000002Z_train-smoke-index.json"),
        "indexed_artifact_count": 2,
    }

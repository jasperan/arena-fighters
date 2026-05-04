import json
from pathlib import Path
import sys

from arena_fighters.evaluation import artifact_metadata
from scripts.smoke_suite import (
    build_smoke_commands,
    build_smoke_suite_summary,
    run_command,
    write_smoke_suite_summary,
)


def test_build_smoke_commands_defaults_to_no_training_smokes(tmp_path):
    commands = build_smoke_commands(Path("/repo"), tmp_path)
    reward, long_run_artifact = commands

    assert [command["id"] for command in commands] == [
        "reward_shaping",
        "long_run_artifact",
    ]
    assert [command["compute_class"] for command in commands] == [
        "no_training_eval",
        "no_training_artifact",
    ]
    assert all(command["id"] != "train_eval" for command in commands)
    assert reward["summary_output"] == (
        tmp_path / "reward-shaping" / "reward-summary.json"
    )
    assert "--summary-output" in reward["cmd"]
    assert (
        reward["cmd"][reward["cmd"].index("--summary-output") + 1]
        == str(reward["summary_output"])
    )
    assert long_run_artifact["summary_output"] == (
        tmp_path / "long-run-artifact" / "artifact-smoke-summary.json"
    )
    assert "--summary-output" in long_run_artifact["cmd"]
    assert (
        long_run_artifact["cmd"][
            long_run_artifact["cmd"].index("--summary-output") + 1
        ]
        == str(long_run_artifact["summary_output"])
    )


def test_build_smoke_commands_can_include_tiny_training_smoke(tmp_path):
    commands = build_smoke_commands(
        Path("/repo"),
        tmp_path,
        include_train_eval=True,
        train_eval_timesteps=64,
        train_eval_rounds=2,
        train_eval_opponent_pool_seed=123,
    )
    train_eval = commands[-1]

    assert [command["id"] for command in commands] == [
        "reward_shaping",
        "long_run_artifact",
        "train_eval",
    ]
    assert train_eval["compute_class"] == "tiny_training"
    assert "--timesteps" in train_eval["cmd"]
    assert train_eval["cmd"][train_eval["cmd"].index("--timesteps") + 1] == "64"
    assert "--rounds" in train_eval["cmd"]
    assert train_eval["cmd"][train_eval["cmd"].index("--rounds") + 1] == "2"
    assert "--opponent-pool-seed" in train_eval["cmd"]
    assert (
        train_eval["cmd"][train_eval["cmd"].index("--opponent-pool-seed") + 1]
        == "123"
    )


def test_build_smoke_suite_summary_reads_no_training_smokes(tmp_path):
    reward_dir = tmp_path / "reward-shaping"
    artifact_output = tmp_path / "long-run-artifact"
    artifact_root = artifact_output / "evals"
    eval_dir = artifact_root / "artifact-smoke"
    reward_dir.mkdir()
    eval_dir.mkdir(parents=True)
    (reward_dir / "20260504T000000Z_idle-default.json").write_text("{}\n")
    (reward_dir / "20260504T000001Z_idle-anti-stall.json").write_text("{}\n")
    (reward_dir / "20260504T000002Z_idle-reward-compare.json").write_text(
        json.dumps({"deltas": {"draw_rate": 0.0}}) + "\n"
    )
    (reward_dir / "20260504T000003Z_strategy-report.json").write_text(
        json.dumps({"issue_count": 2}) + "\n"
    )
    (reward_dir / "20260504T000004Z_artifact-index.json").write_text(
        json.dumps({"index_config": {"artifact_count": 5}}) + "\n"
    )
    (artifact_root / "20260504T000000Z_artifact-smoke-plan.json").write_text(
        json.dumps({"artifact": artifact_metadata("long_run_manifest")}) + "\n"
    )
    (eval_dir / "20260504T000001Z_long-run-status.json").write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_status"),
                "blocked_reason": "latest_launcher_not_executed",
                "missing_evidence": ["train_exitcode"],
            }
        )
        + "\n"
    )
    (eval_dir / "20260504T000002Z_league-health.json").write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("league_health"),
                "health_config": {"artifact_scope_dir": str(eval_dir)},
                "health": {"ready": False, "blockers": [], "warnings": []},
            }
        )
        + "\n"
    )
    (eval_dir / "20260504T000003Z_artifact-smoke-index.json").write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("artifact_index"),
                "artifact_counts": {
                    "long_run_manifest": 1,
                    "long_run_status": 1,
                    "league_health": 1,
                },
                "index_config": {"artifact_count": 3},
            }
        )
        + "\n"
    )
    commands = build_smoke_commands(Path("/repo"), tmp_path)

    summary = build_smoke_suite_summary(tmp_path, commands)

    assert summary["artifact"] == {"artifact_type": "smoke_suite", "schema_version": 1}
    assert summary["smoke_count"] == 2
    assert summary["smoke_order"] == ["reward_shaping", "long_run_artifact"]
    assert summary["compute_classes"] == {
        "reward_shaping": "no_training_eval",
        "long_run_artifact": "no_training_artifact",
    }
    assert summary["summary_paths"] == {
        "reward_shaping": str(reward_dir / "reward-summary.json"),
        "long_run_artifact": str(artifact_output / "artifact-smoke-summary.json"),
    }
    assert summary["smokes"]["reward_shaping"]["strategy_issue_count"] == 2
    assert (
        summary["smokes"]["long_run_artifact"]["health_artifact_scope_dir"]
        == str(eval_dir)
    )


def test_write_smoke_suite_summary_creates_parent_dirs(tmp_path):
    summary = {
        "smoke_order": ["reward_shaping", "long_run_artifact"],
        "smoke_count": 2,
    }
    path = tmp_path / "nested" / "smoke-summary.json"

    written = write_smoke_suite_summary(summary, path)

    assert written == path
    assert json.loads(path.read_text()) == summary


def test_run_command_times_out_and_marks_stdout(tmp_path):
    stdout_path = tmp_path / "timeout.out"

    try:
        run_command(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            tmp_path,
            stdout_path,
            timeout_seconds=0.05,
        )
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected smoke command timeout")

    assert "Command timed out after 0.05 seconds" in stdout_path.read_text()

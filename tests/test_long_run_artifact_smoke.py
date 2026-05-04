import json
import sys

from arena_fighters.evaluation import artifact_metadata
from scripts.long_run_artifact_smoke import (
    build_artifact_smoke_summary,
    run_command,
    validate_artifact_smoke_summary,
    write_artifact_smoke_summary,
)


def test_build_artifact_smoke_summary_reads_expected_artifacts(tmp_path):
    artifact_root = tmp_path / "evals"
    run_id = "artifact-smoke"
    eval_dir = artifact_root / run_id
    eval_dir.mkdir(parents=True)
    manifest = {
        "artifact": artifact_metadata("long_run_manifest"),
        "manifest_config": {"run_id": run_id},
    }
    status = {
        "artifact": artifact_metadata("long_run_status"),
        "blocked_reason": "latest_launcher_not_executed",
        "missing_evidence": ["train_exitcode"],
        "latest_manifest": {
            "self_play_sampling_preflight": {
                "available": True,
                "passed": True,
                "historical_samples": 18,
                "historical_sample_rate": 0.28125,
                "latest_samples": 46,
                "unique_maps_seen": 4,
                "failed_checks": [],
            },
        },
    }
    preflight_signal = {
        "available": True,
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "failed_checks": [],
    }
    health = {
        "artifact": artifact_metadata("league_health"),
        "health_config": {"artifact_scope_dir": str(eval_dir)},
        "health": {
            "ready": False,
            "blockers": ["long_run_status_blocked"],
            "warnings": ["missing_rank"],
        },
        "signals": {
            "self_play_sampling_preflight": preflight_signal,
        },
    }
    index = {
        "artifact": artifact_metadata("artifact_index"),
        "artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
            "league_health": 1,
        },
        "index_config": {"artifact_count": 3},
    }
    (artifact_root / "20260504T000000Z_artifact-smoke-plan.json").write_text(
        json.dumps(manifest) + "\n"
    )
    (eval_dir / "20260504T000001Z_long-run-status.json").write_text(
        json.dumps(status) + "\n"
    )
    (eval_dir / "20260504T000002Z_league-health.json").write_text(
        json.dumps(health) + "\n"
    )
    (eval_dir / "20260504T000003Z_artifact-smoke-index.json").write_text(
        json.dumps(index) + "\n"
    )

    summary = build_artifact_smoke_summary(tmp_path, artifact_root, run_id)

    assert summary == {
        "artifact": {"artifact_type": "long_run_artifact_smoke", "schema_version": 1},
        "output_dir": str(tmp_path),
        "artifact_root": str(artifact_root),
        "run_id": run_id,
        "manifest": str(
            artifact_root / "20260504T000000Z_artifact-smoke-plan.json"
        ),
        "long_run_status": str(
            eval_dir / "20260504T000001Z_long-run-status.json"
        ),
        "league_health": str(eval_dir / "20260504T000002Z_league-health.json"),
        "artifact_index": str(
            eval_dir / "20260504T000003Z_artifact-smoke-index.json"
        ),
        "status_blocked_reason": "latest_launcher_not_executed",
        "status_missing_evidence": ["train_exitcode"],
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "health_artifact_scope_dir": str(eval_dir),
        "self_play_sampling_preflight_state": "passed",
        "status_self_play_sampling_preflight": preflight_signal,
        "health_self_play_sampling_preflight": preflight_signal,
        "indexed_artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
            "league_health": 1,
        },
        "indexed_artifact_count": 3,
        "checks": [
            {
                "id": "required_artifacts_indexed",
                "passed": True,
                "details": {
                    "missing_artifact_types": [],
                    "required_counts": {
                        "long_run_manifest": 1,
                        "long_run_status": 1,
                        "league_health": 1,
                    },
                },
            },
            {
                "id": "league_health_scoped_to_eval_dir",
                "passed": True,
                "details": {
                    "expected": str(eval_dir),
                    "actual": str(eval_dir),
                },
            },
            {
                "id": "status_blocked_reason_allowed",
                "passed": True,
                "details": {
                    "status_blocked_reason": "latest_launcher_not_executed",
                    "allowed": [
                        "latest_launcher_not_executed",
                        "latest_manifest_source_stale",
                    ],
                },
            },
            {
                "id": "self_play_preflight_not_failed",
                "passed": True,
                "details": {
                    "state": "passed",
                    "health_blockers": ["long_run_status_blocked"],
                },
            },
        ],
        "passed": True,
    }


def test_write_artifact_smoke_summary_creates_parent_dirs(tmp_path):
    summary = {
        "artifact": {"artifact_type": "long_run_artifact_smoke", "schema_version": 1},
        "run_id": "artifact-smoke",
        "health_ready": False,
    }
    path = tmp_path / "nested" / "artifact-smoke-summary.json"

    written = write_artifact_smoke_summary(summary, path)

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
        raise AssertionError("expected artifact smoke command timeout")

    assert "Command timed out after 0.05 seconds" in stdout_path.read_text()


def test_validate_artifact_smoke_summary_rejects_missing_artifact_counts(tmp_path):
    summary = {
        "indexed_artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
        },
        "health_artifact_scope_dir": str(tmp_path),
        "status_blocked_reason": "latest_launcher_not_executed",
    }

    try:
        validate_artifact_smoke_summary(summary, tmp_path)
    except RuntimeError as exc:
        assert "league_health" in str(exc)
    else:
        raise AssertionError("Expected missing league_health count to fail")


def test_validate_artifact_smoke_summary_rejects_wrong_health_scope(tmp_path):
    summary = {
        "indexed_artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
            "league_health": 1,
        },
        "health_artifact_scope_dir": str(tmp_path / "other"),
        "status_blocked_reason": "latest_launcher_not_executed",
    }

    try:
        validate_artifact_smoke_summary(summary, tmp_path)
    except RuntimeError as exc:
        assert "scope artifacts" in str(exc)
    else:
        raise AssertionError("Expected wrong league health scope to fail")


def test_validate_artifact_smoke_summary_rejects_failed_self_play_preflight(tmp_path):
    summary = {
        "indexed_artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
            "league_health": 1,
        },
        "health_artifact_scope_dir": str(tmp_path),
        "status_blocked_reason": "latest_launcher_not_executed",
        "self_play_sampling_preflight_state": "failed",
        "health_blockers": ["self_play_sampling_preflight_failed"],
    }

    try:
        validate_artifact_smoke_summary(summary, tmp_path)
    except RuntimeError as exc:
        assert "self_play_preflight_not_failed" in str(exc)
    else:
        raise AssertionError("Expected failed self-play preflight to fail")

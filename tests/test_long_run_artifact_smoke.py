import json

from arena_fighters.evaluation import artifact_metadata
from scripts.long_run_artifact_smoke import (
    build_artifact_smoke_summary,
    validate_artifact_smoke_summary,
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
    }
    health = {
        "artifact": artifact_metadata("league_health"),
        "health_config": {"artifact_scope_dir": str(eval_dir)},
        "health": {
            "ready": False,
            "blockers": ["long_run_status_blocked"],
            "warnings": ["missing_rank"],
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
        "indexed_artifact_counts": {
            "long_run_manifest": 1,
            "long_run_status": 1,
            "league_health": 1,
        },
        "indexed_artifact_count": 3,
    }


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

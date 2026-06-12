"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


def test_build_league_health_report_summarizes_latest_league_signals(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "candidate:candidate:checkpoint_opponent_pool",
                "metric": "checkpoint_historical_opponent_samples",
                "value": 0,
                "threshold": 1,
            }
        ],
        "weakness_count": 1,
        "weaknesses": [
            {
                "scope": "suite:flat/idle",
                "map_name": "flat",
                "opponent": "idle",
                "score": -0.25,
                "episodes": 20,
                "win_rate_agent_0": 0.0,
                "draw_rate": 0.5,
                "no_damage_rate": 0.25,
                "low_engagement_rate": 0.5,
                "avg_length": 50.0,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": False,
        "blocked_reason": "latest_long_run_check_not_passing",
        "missing_evidence": ["checkpoint_historical_opponent_samples"],
        "latest_manifest": {
            "run_id": "status-run",
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 0,
                "meets_min_opponent_historical_samples": False,
            },
            "self_play_sampling_preflight": {
                "available": True,
                "path": str(artifact_dir / "preflight-sampling.json"),
                "passed": True,
                "historical_samples": 18,
                "historical_sample_rate": 0.28125,
                "latest_samples": 46,
                "unique_maps_seen": 4,
                "failed_checks": [],
            },
        },
    }
    rank = _rank_summary(label="candidate", score=0.5)
    rank["rankings"][0]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": -0.25, "episodes": 20},
    ]
    rank["head_to_head"] = {
        "overview": {"total_episodes": 4},
        "standings": [
            {"label": "candidate", "elo": 1012.0, "score": 0.6},
            {"label": "older", "elo": 988.0, "score": 0.4},
        ],
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": False,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [
            {
                "id": "no_candidate_bad_strategy_issues",
                "required": True,
                "passed": False,
            }
        ],
    }
    promotion = _promotion_audit_summary()
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["artifact"] == {
        "artifact_type": "league_health",
        "schema_version": 1,
    }
    assert report["health"] == {
        "ready": False,
        "blockers": [
            "long_run_status_blocked",
            "candidate_strategy_issues",
            "historical_opponent_sampling",
            "long_run_check_failed",
        ],
        "warnings": [],
    }
    assert report["signals"]["candidate"]["label"] == "candidate"
    assert report["signals"]["opponent_pool"] == {
        "historical_sample_ready": False,
        "max_historical_samples": 0,
        "min_historical_samples": 1,
    }
    assert report["signals"]["strategy"]["candidate_issue_count"] == 1
    assert report["signals"]["strategy"]["historical_sampling_issue_count"] == 1
    assert report["signals"]["strategy"]["invalid_matchup_metric_count"] == 0
    assert (
        report["signals"]["strategy"]["candidate_invalid_matchup_metric_count"]
        == 0
    )
    assert report["signals"]["strategy"]["invalid_matchup_metrics"] == []
    assert report["signals"]["strategy"]["skipped_artifact_count"] == 0
    assert report["signals"]["strategy_trend"]["available"] is False
    assert report["signals"]["strategy_trend"]["previous_path"] is None
    assert report["source_artifacts"]["previous_strategy_report"] is None
    assert report["signals"]["map_weaknesses"]["maps"] == ["flat"]
    assert report["signals"]["map_weaknesses"]["worst"]["scope"] == "suite:flat/idle"
    assert report["signals"]["rank_map_scores"] == {
        "available": True,
        "candidate_label": "candidate",
        "map_count": 2,
        "per_map_scores": [
            {
                "map_name": "classic",
                "mean_score": 0.5,
                "matchup_count": 1,
                "episode_count": 20,
            },
            {
                "map_name": "flat",
                "mean_score": -0.25,
                "matchup_count": 1,
                "episode_count": 20,
            },
        ],
        "invalid_map_scores": [],
        "worst": {
            "map_name": "flat",
            "mean_score": -0.25,
            "matchup_count": 1,
            "episode_count": 20,
        },
    }
    assert report["signals"]["head_to_head"]["candidate_elo"] == 1012.0
    assert report["signals"]["head_to_head"]["standing_rank"] == 1
    assert report["signals"]["self_play_sampling_preflight"] == {
        "available": True,
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "failed_checks": [],
    }
    assert report["source_artifacts"]["self_play_sampling_preflight"] == str(
        artifact_dir / "preflight-sampling.json"
    )
    assert report["signals"]["long_run"]["failed_required_checks"] == [
        "no_candidate_bad_strategy_issues"
    ]


def test_build_league_health_surfaces_invalid_matchup_metrics(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 2,
        "issues": [
            {
                "scope": "candidate:candidate:rank_suite:flat/idle",
                "metric": "invalid_matchup_metric",
                "invalid_metric": "episodes",
                "value": "bad-count",
            },
            {
                "scope": "suite:classic/random",
                "metric": "invalid_matchup_metric",
                "invalid_metric": "win_rate_agent_0",
                "value": "bad-rate",
            },
        ],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "missing_evidence": [],
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert report["health"]["blockers"] == ["candidate_strategy_issues"]
    assert report["signals"]["strategy"]["invalid_matchup_metric_count"] == 2
    assert (
        report["signals"]["strategy"]["candidate_invalid_matchup_metric_count"]
        == 1
    )
    assert report["signals"]["strategy"]["invalid_matchup_metrics"] == [
        "episodes",
        "win_rate_agent_0",
    ]

    health_path = artifact_dir / "health.json"
    health_path.write_text(json.dumps(report) + "\n")
    index = build_artifact_index(artifact_dir)
    health_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "league_health"
    )
    assert health_entry["summary"]["strategy_invalid_matchup_metric_count"] == 2
    assert (
        health_entry["summary"]["candidate_strategy_invalid_matchup_metric_count"]
        == 1
    )
    assert health_entry["summary"]["strategy_invalid_matchup_metrics"] == [
        "episodes",
        "win_rate_agent_0",
    ]


def test_build_league_health_compares_previous_strategy_report(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    previous_strategy = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 2,
        "issues": [
            {"scope": "candidate:candidate", "metric": "draw_rate"},
            {"scope": "replay:episode_0001", "metric": "replay_no_damage"},
        ],
        "skipped_artifacts": [
            {
                "path": str(artifact_dir / "bad-suite.json"),
                "relative_path": "bad-suite.json",
                "artifact_type": "suite",
                "reason": "ValueError: bad suite metric",
            }
        ],
        "weakness_count": 2,
        "weaknesses": [
            {"scope": "suite:flat/idle", "map_name": "flat"},
            {"scope": "suite:classic/random", "map_name": "classic"},
        ],
    }
    current_strategy = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 3,
        "issues": [
            {
                "scope": "candidate:candidate:rank_suite:flat/idle",
                "metric": "invalid_matchup_metric",
                "invalid_metric": "episodes",
            },
            {
                "scope": "suite:classic/random",
                "metric": "invalid_matchup_metric",
                "invalid_metric": "win_rate_agent_0",
            },
            {"scope": "smoke:reward_shaping", "metric": "reward_smoke_failed"},
        ],
        "skipped_artifacts": [],
        "weakness_count": 1,
        "weaknesses": [
            {"scope": "suite:flat/idle", "map_name": "flat"},
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "missing_evidence": [],
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    previous_path = artifact_dir / "a-strategy.json"
    current_path = artifact_dir / "z-strategy.json"
    previous_path.write_text(json.dumps(previous_strategy) + "\n")
    current_path.write_text(json.dumps(current_strategy) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    trend = report["signals"]["strategy_trend"]
    assert trend["available"] is True
    assert trend["current_path"] == str(current_path)
    assert trend["previous_path"] == str(previous_path)
    assert report["source_artifacts"]["previous_strategy_report"] == str(
        previous_path
    )
    assert trend["current"]["issue_count"] == 3
    assert trend["previous"]["issue_count"] == 2
    assert trend["deltas"] == {
        "issue_count": 1,
        "candidate_issue_count": 0,
        "invalid_matchup_metric_count": 2,
        "candidate_invalid_matchup_metric_count": 1,
        "skipped_artifact_count": -1,
        "weakness_count": -1,
    }
    assert trend["regressions"] == [
        "candidate_invalid_matchup_metric_count",
        "invalid_matchup_metric_count",
        "issue_count",
    ]
    assert trend["improvements"] == [
        "skipped_artifact_count",
        "weakness_count",
    ]

    health_path = artifact_dir / "health.json"
    health_path.write_text(json.dumps(report) + "\n")
    index = build_artifact_index(artifact_dir)
    health_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "league_health"
    )
    assert health_entry["summary"]["strategy_trend_available"] is True
    assert health_entry["summary"]["strategy_issue_count_delta"] == 1
    assert (
        health_entry["summary"]["strategy_invalid_matchup_metric_count_delta"]
        == 2
    )
    assert health_entry["summary"]["strategy_skipped_artifact_count_delta"] == -1


def test_build_league_health_warns_on_strategy_report_skips(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "skipped_artifacts": [
            {
                "path": str(artifact_dir / "malformed-suite.json"),
                "relative_path": "malformed-suite.json",
                "artifact_type": "suite",
                "reason": "ValueError: bad suite metric",
            }
        ],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "missing_evidence": [],
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"] == {
        "ready": False,
        "blockers": [],
        "warnings": ["strategy_report_skipped_artifacts"],
    }
    assert report["signals"]["strategy"]["skipped_artifact_count"] == 1
    assert report["signals"]["strategy"]["skipped_artifacts"] == [
        {
            "path": str(artifact_dir / "malformed-suite.json"),
            "relative_path": "malformed-suite.json",
            "artifact_type": "suite",
            "reason": "ValueError: bad suite metric",
        }
    ]


def test_build_league_health_blocks_on_replay_strategy_issues(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "replay:episode_0001",
                "metric": "replay_dominant_action_rate_agent_0",
                "value": 1.0,
                "threshold": 0.95,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "replay_strategy_issues" in report["health"]["blockers"]
    assert report["signals"]["strategy"]["replay_issue_count"] == 1


def test_build_league_health_blocks_on_smoke_strategy_issues(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "smoke:reward_shaping",
                "metric": "reward_smoke_strategy_issue_count",
                "value": 3,
                "threshold": 0,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "smoke_strategy_issues" in report["health"]["blockers"]
    assert report["signals"]["strategy"]["smoke_issue_count"] == 1


def test_build_league_health_blocks_on_failed_self_play_sampling_smoke(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    sampling_smoke = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": False,
        "historical_samples": 0,
        "unique_maps_seen": 4,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": False},
        ],
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "sampling.json").write_text(json.dumps(sampling_smoke) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "self_play_sampling_smoke_failed" in report["health"]["blockers"]
    assert report["signals"]["self_play_sampling"] == {
        "available": True,
        "passed": False,
        "historical_samples": 0,
        "historical_sample_rate": None,
        "unique_maps_seen": 4,
        "failed_checks": ["historical_samples_meet_minimum"],
    }
    assert report["source_artifacts"]["self_play_sampling_smoke"] == str(
        artifact_dir / "sampling.json"
    )


def test_build_league_health_blocks_on_failed_self_play_preflight(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {
            "run_id": "status-run",
            "self_play_sampling_preflight": {
                "available": True,
                "path": str(artifact_dir / "preflight-sampling.json"),
                "passed": False,
                "historical_samples": 0,
                "historical_sample_rate": 0.0,
                "latest_samples": 64,
                "unique_maps_seen": 4,
                "failed_checks": ["historical_samples_meet_minimum"],
            },
        },
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(
        json.dumps({"artifact": artifact_metadata("strategy_report"), "issues": []})
        + "\n"
    )
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(
        json.dumps(_rank_summary(label="candidate", score=0.5)) + "\n"
    )
    (artifact_dir / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert "self_play_sampling_preflight_failed" in report["health"]["blockers"]
    assert report["signals"]["self_play_sampling_preflight"]["failed_checks"] == [
        "historical_samples_meet_minimum"
    ]


def test_build_league_health_blocks_on_latest_long_run_status(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": False,
        "blocked_reason": "latest_long_run_check_not_passing",
        "missing_evidence": ["passing_latest_long_run_check"],
        "latest_manifest": {
            "run_id": "latest-run",
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 2,
                "meets_min_opponent_historical_samples": True,
            },
        },
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "older", "score": 0.5},
        "checks": [
            {
                "id": "promotion_audit_passed",
                "required": True,
                "passed": True,
            }
        ],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(_rank_summary()) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (artifact_dir / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert report["health"]["blockers"] == ["long_run_status_blocked"]
    assert report["health"]["warnings"] == []
    assert report["signals"]["long_run"]["status_blocked_reason"] == (
        "latest_long_run_check_not_passing"
    )
    assert report["signals"]["long_run"]["latest_check_passed"] is True


def test_build_league_health_scopes_sources_to_latest_status_run(tmp_path):
    artifact_dir = tmp_path / "evals"
    older_run = artifact_dir / "older-run"
    latest_run = artifact_dir / "latest-run"
    older_run.mkdir(parents=True)
    latest_run.mkdir(parents=True)
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "blocked_reason": None,
        "missing_evidence": [],
        "latest_manifest": {
            "run_id": "latest-run",
            "eval_dir": str(latest_run),
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 2,
                "meets_min_opponent_historical_samples": True,
            },
        },
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [
            {
                "id": "promotion_audit_passed",
                "required": True,
                "passed": True,
            }
        ],
    }
    (latest_run / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (latest_run / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (latest_run / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (latest_run / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )
    (older_run / "newer-but-unrelated-rank.json").write_text(
        json.dumps(_rank_summary(label="unrelated", score=0.99)) + "\n"
    )

    report = build_league_health_report(artifact_dir)

    assert report["health_config"]["artifact_scope_dir"] == str(latest_run)
    assert report["source_artifacts"]["rank"] is None
    assert report["source_artifacts"]["strategy_report"] == str(
        latest_run / "strategy.json"
    )
    assert report["health"]["ready"] is False
    assert report["health"]["blockers"] == []
    assert report["health"]["warnings"] == ["missing_rank"]
    assert report["signals"]["candidate"]["rank_score"] is None
    assert report["signals"]["head_to_head"]["candidate_label"] is None


def test_run_league_health_can_save_indexable_artifact(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    output_dir = tmp_path / "health"

    run_league_health(
        str(artifact_dir),
        output_dir=str(output_dir),
        output_label="league-health",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_league-health.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)
    [health_entry] = index["artifacts"]
    assert "Saved league health report to" in stdout
    assert saved["health"]["ready"] is False
    assert set(saved["health"]["warnings"]) == {
        "missing_strategy_report",
        "missing_rank",
        "missing_promotion_audit",
        "missing_long_run_status",
        "missing_long_run_check",
    }
    assert index["artifact_counts"] == {"league_health": 1}
    assert health_entry["summary"]["ready"] is False
    assert "missing_rank" in health_entry["summary"]["warnings"]

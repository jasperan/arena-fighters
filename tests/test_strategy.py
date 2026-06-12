"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


def test_build_strategy_report_flags_stalled_eval_and_rank_artifacts(tmp_path):
    eval_summary = _eval_summary(
        "stalled",
        win_rate=0.0,
        draw_rate=1.0,
        idle_rate=0.9,
        no_damage_episodes=4,
        low_engagement_episodes=4,
    )
    eval_summary["behavior"]["avg_dominant_action_rate"]["agent_0"] = 0.99
    rank_summary = _rank_summary(
        label="stalled",
        score=0.0,
        no_damage_rate=1.0,
        low_engagement_rate=1.0,
    )
    rank_summary["rankings"][0]["mean_draw_rate"] = 1.0
    (tmp_path / "stalled-eval.json").write_text(json.dumps(eval_summary) + "\n")
    (tmp_path / "stalled-rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["artifact"] == {
        "artifact_type": "strategy_report",
        "schema_version": 1,
    }
    assert report["scanned_artifacts"] == 2
    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "draw_rate",
        "no_damage_rate",
        "low_engagement_rate",
        "idle_rate_agent_0",
        "dominant_action_rate_agent_0",
        "mean_draw_rate",
        "mean_no_damage_rate",
        "mean_low_engagement_rate",
    }.issubset(issue_metrics)


def test_build_strategy_report_scans_rank_embedded_suite_behavior(tmp_path):
    rank_summary = _rank_summary()
    rank_summary["suites"] = [
        {
            "label": "candidate",
            "suite": {
                "matchups": {
                    "classic": {
                        "idle": _eval_summary("candidate", idle_rate=0.9),
                    }
                }
            },
        }
    ]
    (tmp_path / "rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert any(
        issue["artifact_type"] == "rank"
        and issue["scope"] == "candidate:candidate:rank_suite:classic/idle"
        and issue["metric"] == "idle_rate_agent_0"
        for issue in report["issues"]
    )


def test_build_strategy_report_ranks_low_score_matchup_weaknesses(tmp_path):
    suite = {
        "artifact": artifact_metadata("suite"),
        "matchups": {
            "flat": {"idle": _eval_summary("weak", win_rate=0.0)},
            "tower": {"scripted": _eval_summary("strong", win_rate=1.0)},
        },
    }
    (tmp_path / "suite.json").write_text(json.dumps(suite) + "\n")

    report = build_strategy_report(tmp_path, max_weaknesses=1)

    assert report["issue_count"] == 0
    assert report["weakness_count"] == 2
    assert report["weaknesses"] == [
        {
            "path": str(tmp_path / "suite.json"),
            "relative_path": "suite.json",
            "artifact_type": "suite",
            "scope": "suite:flat/idle",
            "map_name": "flat",
            "opponent": "idle",
            "score": 0.0,
            "episodes": 4,
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "no_damage_rate": 0.0,
            "low_engagement_rate": 0.0,
            "avg_length": 10.0,
        }
    ]


def test_build_strategy_report_reports_malformed_matchups_without_skipping(tmp_path):
    healthy_eval = _eval_summary("healthy", win_rate=1.0)
    malformed_suite = {
        "artifact": artifact_metadata("suite"),
        "matchups": {
            "flat": {
                "idle": {
                    "episodes": "not-an-int",
                    "win_rate_agent_0": 0.0,
                    "draw_rate": 1.0,
                }
            }
        },
    }
    (tmp_path / "healthy-eval.json").write_text(json.dumps(healthy_eval) + "\n")
    (tmp_path / "malformed-suite.json").write_text(
        json.dumps(malformed_suite) + "\n"
    )

    report = build_strategy_report(tmp_path)

    invalid_metric_issues = [
        issue
        for issue in report["issues"]
        if issue["metric"] == "invalid_matchup_metric"
    ]
    assert report["scanned_artifacts"] == 2
    assert report["skipped_artifacts"] == []
    assert {
        "path": str(tmp_path / "malformed-suite.json"),
        "relative_path": "malformed-suite.json",
        "artifact_type": "suite",
        "scope": "suite:flat/idle",
        "metric": "invalid_matchup_metric",
        "invalid_metric": "episodes",
        "map_name": "flat",
        "opponent": "idle",
        "value": "not-an-int",
        "reason": "invalid_metric",
    } in invalid_metric_issues


def test_build_strategy_report_extracts_rank_matchup_weaknesses(tmp_path):
    rank = _rank_summary()
    rank["rankings"][0]["matchup_scores"] = [
        {
            "map_name": "classic",
            "opponent": "scripted",
            "score": -0.25,
            "episodes": 4,
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "no_damage_rate": 1.0,
            "low_engagement_rate": 0.0,
            "avg_length": 10.0,
        }
    ]
    (tmp_path / "rank.json").write_text(json.dumps(rank) + "\n")

    report = build_strategy_report(tmp_path)

    assert {
        "artifact_type": "rank",
        "scope": "rank:candidate:classic/scripted",
        "label": "candidate",
        "map_name": "classic",
        "opponent": "scripted",
        "score": -0.25,
    }.items() <= report["weaknesses"][0].items()


def test_build_strategy_report_flags_no_damage_replay_analysis(tmp_path):
    replay_summary = {
        "artifact": artifact_metadata("replay_analysis"),
        "episode_id": 7,
        "winner": "draw",
        "map_name": "flat",
        "flags": {"no_damage": True, "no_attacks": True},
        "totals": {"damage_dealt": 0},
    }
    (tmp_path / "no-damage-replay.json").write_text(
        json.dumps(replay_summary) + "\n"
    )

    report = build_strategy_report(tmp_path)

    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "replay_no_damage",
        "replay_low_engagement",
        "replay_no_attacks",
    }.issubset(issue_metrics)
    assert all(issue["artifact_type"] == "replay_analysis" for issue in report["issues"])


def test_build_strategy_report_flags_replay_action_collapse(tmp_path):
    replay_summary = {
        "artifact": artifact_metadata("replay_analysis"),
        "episode_id": 11,
        "winner": "draw",
        "map_name": "flat",
        "flags": {"no_damage": False, "no_attacks": False},
        "totals": {"damage_dealt": 10},
        "behavior": {
            "avg_idle_rate": {"agent_0": 1.0, "agent_1": 0.0},
            "avg_dominant_action_rate": {"agent_0": 1.0, "agent_1": 0.5},
        },
    }
    (tmp_path / "idle-replay.json").write_text(json.dumps(replay_summary) + "\n")

    report = build_strategy_report(tmp_path)

    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "replay_idle_rate_agent_0",
        "replay_dominant_action_rate_agent_0",
    }.issubset(issue_metrics)


def test_build_strategy_report_flags_candidate_draw_rate(tmp_path):
    promotion = _promotion_audit_summary()
    promotion["candidate"]["mean_draw_rate"] = 1.0
    (tmp_path / "promotion.json").write_text(json.dumps(promotion) + "\n")

    report = build_strategy_report(tmp_path)

    assert {
        "scope": "candidate:candidate",
        "metric": "mean_draw_rate",
        "value": 1.0,
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_long_run_status_missing_historical_samples(
    tmp_path,
):
    status = {
        "artifact": artifact_metadata("long_run_status"),
        "missing_evidence": ["checkpoint_historical_opponent_samples"],
        "latest_manifest": {
            "run_id": "status-run",
            "min_opponent_historical_samples": 1,
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 0,
                "meets_min_opponent_historical_samples": False,
            },
        },
    }
    (tmp_path / "status.json").write_text(json.dumps(status) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "long_run_status",
        "scope": "candidate:status-run:checkpoint_opponent_pool",
        "metric": "checkpoint_historical_opponent_samples",
        "value": 0,
        "threshold": 1,
        "reason": "checkpoint_historical_opponent_samples_below_min",
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_smoke_suite_failures(tmp_path):
    smoke_suite = {
        "artifact": artifact_metadata("smoke_suite"),
        "smokes": {
            "reward_shaping": {
                "strategy_issue_count": 15,
                "indexed_artifact_count": 11,
                "passed": False,
                "checks": [
                    {"id": "reward_delta_agent_0_negative", "passed": False},
                ],
            },
            "long_run_artifact": {
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "passed": False,
                "checks": [
                    {"id": "required_artifacts_indexed", "passed": False},
                ],
            },
            "self_play_sampling": {
                "passed": False,
                "checks": [
                    {"id": "historical_samples_meet_minimum", "passed": False},
                ],
            },
            "train_eval": {
                "long_run_check_passed": False,
                "long_run_check_failed_checks": [
                    "no_candidate_bad_strategy_issues",
                ],
                "strategy_issue_count": 2,
            },
        },
    }
    (tmp_path / "smoke-suite-summary.json").write_text(
        json.dumps(smoke_suite) + "\n"
    )

    report = build_strategy_report(tmp_path)

    issue_by_metric = {issue["metric"]: issue for issue in report["issues"]}
    assert report["issue_count"] == 5
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:reward_shaping",
        "metric": "smoke_reward_shaping_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_reward_shaping_checks_failed",
        "failed_checks": ["reward_delta_agent_0_negative"],
    }.items() <= issue_by_metric["smoke_reward_shaping_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:long_run_artifact",
        "metric": "smoke_long_run_artifact_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_long_run_artifact_checks_failed",
        "failed_checks": ["required_artifacts_indexed"],
        "blockers": ["long_run_status_blocked"],
        "warnings": ["missing_rank"],
    }.items() <= issue_by_metric["smoke_long_run_artifact_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:self_play_sampling",
        "metric": "smoke_self_play_sampling_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_self_play_sampling_checks_failed",
        "failed_checks": ["historical_samples_meet_minimum"],
    }.items() <= issue_by_metric["smoke_self_play_sampling_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:train_eval",
        "metric": "smoke_train_eval_strategy_issue_count",
        "value": 2,
        "threshold": 0,
        "reason": "smoke_train_eval_strategy_issues_present",
    }.items() <= issue_by_metric["smoke_train_eval_strategy_issue_count"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:train_eval",
        "metric": "smoke_train_eval_long_run_check_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_train_eval_long_run_check_failed",
        "failed_checks": ["no_candidate_bad_strategy_issues"],
    }.items() <= issue_by_metric[
        "smoke_train_eval_long_run_check_failed"
    ].items()


def test_build_strategy_report_ignores_healthy_smoke_suite(tmp_path):
    smoke_suite = {
        "artifact": artifact_metadata("smoke_suite"),
        "smokes": {
            "reward_shaping": {
                "strategy_issue_count": 15,
                "passed": True,
            },
            "long_run_artifact": {
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "passed": True,
            },
            "self_play_sampling": {
                "passed": True,
                "historical_samples": 12,
            },
            "train_eval": {
                "long_run_check_passed": True,
                "long_run_check_failed_checks": [],
                "strategy_issue_count": 0,
            },
        },
    }
    (tmp_path / "smoke-suite-summary.json").write_text(
        json.dumps(smoke_suite) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_flags_self_play_sampling_smoke_failures(tmp_path):
    sampling_smoke = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": False,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": False},
        ],
    }
    (tmp_path / "sampling-summary.json").write_text(
        json.dumps(sampling_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "self_play_sampling_smoke",
        "scope": "smoke:self_play_sampling",
        "metric": "self_play_sampling_smoke_failed",
        "value": 1,
        "threshold": 0,
        "reason": "self_play_sampling_smoke_checks_failed",
        "failed_checks": ["historical_samples_meet_minimum"],
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_reward_shaping_smoke_failures(tmp_path):
    reward_smoke = {
        "artifact": artifact_metadata("reward_shaping_smoke"),
        "reward_delta_agent_0": 0.0,
        "reward_delta_agent_1": 1.25,
        "draw_rate_delta": 0.5,
        "strategy_issue_count": 3,
        "passed": False,
        "checks": [
            {"id": "reward_delta_agent_0_negative", "passed": False},
            {"id": "reward_delta_agent_1_negative", "passed": False},
            {"id": "draw_rate_delta_not_positive", "passed": False},
        ],
    }
    (tmp_path / "reward-summary.json").write_text(json.dumps(reward_smoke) + "\n")

    report = build_strategy_report(tmp_path)

    issue_by_metric = {issue["metric"]: issue for issue in report["issues"]}
    assert report["issue_count"] == 4
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping",
        "metric": "reward_shaping_smoke_failed",
        "value": 3,
        "threshold": 0,
        "reason": "reward_shaping_smoke_checks_failed",
        "failed_checks": [
            "reward_delta_agent_0_negative",
            "reward_delta_agent_1_negative",
            "draw_rate_delta_not_positive",
        ],
    }.items() <= issue_by_metric["reward_shaping_smoke_failed"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping:agent_0",
        "metric": "reward_delta_agent_0",
        "value": 0.0,
        "threshold": 0.0,
        "reason": "anti_stall_idle_reward_not_reduced",
    }.items() <= issue_by_metric["reward_delta_agent_0"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping:agent_1",
        "metric": "reward_delta_agent_1",
        "value": 1.25,
        "threshold": 0.0,
        "reason": "anti_stall_idle_reward_not_reduced",
    }.items() <= issue_by_metric["reward_delta_agent_1"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping",
        "metric": "draw_rate_delta",
        "value": 0.5,
        "threshold": 0.0,
        "reason": "anti_stall_draw_rate_increased",
    }.items() <= issue_by_metric["draw_rate_delta"].items()


def test_build_strategy_report_ignores_healthy_reward_shaping_smoke(tmp_path):
    reward_smoke = {
        "artifact": artifact_metadata("reward_shaping_smoke"),
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "strategy_issue_count": 3,
        "passed": True,
    }
    (tmp_path / "reward-summary.json").write_text(json.dumps(reward_smoke) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_flags_long_run_artifact_smoke_failures(tmp_path):
    artifact_smoke = {
        "artifact": artifact_metadata("long_run_artifact_smoke"),
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "passed": False,
        "checks": [
            {"id": "required_artifacts_indexed", "passed": False},
        ],
    }
    (tmp_path / "artifact-smoke-summary.json").write_text(
        json.dumps(artifact_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "long_run_artifact_smoke",
        "scope": "smoke:long_run_artifact",
        "metric": "long_run_artifact_smoke_failed",
        "value": 1,
        "threshold": 0,
        "reason": "long_run_artifact_smoke_checks_failed",
        "failed_checks": ["required_artifacts_indexed"],
        "blockers": ["long_run_status_blocked"],
        "warnings": ["missing_rank"],
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_ignores_healthy_long_run_artifact_smoke(tmp_path):
    artifact_smoke = {
        "artifact": artifact_metadata("long_run_artifact_smoke"),
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "passed": True,
    }
    (tmp_path / "artifact-smoke-summary.json").write_text(
        json.dumps(artifact_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_allows_values_at_max_thresholds(tmp_path):
    eval_summary = _eval_summary(
        "threshold",
        win_rate=0.0,
        draw_rate=0.9,
        idle_rate=0.75,
        no_damage_episodes=3,
        low_engagement_episodes=2,
    )
    eval_summary["behavior"]["avg_dominant_action_rate"]["agent_0"] = 0.95
    rank_summary = _rank_summary(
        label="threshold",
        score=0.1,
        win_rate=0.0,
        no_damage_rate=0.75,
        low_engagement_rate=0.5,
    )
    rank_summary["rankings"][0]["mean_draw_rate"] = 0.9
    (tmp_path / "threshold-eval.json").write_text(json.dumps(eval_summary) + "\n")
    (tmp_path / "threshold-rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_run_strategy_report_can_save_report(tmp_path, capsys):
    eval_summary = _eval_summary(
        "stalled",
        win_rate=0.0,
        draw_rate=1.0,
        no_damage_episodes=4,
    )
    (tmp_path / "stalled-eval.json").write_text(json.dumps(eval_summary) + "\n")
    output_dir = tmp_path / "reports"

    run_strategy_report(
        str(tmp_path),
        recursive=False,
        max_draw_rate=0.9,
        max_no_damage_rate=0.75,
        max_low_engagement_rate=0.5,
        max_idle_rate=0.75,
        max_dominant_action_rate=0.95,
        output_dir=str(output_dir),
        output_label="bad-strategy",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_bad-strategy.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved strategy report to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "strategy_report",
        "schema_version": 1,
    }
    assert saved["issue_count"] >= 2

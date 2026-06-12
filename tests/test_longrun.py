"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


def test_build_long_run_check_passes_documented_promotion_criteria():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    assert result["artifact"] == {
        "artifact_type": "long_run_check",
        "schema_version": 1,
    }
    assert result["passed"] is True
    assert {check["id"] for check in result["checks"] if check["required"]} == {
        "promotion_audit_passed",
        "no_candidate_bad_strategy_issues",
        "strategy_report_analyzed_all_artifacts",
        "candidate_map_coverage",
        "artifact_index_has_required_artifacts",
        "replay_analysis_has_combat",
        "no_replay_bad_strategy_issues",
        "head_to_head_candidate_not_worse",
    }


def test_build_long_run_check_fails_bad_strategy_and_low_map_coverage():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "flat", "score": 0.0, "episodes": 20}
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(candidate_issue=True),
        _long_run_artifact_index(replay_analysis=False),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert {
        "no_candidate_bad_strategy_issues",
        "candidate_map_coverage",
        "replay_analysis_has_combat",
    }.issubset(failed)


def test_build_long_run_check_fails_replay_strategy_issues_when_required():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(replay_issue=True),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "no_replay_bad_strategy_issues"
    )
    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["issue_count"] == 1
    assert check["details"]["issues"][0]["metric"] == "replay_idle_rate_agent_0"


def test_build_long_run_check_fails_when_strategy_report_skipped_artifacts():
    strategy_report = _long_run_strategy_report()
    strategy_report["skipped_artifacts"] = [
        {
            "path": "evals/malformed-suite.json",
            "relative_path": "malformed-suite.json",
            "artifact_type": "suite",
            "reason": "ValueError: bad suite metric",
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "strategy_report_analyzed_all_artifacts"
    )
    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["skipped_artifact_count"] == 1
    assert check["details"]["skipped_artifacts"] == strategy_report[
        "skipped_artifacts"
    ]


def test_build_long_run_check_can_require_replay_combat_map_coverage():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(replay_maps=("classic",)),
        min_maps=2,
        min_replay_combat_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "replay_combat_map_coverage"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["combat_maps"] == ["classic"]
    assert check["details"]["min_replay_combat_maps"] == 2


def test_build_long_run_check_only_counts_required_replay_combat_maps():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(replay_maps=("debug_a", "debug_b")),
        min_maps=2,
        required_maps=("classic", "flat"),
        min_replay_combat_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "replay_combat_map_coverage"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["eligible_combat_maps"] == []
    assert check["details"]["ignored_combat_maps"] == ["debug_a", "debug_b"]


def test_build_long_run_check_treats_candidate_draw_rate_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate",
            "metric": "mean_draw_rate",
            "value": 1.0,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_ignores_other_candidate_strategy_issues():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:older",
            "metric": "mean_no_damage_rate",
            "value": 1.0,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "no_candidate_bad_strategy_issues"
    )
    assert result["passed"] is True
    assert check["details"]["issue_count"] == 0


def test_build_long_run_check_treats_candidate_idle_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate:rank_suite:classic/idle",
            "metric": "idle_rate_agent_0",
            "value": 0.9,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_treats_candidate_invalid_matchups_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate:rank_suite:flat/idle",
            "metric": "invalid_matchup_metric",
            "invalid_metric": "episodes",
            "value": "bad-count",
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_treats_candidate_historical_status_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate:checkpoint_opponent_pool",
            "metric": "checkpoint_historical_opponent_samples",
            "value": 0,
            "threshold": 1,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_can_require_head_to_head_standings():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["head_to_head"] = {
        "skipped": "requires_at_least_two_checkpoints",
        "checkpoint_count": 1,
    }

    optional = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )
    required = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
    )

    optional_check = next(
        check
        for check in optional["checks"]
        if check["id"] == "head_to_head_candidate_not_worse"
    )
    required_check = next(
        check
        for check in required["checks"]
        if check["id"] == "head_to_head_candidate_not_worse"
    )

    assert optional["passed"] is True
    assert optional_check["required"] is False
    assert optional_check["passed"] is False
    assert required["passed"] is False
    assert required_check["required"] is True
    assert required_check["passed"] is False
    assert required_check["details"]["reason"] == "requires_at_least_two_checkpoints"


def test_run_long_run_check_can_save_result(tmp_path, capsys):
    promotion_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    index_path = tmp_path / "index.json"
    promotion_path.write_text(json.dumps(_long_run_promotion_audit()) + "\n")
    strategy_path.write_text(json.dumps(_long_run_strategy_report()) + "\n")
    index = _long_run_artifact_index()
    index["artifacts"].extend(
        [
            {
                "artifact_type": "promotion_audit",
                "path": str(promotion_path),
                "relative_path": promotion_path.name,
                "summary": {},
            },
            {
                "artifact_type": "strategy_report",
                "path": str(strategy_path),
                "relative_path": strategy_path.name,
                "summary": {},
            },
        ]
    )
    index_path.write_text(json.dumps(index) + "\n")
    output_dir = tmp_path / "outputs"

    run_long_run_check(
        str(promotion_path),
        str(strategy_path),
        str(index_path),
        min_maps=2,
        required_maps=("classic", "flat"),
        min_eval_episodes=1,
        min_map_score=0.0,
        require_replay_analysis=True,
        output_dir=str(output_dir),
        output_label="long-run-check",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_long-run-check.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved long-run check to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "long_run_check",
        "schema_version": 1,
    }
    assert saved["passed"] is True
    assert saved["inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
        "artifact_index": str(index_path),
    }
    assert any(
        check["id"] == "artifact_index_contains_input_artifacts"
        and check["passed"]
        for check in saved["checks"]
    )


def test_run_long_run_check_saves_missing_input_failure(tmp_path, capsys):
    output_dir = tmp_path / "outputs"
    promotion_path = tmp_path / "missing-promotion.json"
    strategy_path = tmp_path / "missing-strategy.json"
    index_path = tmp_path / "missing-index.json"

    try:
        run_long_run_check(
            str(promotion_path),
            str(strategy_path),
            str(index_path),
            min_maps=2,
            required_maps=("classic", "flat"),
            min_eval_episodes=20,
            min_map_score=0.0,
            require_replay_analysis=True,
            output_dir=str(output_dir),
            output_label="missing-long-run-check",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected missing long-run inputs to exit non-zero")

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_missing-long-run-check.json")
    saved = json.loads(saved_path.read_text())
    assert "Traceback" not in stdout
    assert "Saved long-run check to" in stdout
    assert saved["passed"] is False
    assert saved["check_config"]["required_maps"] == ["classic", "flat"]
    assert saved["inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
        "artifact_index": str(index_path),
    }
    [check] = saved["checks"]
    assert check["id"] == "input_artifacts_loadable"
    assert check["required"] is True
    assert check["passed"] is False
    assert {error["name"] for error in check["details"]["errors"]} == {
        "promotion_audit",
        "strategy_report",
        "artifact_index",
    }
    assert {error["error_type"] for error in check["details"]["errors"]} == {
        "FileNotFoundError"
    }


def test_build_long_run_check_fails_when_index_omits_input_artifacts(tmp_path):
    promotion_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    promotion_path.write_text(json.dumps(_long_run_promotion_audit()) + "\n")
    strategy_path.write_text(json.dumps(_long_run_strategy_report()) + "\n")

    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        promotion_audit_path=str(promotion_path),
        strategy_report_path=str(strategy_path),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "artifact_index_contains_input_artifacts"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["missing_inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
    }


def test_rank_evaluation_episode_counts_include_baseline_and_head_to_head():
    counts = rank_evaluation_episode_counts(_long_run_promotion_audit()["rank"])

    assert counts["baseline_episodes"] == 40
    assert counts["candidate_baseline_episodes"] == 40
    assert counts["head_to_head_episodes"] == 4
    assert counts["total_episodes"] == 44
    assert counts["configured_baseline_episodes"] == 40
    assert counts["configured_head_to_head_episodes"] == 4
    assert counts["configured_total_episodes"] == 44
    assert counts["baseline_matchups_counted"] == 4
    assert counts["candidate_baseline_matchups_counted"] == 4
    assert counts["head_to_head_sides_counted"] == 4
    assert counts["head_to_head_map_episodes"] == {"classic": 2, "flat": 2}


def test_rank_evaluation_episode_counts_uses_nested_matchups_over_config():
    rank = _long_run_promotion_audit()["rank"]
    rank["rank_config"]["episodes_per_matchup"] = 999
    rank["head_to_head"]["overview"]["total_episodes"] = 999

    counts = rank_evaluation_episode_counts(rank)

    assert counts["baseline_episodes"] == 40
    assert counts["candidate_baseline_episodes"] == 40
    assert counts["head_to_head_episodes"] == 4
    assert counts["total_episodes"] == 44
    assert counts["configured_baseline_episodes"] == 3996
    assert counts["configured_head_to_head_episodes"] == 999
    assert counts["configured_total_episodes"] == 4995
    assert counts["head_to_head_map_episodes"] == {"classic": 2, "flat": 2}


def test_build_long_run_check_can_require_minimum_head_to_head_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_episodes=4,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_episodes=5,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "minimum_head_to_head_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "minimum_head_to_head_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["head_to_head_episodes"] == 4
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["head_to_head_episodes"] == 4
    assert failing_check["details"]["min_head_to_head_episodes"] == 5


def test_build_long_run_check_can_require_head_to_head_map_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat"),
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_map_episodes=2,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat"),
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_map_episodes=3,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "head_to_head_min_map_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "head_to_head_min_map_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["head_to_head_map_episodes"] == {
        "classic": 2,
        "flat": 2,
    }
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["low_head_to_head_maps"] == [
        {"map_name": "classic", "episode_count": 2},
        {"map_name": "flat", "episode_count": 2},
    ]


def test_build_long_run_check_rejects_config_only_episode_counts():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["rank_config"]["episodes_per_matchup"] = 999
    promotion["rank"]["head_to_head"]["overview"]["total_episodes"] = 999
    promotion["rank"].pop("suites")
    promotion["rank"]["head_to_head"].pop("matchups")

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=1,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["candidate_baseline_episodes"] == 0
    assert check["details"]["total_episodes"] == 0
    assert check["details"]["configured_total_episodes"] == 4995


def test_candidate_per_map_scores_average_matchup_scores():
    candidate = _long_run_promotion_audit()["candidate"]
    candidate["matchup_scores"].append(
        {"map_name": "flat", "score": 0.25, "episodes": 5}
    )

    assert candidate_per_map_scores(candidate) == [
        {
            "map_name": "classic",
            "mean_score": 0.5,
            "matchup_count": 1,
            "episode_count": 20,
        },
        {
            "map_name": "flat",
            "mean_score": 0.375,
            "matchup_count": 2,
            "episode_count": 25,
        },
    ]


def test_missing_required_maps_preserves_requested_order():
    assert missing_required_maps(
        ["classic", "tower"],
        ("flat", "classic", "split"),
    ) == ["flat", "split"]


def test_build_long_run_check_can_require_minimum_eval_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=40,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=41,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["candidate_baseline_episodes"] == 40
    assert failing_check["details"]["total_episodes"] == 44
    assert failing_check["details"]["min_eval_episodes"] == 41


def test_build_long_run_check_requires_candidate_eval_episodes_not_rank_total():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["suites"].append(
        {
            "label": "older",
            "checkpoint": "checkpoints/older.zip",
            "suite": {
                "matchups": {
                    "classic": {
                        "idle": {"episodes": 100},
                    }
                }
            },
        }
    )

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=41,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["baseline_episodes"] == 140
    assert check["details"]["candidate_baseline_episodes"] == 40


def test_build_long_run_check_can_require_minimum_per_map_score():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": -0.1, "episodes": 20},
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_map_score=0.0,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_min_map_score"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["low_score_maps"] == [
        {
            "map_name": "flat",
            "mean_score": -0.1,
            "matchup_count": 1,
            "episode_count": 20,
        }
    ]


def test_build_long_run_check_fails_closed_on_invalid_per_map_scores():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": "nan", "episodes": 20},
        {"map_name": "split", "episodes": 20},
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_map_score=0.0,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_map_scores_valid"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["invalid_map_scores"] == [
        {
            "map_name": "flat",
            "matchup_index": 1,
            "score": "nan",
            "reason": "invalid_score",
        },
        {
            "map_name": "split",
            "matchup_index": 2,
            "score": None,
            "reason": "missing_score",
        },
    ]
    assert check["details"]["per_map_scores"] == [
        {
            "map_name": "classic",
            "mean_score": 0.5,
            "matchup_count": 1,
            "episode_count": 20,
        }
    ]


def test_build_long_run_check_can_require_candidate_checkpoint(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_checkpoint=True,
        require_replay_analysis=True,
    )

    promotion["candidate"]["checkpoint"] = str(tmp_path / "missing.zip")
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_checkpoint=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_exists"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_exists"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False


def test_build_long_run_check_can_require_candidate_metadata(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    write_checkpoint_metadata(tmp_path / "candidate", Config(), num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    missing_metadata_checkpoint = tmp_path / "missing_metadata.zip"
    missing_metadata_checkpoint.touch()
    promotion["candidate"]["checkpoint"] = str(missing_metadata_checkpoint)
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_metadata_exists"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_metadata_exists"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert "num_timesteps" in passing_check["details"]["metadata_keys"]
    assert failing["passed"] is False
    assert failing_check["passed"] is False


def test_build_long_run_check_can_require_candidate_integrity(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.write_bytes(b"candidate-v1")
    write_checkpoint_metadata(tmp_path / "candidate", Config(), num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_candidate_integrity=True,
        require_replay_analysis=True,
    )
    checkpoint.write_bytes(b"candidate-v2")
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_candidate_integrity=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_integrity"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_integrity"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["reason"] == "sha256_mismatch"


def test_build_long_run_check_can_require_historical_opponent_samples(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    write_checkpoint_metadata(
        tmp_path / "candidate",
        Config(),
        num_timesteps=100,
        opponent_pool_stats={
            "size": 3,
            "latest_samples": 8,
            "historical_samples": 2,
            "historical_sample_rate": 0.2,
        },
    )
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_opponent_historical_samples=1,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_opponent_historical_samples=3,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_historical_opponent_samples"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_historical_opponent_samples"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["historical_samples"] == 2
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["min_opponent_historical_samples"] == 3


def test_build_long_run_check_rejects_invalid_historical_opponent_samples(tmp_path):
    for index, historical_samples in enumerate((True, -1)):
        promotion = _long_run_promotion_audit()
        checkpoint = tmp_path / f"candidate-{index}.zip"
        checkpoint.touch()
        write_checkpoint_metadata(
            tmp_path / f"candidate-{index}",
            Config(),
            num_timesteps=100,
            opponent_pool_stats={
                "size": 3,
                "latest_samples": 8,
                "historical_samples": historical_samples,
            },
        )
        promotion["candidate"]["checkpoint"] = str(checkpoint)

        result = build_long_run_check(
            promotion,
            _long_run_strategy_report(),
            _long_run_artifact_index(),
            min_maps=2,
            min_opponent_historical_samples=1,
            require_candidate_metadata=True,
            require_replay_analysis=True,
        )

        check = next(
            check
            for check in result["checks"]
            if check["id"] == "candidate_historical_opponent_samples"
        )

        assert result["passed"] is False
        assert check["passed"] is False
        assert check["details"]["historical_samples"] is None


def test_build_long_run_check_validates_candidate_metadata_required_maps(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    cfg = replace(
        Config(),
        arena=replace(
            Config().arena,
            randomize_maps=True,
            map_choices=("flat", "classic"),
        ),
    )
    write_checkpoint_metadata(tmp_path / "candidate", cfg, num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("flat", "classic"),
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("flat", "tower"),
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_metadata_required_maps"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_metadata_required_maps"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["missing_maps"] == ["tower"]


def test_checkpoint_metadata_maps_prefers_curriculum_stage_coverage():
    metadata = {
        "map_name": "classic",
        "randomize_maps": True,
        "map_choices": ["classic", "flat", "split", "tower"],
        "curriculum": {
            "active_map_pool": ["flat"],
            "stage": {"map_choices": ["flat"]},
        },
    }

    assert checkpoint_metadata_maps(metadata) == ["flat"]


def test_build_long_run_check_validates_candidate_curriculum_metadata(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    cfg = replace(
        Config(),
        training=replace(Config().training, curriculum_name="map_progression"),
    )
    write_checkpoint_metadata(tmp_path / "candidate", cfg, num_timesteps=3_000_000)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="anti_stall",
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="default",
        require_replay_analysis=True,
    )

    stage_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_metadata_curriculum_stage"
    )
    reward_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_metadata_reward_preset"
    )

    assert passing["passed"] is True
    assert stage_check["details"]["actual_curriculum_stage"] == "full_map_pool"
    assert failing["passed"] is False
    assert reward_check["passed"] is False
    assert reward_check["details"]["actual_reward_preset"] == "anti_stall"


def test_build_long_run_check_can_require_minimum_per_map_episodes():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": 0.5, "episodes": 3},
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_map_episodes=10,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_min_map_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["low_episode_maps"] == [
        {
            "map_name": "flat",
            "mean_score": 0.5,
            "matchup_count": 1,
            "episode_count": 3,
        }
    ]


def test_build_long_run_check_can_require_specific_maps():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat", "tower"),
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_required_maps"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["missing_maps"] == ["tower"]


def test_build_long_run_manifest_emits_non_executing_command_bundle():
    manifest = build_long_run_manifest(
        run_id="arena-test",
        timesteps=1234,
        suite_opponents="idle,scripted",
        suite_maps="flat,tower",
        rounds=3,
        replay_samples_per_bucket=2,
        replay_save_interval=5,
        opponent_pool_seed=123,
        rank_min_score=0.2,
        rank_min_win_rate=0.1,
        rank_max_draw_rate=0.8,
        rank_max_no_damage_rate=0.6,
        rank_max_low_engagement_rate=0.4,
        strategy_max_draw_rate=0.85,
        strategy_max_no_damage_rate=0.65,
        strategy_max_low_engagement_rate=0.45,
        strategy_max_idle_rate=0.7,
        strategy_max_dominant_action_rate=0.9,
        min_maps=2,
    )

    command_ids = [command["id"] for command in manifest["commands"]]

    assert manifest["artifact"] == {
        "artifact_type": "long_run_manifest",
        "schema_version": 1,
    }
    assert manifest["guardrails"] == {
        "executes_training": False,
        "deletes_artifacts": False,
        "contains_expensive_training_command": True,
    }
    assert "python scripts/self_play_sampling_smoke.py" in manifest[
        "preflight_shell_script"
    ]
    assert "python scripts/train_eval_smoke.py" in manifest["preflight_shell_script"]
    assert manifest["preflight_shell_script"].index(
        "python scripts/self_play_sampling_smoke.py"
    ) < manifest["preflight_shell_script"].index("python scripts/train_eval_smoke.py")
    assert "--opponent-pool-seed 123" in manifest["preflight_shell_script"]
    assert "--pool-seed 123" in manifest["preflight_shell_script"]
    assert "--map-pool flat,tower" in manifest["preflight_shell_script"]
    assert "self-play-sampling-summary.json" in manifest["preflight_shell_script"]
    assert "self-play-sampling-preflight.exitcode" in manifest[
        "preflight_shell_script"
    ]
    assert "python scripts/train.py --mode train" not in manifest[
        "preflight_shell_script"
    ]
    assert "TRAIN_EXIT=$?" not in manifest["preflight_shell_script"]
    assert "promotion_audit" not in manifest["preflight_shell_script"]
    assert 'exit "$PREFLIGHT_EXIT"' in manifest["preflight_shell_script"]
    assert command_ids == [
        "create_run_dirs",
        "archive_launcher",
        "self_play_sampling_smoke_preflight",
        "train_eval_smoke_preflight",
        "train",
        "checkpoint_trust_manifest",
        "promotion_audit",
        "resolve_promotion_audit",
        "audit_summary",
        "sample_replay_analysis",
        "strategy_report",
        "artifact_index",
        "resolve_validation_artifacts",
        "long_run_check",
        "long_run_status",
        "league_health",
        "final_artifact_index",
        "exit_with_long_run_check_status",
    ]
    assert "--timesteps 1234" in manifest["shell_script"]
    assert "EVAL_ROOT=evals" in manifest["shell_script"]
    assert 'cp "$0" "$EVAL_DIR/long-run-launcher.sh"' in manifest["shell_script"]
    assert "python scripts/train_eval_smoke.py" in manifest["shell_script"]
    assert 'PREFLIGHT_DIR=evals/arena-test-preflight-smoke' in manifest["shell_script"]
    assert '--output-dir "$PREFLIGHT_DIR"' in manifest["shell_script"]
    assert "--timesteps 128" in manifest["shell_script"]
    assert "--rounds 1" in manifest["shell_script"]
    assert "PREFLIGHT_EXIT=$?" in manifest["shell_script"]
    assert "preflight.exitcode" in manifest["shell_script"]
    assert "preflight.out" in manifest["shell_script"]
    assert "SELF_PLAY_SAMPLING_PREFLIGHT_EXIT=$?" in manifest["shell_script"]
    assert "self-play-sampling-preflight.out" in manifest["shell_script"]
    assert "self-play-sampling-preflight.exitcode" in manifest["shell_script"]
    assert 'if [ "$PREFLIGHT_EXIT" -ne 0 ]; then' in manifest["shell_script"]
    assert "--eval-label preflight-artifact-index" in manifest["shell_script"]
    assert "preflight-artifact-index.out" in manifest["shell_script"]
    assert "--replay-save-interval 5" in manifest["shell_script"]
    assert "--opponent-pool-seed 123" in manifest["shell_script"]
    assert "TRAIN_EXIT=$?" in manifest["shell_script"]
    assert "train.exitcode" in manifest["shell_script"]
    assert "train.out" in manifest["shell_script"]
    assert 'if [ "$TRAIN_EXIT" -ne 0 ]; then' in manifest["shell_script"]
    assert "python scripts/train.py --mode checkpoint_trust_manifest" in manifest[
        "shell_script"
    ]
    assert 'TRUSTED_CHECKPOINT_MANIFEST="$EVAL_DIR/checkpoint-trust-manifest.json"' in (
        manifest["shell_script"]
    )
    assert "--trusted-checkpoint-manifest \"$TRUSTED_CHECKPOINT_MANIFEST\"" in (
        manifest["shell_script"]
    )
    assert "--suite-maps flat,tower" in manifest["shell_script"]
    assert "--rank-min-score 0.2" in manifest["shell_script"]
    assert "--rank-min-win-rate 0.1" in manifest["shell_script"]
    assert "--rank-max-draw-rate 0.8" in manifest["shell_script"]
    assert "--rank-max-no-damage-rate 0.6" in manifest["shell_script"]
    assert "--rank-max-low-engagement-rate 0.4" in manifest["shell_script"]
    assert "--rank-min-map-score 0.0" in manifest["shell_script"]
    assert "PROMOTION_AUDIT_EXIT=$?" in manifest["shell_script"]
    assert "promotion-audit.exitcode" in manifest["shell_script"]
    assert "promotion-audit.out" in manifest["shell_script"]
    assert "MISSING_promotion.json" in manifest["shell_script"]
    assert 'if [ -f "$PROMOTION_AUDIT" ]; then' in manifest["shell_script"]
    assert "audit-summary.out" in manifest["shell_script"]
    assert "--strategy-max-draw-rate 0.85" in manifest["shell_script"]
    assert "--strategy-max-no-damage-rate 0.65" in manifest["shell_script"]
    assert "--strategy-max-low-engagement-rate 0.45" in manifest["shell_script"]
    assert "--strategy-max-idle-rate 0.7" in manifest["shell_script"]
    assert "--strategy-max-dominant-action-rate 0.9" in manifest["shell_script"]
    assert "--strategy-max-weaknesses 10" in manifest["shell_script"]
    assert "--long-run-required-maps flat,tower" in manifest["shell_script"]
    assert "--long-run-min-eval-episodes 12" in manifest["shell_script"]
    assert "--long-run-min-map-episodes 6" in manifest["shell_script"]
    assert "--long-run-min-map-score 0.0" in manifest["shell_script"]
    assert "--long-run-min-replay-combat-maps 2" in manifest["shell_script"]
    assert "--long-run-min-opponent-historical-samples" not in manifest["shell_script"]
    assert "--long-run-min-head-to-head-episodes" not in manifest["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes" not in manifest["shell_script"]
    assert "--long-run-require-candidate-checkpoint" in manifest["shell_script"]
    assert "--long-run-require-candidate-metadata" in manifest["shell_script"]
    assert "--long-run-require-candidate-integrity" in manifest["shell_script"]
    assert "--long-run-require-head-to-head" not in manifest["shell_script"]
    assert "--long-run-required-curriculum-stage full_map_pool" in manifest["shell_script"]
    assert "--long-run-required-reward-preset anti_stall" in manifest["shell_script"]
    assert "LONG_RUN_CHECK_EXIT=$?" in manifest["shell_script"]
    assert "long-run-check.exitcode" in manifest["shell_script"]
    assert "long-run-check.out" in manifest["shell_script"]
    assert "python scripts/train.py --mode long_run_status" in manifest["shell_script"]
    assert '--artifact-dir "$EVAL_ROOT"' in manifest["shell_script"]
    assert "--eval-label long-run-status" in manifest["shell_script"]
    assert "long-run-status.out" in manifest["shell_script"]
    assert "python scripts/train.py --mode league_health" in manifest["shell_script"]
    assert "--eval-label league-health" in manifest["shell_script"]
    assert "league-health.out" in manifest["shell_script"]
    assert "--eval-label final-artifact-index" in manifest["shell_script"]
    assert "final-artifact-index.out" in manifest["shell_script"]
    assert 'exit "$LONG_RUN_CHECK_EXIT"' in manifest["shell_script"]
    assert manifest["manifest_config"]["required_maps"] == ["flat", "tower"]
    assert manifest["manifest_config"]["min_eval_episodes"] == 12
    assert manifest["manifest_config"]["min_map_episodes"] == 6
    assert manifest["manifest_config"]["min_map_score"] == 0.0
    assert manifest["manifest_config"]["min_replay_combat_maps"] == 2
    assert manifest["manifest_config"]["min_opponent_historical_samples"] == 0
    assert manifest["manifest_config"]["min_head_to_head_episodes"] == 0
    assert manifest["manifest_config"]["min_head_to_head_map_episodes"] is None
    assert manifest["manifest_config"]["require_candidate_checkpoint"] is True
    assert manifest["manifest_config"]["require_candidate_metadata"] is True
    assert manifest["manifest_config"]["require_candidate_integrity"] is True
    assert manifest["manifest_config"]["require_head_to_head"] is False
    assert manifest["manifest_config"]["required_curriculum_stage"] == "full_map_pool"
    assert manifest["manifest_config"]["required_reward_preset"] == "anti_stall"
    assert manifest["manifest_config"]["preflight_dir"] == (
        "evals/arena-test-preflight-smoke"
    )
    assert not manifest["manifest_config"]["preflight_dir"].startswith(
        f"{manifest['manifest_config']['eval_dir']}/"
    )
    assert manifest["manifest_config"]["preflight_timesteps"] == 128
    assert manifest["manifest_config"]["preflight_rounds"] == 1
    assert manifest["manifest_config"]["self_play_sampling_preflight_min_maps"] == 2
    assert manifest["manifest_config"]["replay_save_interval"] == 5
    assert manifest["manifest_config"]["replay_save_interval_source"] == "user"
    assert manifest["manifest_config"]["opponent_pool_seed"] == 123
    assert manifest["manifest_config"]["rank_gate"] == {
        "min_score": 0.2,
        "min_win_rate": 0.1,
        "max_draw_rate": 0.8,
        "max_no_damage_rate": 0.6,
        "max_low_engagement_rate": 0.4,
        "min_map_score": 0.0,
    }
    assert manifest["manifest_config"]["strategy_report"] == {
        "max_draw_rate": 0.85,
        "max_no_damage_rate": 0.65,
        "max_low_engagement_rate": 0.45,
        "max_idle_rate": 0.7,
        "max_dominant_action_rate": 0.9,
        "max_weaknesses": 10,
    }
    source_control = manifest["manifest_config"]["source_control"]
    assert source_control["vcs"] == "git"
    assert isinstance(source_control["available"], bool)
    if source_control["available"]:
        assert "commit" in source_control
        assert "dirty" in source_control
        assert "status_short_count" in source_control
    assert "--long-run-require-replay-analysis" in manifest["shell_script"]
    assert 'RUN_ID=arena-test' in manifest["shell_script"]


def test_build_long_run_manifest_rejects_shell_injection_values():
    cases = [
        (
            {"required_curriculum_stage": "full_map_pool; echo injected >&2"},
            "Unknown curriculum stage",
        ),
        (
            {"required_reward_preset": "anti_stall; echo injected >&2"},
            "Unknown reward preset",
        ),
        (
            {"suite_maps": "classic; echo injected >&2"},
            "Unknown map names",
        ),
        (
            {"suite_opponents": "idle; echo injected >&2"},
            "Unknown opponent names",
        ),
        (
            {"opponent_pool_seed": -1},
            "opponent_pool_seed must be non-negative",
        ),
    ]

    for kwargs, expected_message in cases:
        try:
            build_long_run_manifest(run_id="arena-test", timesteps=1234, **kwargs)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected {kwargs} to be rejected")


def test_build_long_run_manifest_rejects_unsafe_run_ids():
    for run_id in ("", "../escape", "arena/test", "arena test", "arena;echo injected"):
        try:
            build_long_run_manifest(run_id=run_id, timesteps=1234)
        except ValueError as exc:
            assert "run_id must start" in str(exc)
        else:
            raise AssertionError(f"Expected run_id={run_id!r} to be rejected")


def test_build_long_run_manifest_indexes_early_failure_artifacts():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    assert (
        'printf "%s\\n" "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" > '
        '"$EVAL_DIR/self-play-sampling-preflight.exitcode"'
    ) in script
    assert 'printf "%s\\n" "$PREFLIGHT_EXIT" > "$EVAL_DIR/preflight.exitcode"' in script
    assert 'printf "%s\\n" "$TRAIN_EXIT" > "$EVAL_DIR/train.exitcode"' in script
    assert 'if [ "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" -ne 0 ]; then' in script
    assert 'if [ "$PREFLIGHT_EXIT" -ne 0 ]; then' in script
    assert 'if [ "$TRAIN_EXIT" -ne 0 ]; then' in script
    assert '--artifact-dir "$PREFLIGHT_DIR"' in script
    assert "--eval-label preflight-artifact-index" in script
    assert script.count("--eval-label final-artifact-index") == 4
    assert 'exit "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT"' in script
    assert 'exit "$PREFLIGHT_EXIT"' in script
    assert 'exit "$TRAIN_EXIT"' in script


def test_build_long_run_manifest_uses_missing_artifact_placeholders():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    assert 'ls -1t "$EVAL_DIR"/*_promotion.json 2>/dev/null' in script
    assert 'PROMOTION_AUDIT="$EVAL_DIR/MISSING_promotion.json"' in script
    assert 'if [ -f "$PROMOTION_AUDIT" ]; then' in script
    assert 'ls -1t "$EVAL_DIR"/*_strategy-report.json 2>/dev/null' in script
    assert 'STRATEGY_REPORT="$EVAL_DIR/MISSING_strategy-report.json"' in script
    assert 'ls -1t "$EVAL_DIR"/*_artifact-index.json 2>/dev/null' in script
    assert 'ARTIFACT_INDEX="$EVAL_DIR/MISSING_artifact-index.json"' in script


def test_build_long_run_manifest_redirects_command_logs():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    for log_name in (
        "self-play-sampling-preflight.out",
        "preflight.out",
        "train.out",
        "checkpoint-trust-manifest.out",
        "promotion-audit.out",
        "audit-summary.out",
        "replay-analysis.out",
        "strategy-report.out",
        "artifact-index.out",
        "long-run-check.out",
        "long-run-status.out",
        "league-health.out",
        "final-artifact-index.out",
    ):
        assert f' > "$EVAL_DIR/{log_name}" 2>&1' in script


def test_run_long_run_manifest_saves_json_and_launcher(tmp_path, capsys):
    output_dir = tmp_path / "manifests"

    run_long_run_manifest(
        run_id="arena-test",
        checkpoint_root="ckpts",
        eval_root="evals",
        replay_root="replays",
        timesteps=1234,
        suite_opponents="idle,scripted",
        suite_maps="flat,tower",
        rounds=3,
        replay_samples_per_bucket=1,
        replay_save_interval=5,
        opponent_pool_seed=123,
        rank_min_score=0.2,
        rank_min_win_rate=0.1,
        rank_max_draw_rate=0.8,
        rank_max_no_damage_rate=0.6,
        rank_max_low_engagement_rate=0.4,
        strategy_max_draw_rate=0.85,
        strategy_max_no_damage_rate=0.65,
        strategy_max_low_engagement_rate=0.45,
        strategy_max_idle_rate=0.7,
        strategy_max_dominant_action_rate=0.9,
        require_replay_analysis=True,
        min_maps=2,
        required_maps=("flat", "tower"),
        min_eval_episodes=None,
        min_map_episodes=None,
        min_map_score=0.0,
        min_replay_combat_maps=None,
        require_candidate_checkpoint=True,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="anti_stall",
        require_head_to_head=True,
        output_dir=str(output_dir),
        output_label="long-run-plan",
    )

    stdout = capsys.readouterr().out
    [manifest_path] = output_dir.glob("*_long-run-plan.json")
    [script_path] = output_dir.glob("*_long-run-plan.sh")
    [preflight_script_path] = output_dir.glob("*_long-run-plan.preflight.sh")
    saved = json.loads(manifest_path.read_text())
    index = build_artifact_index(output_dir)

    assert "Saved long-run manifest to" in stdout
    assert "Saved long-run launcher to" in stdout
    assert "Saved long-run preflight launcher to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "long_run_manifest",
        "schema_version": 1,
    }
    assert saved["manifest_artifact_path"] == str(manifest_path)
    assert saved["shell_script_path"] == str(script_path)
    assert saved["preflight_shell_script_path"] == str(preflight_script_path)
    assert script_path.read_text().startswith("#!/usr/bin/env bash")
    assert preflight_script_path.read_text().startswith("#!/usr/bin/env bash")
    assert "python scripts/self_play_sampling_smoke.py" in (
        preflight_script_path.read_text()
    )
    assert "python scripts/train_eval_smoke.py" in preflight_script_path.read_text()
    assert preflight_script_path.read_text().index(
        "python scripts/self_play_sampling_smoke.py"
    ) < preflight_script_path.read_text().index("python scripts/train_eval_smoke.py")
    assert "python scripts/train.py --mode train" not in (
        preflight_script_path.read_text()
    )
    assert script_path.stat().st_mode & 0o111
    assert preflight_script_path.stat().st_mode & 0o111
    assert index["artifact_counts"] == {
        "long_run_manifest": 1,
        "shell_script": 2,
    }
    manifest_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "long_run_manifest"
    )
    assert manifest_entry["summary"]["run_id"] == "arena-test"
    assert manifest_entry["summary"]["replay_save_interval"] == 5
    assert manifest_entry["summary"]["replay_save_interval_source"] == "user"
    assert manifest_entry["summary"]["opponent_pool_seed"] == 123
    assert manifest_entry["summary"]["min_eval_episodes"] == 12
    assert manifest_entry["summary"]["min_map_episodes"] == 6
    assert manifest_entry["summary"]["min_replay_combat_maps"] == 2
    assert manifest_entry["summary"]["min_opponent_historical_samples"] == 0
    assert manifest_entry["summary"]["min_head_to_head_episodes"] == 12
    assert manifest_entry["summary"]["min_head_to_head_map_episodes"] == 6
    assert manifest_entry["summary"]["require_candidate_checkpoint"] is True
    assert manifest_entry["summary"]["require_candidate_metadata"] is True
    assert manifest_entry["summary"]["require_candidate_integrity"] is True
    assert manifest_entry["summary"]["require_head_to_head"] is True
    assert manifest_entry["summary"]["required_curriculum_stage"] == "full_map_pool"
    assert manifest_entry["summary"]["required_reward_preset"] == "anti_stall"
    assert "source_dirty" in manifest_entry["summary"]
    assert "source_status_short_count" in manifest_entry["summary"]
    assert manifest_entry["summary"]["has_preflight_shell_script"] is True
    assert manifest_entry["summary"]["preflight_shell_script_path"] == str(
        preflight_script_path
    )
    assert manifest_entry["summary"]["rank_gate"]["max_draw_rate"] == 0.8
    assert manifest_entry["summary"]["rank_gate"]["min_map_score"] == 0.0
    assert manifest_entry["summary"]["strategy_report"]["max_draw_rate"] == 0.85
    assert manifest_entry["summary"]["expensive_command_ids"] == ["train"]
    script_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "shell_script"
    )
    assert script_entry["summary"]["starts_with_shebang"] is True
    assert saved["manifest_config"]["replay_save_interval"] == 5
    assert saved["manifest_config"]["opponent_pool_seed"] == 123
    assert saved["manifest_config"]["min_map_episodes"] == 6
    assert saved["manifest_config"]["min_replay_combat_maps"] == 2
    assert saved["manifest_config"]["min_opponent_historical_samples"] == 0
    assert saved["manifest_config"]["min_head_to_head_episodes"] == 12
    assert saved["manifest_config"]["min_head_to_head_map_episodes"] == 6
    assert saved["manifest_config"]["require_candidate_checkpoint"] is True
    assert saved["manifest_config"]["require_candidate_metadata"] is True
    assert saved["manifest_config"]["require_candidate_integrity"] is True
    assert saved["manifest_config"]["require_head_to_head"] is True
    assert saved["manifest_config"]["required_curriculum_stage"] == "full_map_pool"
    assert saved["manifest_config"]["required_reward_preset"] == "anti_stall"
    assert saved["manifest_config"]["preflight_dir"] == (
        "evals/arena-test-preflight-smoke"
    )
    assert saved["manifest_config"]["preflight_timesteps"] == 128
    assert saved["manifest_config"]["preflight_rounds"] == 1
    assert saved["manifest_config"]["self_play_sampling_preflight_min_maps"] == 2
    assert saved["manifest_config"]["replay_save_interval_source"] == "user"
    assert saved["manifest_config"]["rank_gate"]["max_draw_rate"] == 0.8
    assert saved["manifest_config"]["rank_gate"]["min_map_score"] == 0.0
    assert saved["manifest_config"]["strategy_report"]["max_draw_rate"] == 0.85


def test_long_run_manifest_cli_honors_required_maps(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "manifests"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_manifest",
            "--run-id",
            "cli-required",
            "--timesteps",
            "1234",
            "--suite-maps",
            "classic,flat,split,tower",
            "--long-run-required-maps",
            "flat,tower",
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "cli-required",
        ],
    )

    main()

    capsys.readouterr()
    [manifest_path] = output_dir.glob("*_cli-required.json")
    saved = json.loads(manifest_path.read_text())
    assert saved["manifest_config"]["suite_maps"] == "classic,flat,split,tower"
    assert saved["manifest_config"]["required_maps"] == ["flat", "tower"]
    assert "--long-run-required-maps flat,tower" in saved["shell_script"]


def test_long_run_manifest_cli_reports_invalid_run_id(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_manifest",
            "--run-id",
            "../escape",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parser error for invalid run ID")

    assert "run_id must start" in capsys.readouterr().err


def test_long_run_check_cli_rejects_negative_historical_opponent_threshold(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_check",
            "--long-run-min-opponent-historical-samples",
            "-1",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parser error for negative historical threshold")

    assert (
        "--long-run-min-opponent-historical-samples must be non-negative"
        in capsys.readouterr().err
    )


def test_build_long_run_status_reports_latest_manifest_execution_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest["manifest_artifact_path"] = str(manifest_path)
    manifest["shell_script_path"] = str(launcher_path)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    launcher_path.chmod(0o755)
    preflight_launcher_path.chmod(0o755)

    status = build_long_run_status(artifact_dir)

    assert status["artifact"] == {
        "artifact_type": "long_run_status",
        "schema_version": 1,
    }
    assert status["manifest_count"] == 1
    assert status["long_run_check_count"] == 0
    assert status["candidate_evidence_ready"] is False
    assert status["blocked_reason"] == "latest_launcher_not_executed"
    assert status["next_command"] == f"bash {launcher_path}"
    assert status["next_preflight_command"] == f"bash {preflight_launcher_path}"
    assert set(status["missing_evidence"]) >= {
        "self_play_sampling_preflight_exitcode",
        "self_play_sampling_preflight_summary",
        "preflight_exitcode",
        "train_exitcode",
        "promotion_audit_exitcode",
        "long_run_check_exitcode",
        "candidate_checkpoint_files",
        "real_training_replay_files",
        "latest_run_long_run_check",
    }
    latest = status["latest_manifest"]
    assert latest["run_id"] == "status-run"
    assert latest["launcher_exists"] is True
    assert latest["preflight_launcher_exists"] is True
    assert latest["preflight_launcher_path"] == str(preflight_launcher_path)
    assert latest["eval_dir_exists"] is False
    assert latest["checkpoint_file_count"] == 0
    assert latest["replay_file_count"] == 0
    assert latest["long_run_check_count"] == 0
    assert latest["passing_long_run_check_count"] == 0
    assert latest["source_dirty"] in {True, False, None}


def test_build_long_run_status_quotes_copy_paste_launcher_commands(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals with spaces"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest["manifest_artifact_path"] = str(manifest_path)
    manifest["shell_script_path"] = str(launcher_path)
    manifest["preflight_shell_script_path"] = str(preflight_launcher_path)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")

    status = build_long_run_status(artifact_dir)

    assert status["next_command"] == f"bash {shlex.quote(str(launcher_path))}"
    assert status["next_preflight_command"] == (
        f"bash {shlex.quote(str(preflight_launcher_path))}"
    )


def test_build_long_run_status_ignores_unsafe_manifest_launcher_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    outside_launcher = tmp_path / "outside.sh"
    outside_preflight = tmp_path / "outside.preflight.sh"
    outside_launcher.write_text("#!/usr/bin/env bash\n")
    outside_preflight.write_text("#!/usr/bin/env bash\n")
    manifest["shell_script_path"] = str(outside_launcher)
    manifest["preflight_shell_script_path"] = str(outside_preflight)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert latest["launcher_path"] is None
    assert latest["preflight_launcher_path"] is None
    assert latest["launcher_exists"] is False
    assert latest["preflight_launcher_exists"] is False
    assert status["next_command"] is None
    assert status["next_preflight_command"] is None


def test_build_long_run_status_reports_manifest_source_freshness(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="stale-source-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest["manifest_config"]["source_control"] = {
        "vcs": "git",
        "available": True,
        "commit": "old-commit",
        "dirty": True,
        "status_short_count": 1,
    }
    manifest_path = artifact_dir / "stale-source-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    monkeypatch.setattr(
        "scripts.train.source_control_snapshot",
        lambda: {
            "vcs": "git",
            "available": True,
            "commit": "new-commit",
            "dirty": False,
            "status_short_count": 0,
        },
    )

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert status["blocked_reason"] == "latest_manifest_source_stale"
    assert status["next_command"] is None
    assert status["next_preflight_command"] is None
    assert latest["source_current_commit"] == "new-commit"
    assert latest["source_current_dirty"] is False
    assert latest["source_commit_matches_current"] is False
    assert latest["source_manifest_clean"] is False
    assert latest["source_current_clean"] is True
    assert latest["source_safe_to_launch"] is False
    assert latest["source_stale_reasons"] == [
        "commit_mismatch",
        "manifest_created_from_dirty_worktree",
    ]
    assert status["status_config"]["source_control"]["commit"] == "new-commit"


def test_build_long_run_status_distinguishes_preflight_only_run(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    preflight_dir = Path(manifest["manifest_config"]["preflight_dir"])
    eval_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)
    sampling_summary = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": True},
        ],
    }
    (preflight_dir / "self-play-sampling-summary.json").write_text(
        json.dumps(sampling_summary) + "\n"
    )
    (eval_dir / "self-play-sampling-preflight.exitcode").write_text("0\n")
    (eval_dir / "preflight.exitcode").write_text("0\n")

    status = build_long_run_status(artifact_dir)

    assert status["blocked_reason"] == "latest_preflight_only"
    assert status["next_command"] == f"bash {launcher_path}"
    assert status["next_preflight_command"] is None
    assert "self_play_sampling_preflight_exitcode" not in status["missing_evidence"]
    assert "self_play_sampling_preflight_summary" not in status["missing_evidence"]
    assert "preflight_exitcode" not in status["missing_evidence"]
    assert set(status["missing_evidence"]) >= {
        "train_exitcode",
        "promotion_audit_exitcode",
        "long_run_check_exitcode",
        "candidate_checkpoint_files",
        "real_training_replay_files",
        "latest_run_long_run_check",
    }
    latest = status["latest_manifest"]
    assert latest["eval_dir_exists"] is True
    assert latest["preflight_dir_exists"] is True
    assert latest["expects_self_play_sampling_preflight"] is True
    assert latest["self_play_sampling_preflight_exitcode_exists"] is True
    assert latest["self_play_sampling_preflight"] == {
        "available": True,
        "path": str(preflight_dir / "self-play-sampling-summary.json"),
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "failed_checks": [],
    }
    assert latest["preflight_exitcode_exists"] is True
    assert latest["train_exitcode_exists"] is False
    assert latest["checkpoint_file_count"] == 0
    assert latest["replay_file_count"] == 0


def test_build_long_run_status_reports_failed_self_play_preflight_summary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    preflight_dir = Path(manifest["manifest_config"]["preflight_dir"])
    eval_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)
    sampling_summary = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": False,
        "historical_samples": 0,
        "historical_sample_rate": 0.0,
        "latest_samples": 64,
        "unique_maps_seen": 4,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": False},
        ],
    }
    (preflight_dir / "self-play-sampling-summary.json").write_text(
        json.dumps(sampling_summary) + "\n"
    )
    (eval_dir / "self-play-sampling-preflight.exitcode").write_text("1\n")
    (eval_dir / "preflight.exitcode").write_text("1\n")

    status = build_long_run_status(artifact_dir)

    assert "self_play_sampling_preflight_exitcode" not in status["missing_evidence"]
    assert "self_play_sampling_preflight_summary" not in status["missing_evidence"]
    latest = status["latest_manifest"]
    assert latest["self_play_sampling_preflight"] == {
        "available": True,
        "path": str(preflight_dir / "self-play-sampling-summary.json"),
        "passed": False,
        "historical_samples": 0,
        "historical_sample_rate": 0.0,
        "latest_samples": 64,
        "unique_maps_seen": 4,
        "failed_checks": ["historical_samples_meet_minimum"],
    }


def _write_fake_preflight_python(bin_dir: Path) -> Path:
    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            r"""
            import json
            import os
            import sys
            from pathlib import Path


            def option(name):
                if name not in sys.argv:
                    return None
                index = sys.argv.index(name)
                return sys.argv[index + 1]


            script = sys.argv[1] if len(sys.argv) > 1 else ""
            if script.endswith("self_play_sampling_smoke.py"):
                passed = (
                    os.environ.get("ARENA_FAKE_SELF_PLAY_PREFLIGHT_PASSED", "1")
                    == "1"
                )
                summary_output = Path(option("--summary-output"))
                summary_output.parent.mkdir(parents=True, exist_ok=True)
                output_dir = option("--output-dir")
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                historical_samples = 18 if passed else 0
                summary = {
                    "artifact": {
                        "artifact_type": "self_play_sampling_smoke",
                        "schema_version": 1,
                    },
                    "passed": passed,
                    "historical_samples": historical_samples,
                    "historical_sample_rate": 0.28125 if passed else 0.0,
                    "latest_samples": 46 if passed else 64,
                    "unique_maps_seen": 4,
                    "checks": [
                        {"id": "historical_samples_meet_minimum", "passed": passed},
                    ],
                }
                summary_output.write_text(json.dumps(summary) + "\n")
                sys.exit(0 if passed else 1)

            if script.endswith("train_eval_smoke.py"):
                output_dir = option("--output-dir")
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                sys.exit(0)

            if script.endswith("train.py"):
                sys.exit(0)

            sys.exit(0)
            """
        ).lstrip()
    )
    fake_python.chmod(0o755)
    return fake_python


def test_generated_preflight_launcher_summary_flows_to_status_and_health(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_root = tmp_path / "evals"
    bin_dir = tmp_path / "bin"
    artifact_root.mkdir()
    bin_dir.mkdir()
    _write_fake_preflight_python(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    run_id = "preflight-e2e"
    manifest = build_long_run_manifest(
        run_id=run_id,
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_root),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_root / f"{run_id}-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    preflight_launcher_path.chmod(0o755)

    subprocess.run(
        ["bash", str(preflight_launcher_path)],
        cwd=Path.cwd(),
        check=True,
    )
    status = build_long_run_status(artifact_root)
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    (eval_dir / "status.json").write_text(json.dumps(status) + "\n")
    health = build_league_health_report(artifact_root)

    preflight = status["latest_manifest"]["self_play_sampling_preflight"]
    assert preflight == {
        "available": True,
        "path": str(
            Path(manifest["manifest_config"]["preflight_dir"])
            / "self-play-sampling-summary.json"
        ),
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "failed_checks": [],
    }
    assert "self_play_sampling_preflight_summary" not in status["missing_evidence"]
    assert health["signals"]["self_play_sampling_preflight"] == {
        "available": True,
        "passed": True,
        "historical_samples": 18,
        "historical_sample_rate": 0.28125,
        "latest_samples": 46,
        "unique_maps_seen": 4,
        "failed_checks": [],
    }
    assert "self_play_sampling_preflight_failed" not in health["health"]["blockers"]


def test_generated_preflight_launcher_failure_blocks_league_health(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    monkeypatch.setenv("ARENA_FAKE_SELF_PLAY_PREFLIGHT_PASSED", "0")
    artifact_root = tmp_path / "evals"
    bin_dir = tmp_path / "bin"
    artifact_root.mkdir()
    bin_dir.mkdir()
    _write_fake_preflight_python(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    run_id = "preflight-failed-e2e"
    manifest = build_long_run_manifest(
        run_id=run_id,
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_root),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_root / f"{run_id}-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    preflight_launcher_path.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(preflight_launcher_path)],
        cwd=Path.cwd(),
        check=False,
    )
    status = build_long_run_status(artifact_root)
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    (eval_dir / "status.json").write_text(json.dumps(status) + "\n")
    health = build_league_health_report(artifact_root)

    assert completed.returncode == 1
    assert status["latest_manifest"]["self_play_sampling_preflight"]["passed"] is False
    assert status["latest_manifest"]["self_play_sampling_preflight"]["failed_checks"] == [
        "historical_samples_meet_minimum"
    ]
    assert "self_play_sampling_preflight_summary" not in status["missing_evidence"]
    assert "self_play_sampling_preflight_failed" in health["health"]["blockers"]
    assert health["signals"]["self_play_sampling_preflight"]["failed_checks"] == [
        "historical_samples_meet_minimum"
    ]


def test_build_long_run_status_requires_usable_checkpoint_and_replay_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    checkpoint_dir = Path(manifest["manifest_config"]["checkpoint_dir"])
    replay_dir = Path(manifest["manifest_config"]["replay_dir"])
    checkpoint_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    (checkpoint_dir / "ppo_final.meta.json").write_text("{}\n")
    (checkpoint_dir / "notes.txt").write_text("not a checkpoint\n")
    (replay_dir / "notes.txt").write_text("not a replay\n")
    (replay_dir / "episode_0001.json").write_text('{"episode_id": 1}\n')

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert latest["checkpoint_file_count"] == 0
    assert latest["checkpoint_total_file_count"] == 2
    assert latest["replay_file_count"] == 0
    assert latest["replay_total_file_count"] == 2
    assert "candidate_checkpoint_files" in status["missing_evidence"]
    assert "real_training_replay_files" in status["missing_evidence"]

    (checkpoint_dir / "ppo_final.zip").write_bytes(b"checkpoint")
    (replay_dir / "episode_0002.json").write_text(
        json.dumps({"episode_id": 2, "frames": []}) + "\n"
    )

    passing_status = build_long_run_status(artifact_dir)

    passing_latest = passing_status["latest_manifest"]
    assert passing_latest["checkpoint_file_count"] == 1
    assert passing_latest["checkpoint_total_file_count"] == 3
    assert passing_latest["replay_file_count"] == 1
    assert passing_latest["replay_total_file_count"] == 3
    assert "candidate_checkpoint_files" not in passing_status["missing_evidence"]
    assert "real_training_replay_files" not in passing_status["missing_evidence"]


def test_build_long_run_status_reports_checkpoint_historical_opponent_samples(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    checkpoint_dir = Path(manifest["manifest_config"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "ppo_final.zip"
    checkpoint.write_bytes(b"candidate")
    write_checkpoint_metadata(
        checkpoint_dir / "ppo_final",
        Config(),
        num_timesteps=100,
        opponent_pool_stats={
            "size": 2,
            "latest_samples": 4,
            "historical_samples": 0,
        },
    )

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    opponent_pool = latest["checkpoint_opponent_pool"]
    assert latest["min_opponent_historical_samples"] == 1
    assert opponent_pool["metadata_file_count"] == 1
    assert opponent_pool["metadata_with_opponent_pool_count"] == 1
    assert opponent_pool["max_historical_samples"] == 0
    assert opponent_pool["meets_min_opponent_historical_samples"] is False
    assert (
        opponent_pool["best_checkpoint_metadata"]["path"]
        == str(checkpoint_dir / "ppo_final.meta.json")
    )
    assert "checkpoint_historical_opponent_samples" in status["missing_evidence"]

    write_checkpoint_metadata(
        checkpoint_dir / "ppo_final",
        Config(),
        num_timesteps=200,
        opponent_pool_stats={
            "size": 3,
            "latest_samples": 8,
            "historical_samples": 2,
        },
    )

    passing_status = build_long_run_status(artifact_dir)
    passing_pool = passing_status["latest_manifest"]["checkpoint_opponent_pool"]
    assert passing_pool["max_historical_samples"] == 2
    assert passing_pool["metadata_meeting_min_count"] == 1
    assert passing_pool["meets_min_opponent_historical_samples"] is True
    assert (
        "checkpoint_historical_opponent_samples"
        not in passing_status["missing_evidence"]
    )


def test_build_long_run_status_detects_latest_passing_check(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    eval_dir.mkdir(parents=True)
    check_path = eval_dir / "status-long-run-check.json"
    check_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_check"),
                "passed": True,
                "candidate": {
                    "label": "ppo_final",
                    "checkpoint": "checkpoints/ppo_final.zip",
                    "score": 0.75,
                },
                "checks": [
                    {
                        "id": "promotion_audit_passed",
                        "required": True,
                        "passed": True,
                    }
                ],
            }
        )
        + "\n"
    )

    status = build_long_run_status(artifact_dir)

    assert status["candidate_evidence_ready"] is True
    assert status["blocked_reason"] is None
    assert status["next_command"] is None
    assert status["passing_long_run_check_count"] == 1
    latest = status["latest_manifest"]
    assert latest["eval_dir_exists"] is True
    assert latest["long_run_check_count"] == 1
    assert latest["passing_long_run_check_count"] == 1
    assert latest["latest_long_run_check"]["path"] == str(check_path)


def test_run_long_run_status_can_save_indexable_artifact(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    output_dir = tmp_path / "status"

    run_long_run_status(
        str(artifact_dir),
        output_dir=str(output_dir),
        output_label="status-report",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_status-report.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)
    [status_entry] = index["artifacts"]
    assert "Saved long-run status to" in stdout
    assert saved["blocked_reason"] == "no_long_run_manifest_found"
    assert index["artifact_counts"] == {"long_run_status": 1}
    assert status_entry["summary"]["candidate_evidence_ready"] is False
    assert status_entry["summary"]["blocked_reason"] == "no_long_run_manifest_found"
    assert status_entry["summary"]["missing_evidence"] == ["long_run_manifest"]


def test_build_long_run_manifest_auto_pins_replay_interval_for_tiny_runs():
    tiny = build_long_run_manifest(run_id="tiny-run", timesteps=128)
    long = build_long_run_manifest(run_id="long-run", timesteps=5_000_000)

    assert "--replay-save-interval 1" in tiny["shell_script"]
    assert "--long-run-require-head-to-head" not in tiny["shell_script"]
    assert "--long-run-min-head-to-head-episodes" not in tiny["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes" not in tiny["shell_script"]
    assert "--long-run-min-replay-combat-maps 4" in tiny["shell_script"]
    assert "--long-run-min-opponent-historical-samples" not in tiny["shell_script"]
    assert "--long-run-require-candidate-integrity" in tiny["shell_script"]
    assert tiny["manifest_config"]["replay_save_interval"] == 1
    assert tiny["manifest_config"]["replay_save_interval_source"] == "auto_small_run"
    assert tiny["manifest_config"]["require_head_to_head"] is False
    assert tiny["manifest_config"]["min_replay_combat_maps"] == 4
    assert tiny["manifest_config"]["min_opponent_historical_samples"] == 0
    assert tiny["manifest_config"]["min_head_to_head_episodes"] == 0
    assert tiny["manifest_config"]["min_head_to_head_map_episodes"] is None
    assert tiny["manifest_config"]["require_candidate_integrity"] is True
    assert "--replay-save-interval" not in long["shell_script"]
    assert "--long-run-require-head-to-head" in long["shell_script"]
    assert "--long-run-min-head-to-head-episodes 160" in long["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes 40" in long["shell_script"]
    assert "--long-run-min-replay-combat-maps 4" in long["shell_script"]
    assert "--long-run-min-opponent-historical-samples 1" in long["shell_script"]
    assert "--long-run-require-candidate-integrity" in long["shell_script"]
    assert long["manifest_config"]["replay_save_interval"] is None
    assert long["manifest_config"]["replay_save_interval_source"] == "config"
    assert long["manifest_config"]["require_head_to_head"] is True
    assert long["manifest_config"]["min_replay_combat_maps"] == 4
    assert long["manifest_config"]["min_opponent_historical_samples"] == 1
    assert long["manifest_config"]["min_head_to_head_episodes"] == 160
    assert long["manifest_config"]["min_head_to_head_map_episodes"] == 40
    assert long["manifest_config"]["require_candidate_integrity"] is True

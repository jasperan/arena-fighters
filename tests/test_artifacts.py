"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


def test_build_artifact_index_summarizes_artifacts_and_links(tmp_path):
    eval_path = tmp_path / "before.json"
    comparison_path = tmp_path / "comparison.json"
    audit_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    long_run_check_path = tmp_path / "long-run-check.json"
    eval_path.write_text(json.dumps(_eval_summary("before")) + "\n")
    comparison_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("comparison"),
                "before_path": "evals/before.json",
                "after_path": "evals/after.json",
                "deltas": {
                    "win_rate_agent_0": 0.25,
                    "draw_rate": -0.1,
                },
            }
        )
        + "\n"
    )
    audit_path.write_text(json.dumps(_promotion_audit_summary()) + "\n")
    strategy_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("strategy_report"),
                "issue_count": 2,
                "issues": [
                    {"scope": "candidate:ppo_final", "metric": "draw_rate"},
                    {"scope": "suite:flat/idle", "metric": "no_damage_rate"},
                ],
                "scanned_artifacts": 4,
            }
        )
        + "\n"
    )
    long_run_check_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_check"),
                "passed": False,
                "candidate": {"label": "ppo_final", "score": 0.25},
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

    index = build_artifact_index(tmp_path)

    assert index["artifact"] == {
        "artifact_type": "artifact_index",
        "schema_version": 1,
    }
    assert index["index_config"]["artifact_count"] == 5
    assert index["artifact_counts"] == {
        "comparison": 1,
        "eval": 1,
        "long_run_check": 1,
        "promotion_audit": 1,
        "strategy_report": 1,
    }

    comparison = next(
        entry for entry in index["artifacts"] if entry["artifact_type"] == "comparison"
    )
    assert comparison["relative_path"] == "comparison.json"
    assert comparison["summary"] == {
        "delta_count": 2,
        "win_rate_delta": 0.25,
        "draw_rate_delta": -0.1,
    }
    assert comparison["links"] == {
        "before_path": "evals/before.json",
        "after_path": "evals/after.json",
    }

    audit = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "promotion_audit"
    )
    assert audit["summary"]["passed"] is True
    assert audit["links"]["rank_artifact_path"] == "evals/rank.json"

    strategy = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "strategy_report"
    )
    assert strategy["summary"] == {
        "issue_count": 2,
        "weakness_count": 0,
        "worst_weakness": None,
        "skipped_artifact_count": 0,
        "invalid_matchup_metric_count": 0,
        "candidate_invalid_matchup_metric_count": 0,
        "invalid_matchup_metrics": [],
        "candidate_issue_count": 1,
        "smoke_issue_count": 0,
        "issue_metrics": ["draw_rate", "no_damage_rate"],
        "scanned_artifacts": 4,
    }

    long_run_check = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "long_run_check"
    )
    assert long_run_check["summary"] == {
        "passed": False,
        "candidate_label": "ppo_final",
        "candidate_score": 0.25,
        "required_check_count": 2,
        "failed_required_check_count": 1,
        "failed_required_checks": ["no_candidate_bad_strategy_issues"],
    }


def test_build_artifact_index_summarizes_rank_map_scores(tmp_path):
    rank = _rank_summary(label="candidate", score=0.5)
    rank["rankings"][0]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": -0.25, "episodes": 20},
        {"map_name": "split", "score": "nan", "episodes": 20},
    ]
    (tmp_path / "rank.json").write_text(json.dumps(rank) + "\n")

    index = build_artifact_index(tmp_path)

    [rank_entry] = [
        entry for entry in index["artifacts"] if entry["artifact_type"] == "rank"
    ]
    assert rank_entry["summary"] == {
        "checkpoint_count": 1,
        "top_label": "candidate",
        "top_score": 0.5,
        "top_map_count": 2,
        "top_worst_map_name": "flat",
        "top_worst_map_score": -0.25,
        "top_invalid_map_score_count": 1,
        "ranking_labels": ["candidate"],
        "rank_config": {},
    }


def test_build_artifact_index_summarizes_smoke_suite_artifacts(tmp_path):
    smoke_suite_path = tmp_path / "smoke-suite-summary.json"
    smoke_suite_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("smoke_suite"),
                "smoke_count": 3,
                "smoke_order": [
                    "reward_shaping",
                    "self_play_sampling",
                    "long_run_artifact",
                ],
                "compute_classes": {
                    "reward_shaping": "no_training_eval",
                    "self_play_sampling": "no_training_self_play",
                    "long_run_artifact": "no_training_artifact",
                },
                "smokes": {
                    "reward_shaping": {
                        "strategy_issue_count": 15,
                        "indexed_artifact_count": 11,
                        "idle_rate_delta_agent_0": -0.25,
                        "dominant_action_rate_delta_agent_0": -0.1,
                        "no_damage_episodes_delta": -1,
                        "low_engagement_episodes_delta": -1,
                        "damage_events_delta_agent_0": 2,
                    },
                    "self_play_sampling": {
                        "passed": True,
                        "historical_samples": 12,
                        "unique_maps_seen": 4,
                    },
                    "long_run_artifact": {
                        "health_ready": False,
                        "health_blockers": ["long_run_status_blocked"],
                        "health_warnings": ["missing_rank"],
                    },
                },
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"smoke_suite": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "smoke_suite"
    assert entry["summary"] == {
        "smoke_count": 3,
        "smoke_order": [
            "reward_shaping",
            "self_play_sampling",
            "long_run_artifact",
        ],
        "compute_classes": {
            "reward_shaping": "no_training_eval",
            "self_play_sampling": "no_training_self_play",
            "long_run_artifact": "no_training_artifact",
        },
        "summary_paths": {},
        "reward_strategy_issue_count": 15,
        "reward_indexed_artifact_count": 11,
        "reward_idle_rate_delta_agent_0": -0.25,
        "reward_dominant_action_rate_delta_agent_0": -0.1,
        "reward_no_damage_episodes_delta": -1,
        "reward_low_engagement_episodes_delta": -1,
        "reward_damage_events_delta_agent_0": 2,
        "self_play_sampling_passed": True,
        "self_play_sampling_historical_samples": 12,
        "self_play_sampling_unique_maps_seen": 4,
        "long_run_artifact_health_ready": False,
        "long_run_artifact_health_blockers": ["long_run_status_blocked"],
        "long_run_artifact_health_warnings": ["missing_rank"],
        "train_eval_long_run_check_passed": None,
    }


def test_build_artifact_index_summarizes_self_play_sampling_smoke_artifacts(tmp_path):
    sampling_path = tmp_path / "sampling-summary.json"
    sampling_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("self_play_sampling_smoke"),
                "passed": True,
                "latest_samples": 52,
                "historical_samples": 12,
                "historical_sample_rate": 0.1875,
                "unique_maps_seen": 4,
                "map_counts": {"classic": 12, "flat": 20},
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"self_play_sampling_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "self_play_sampling_smoke"
    assert entry["summary"] == {
        "passed": True,
        "latest_samples": 52,
        "historical_samples": 12,
        "historical_sample_rate": 0.1875,
        "unique_maps_seen": 4,
        "map_counts": {"classic": 12, "flat": 20},
    }


def test_build_artifact_index_summarizes_reward_shaping_smoke_artifacts(tmp_path):
    reward_summary_path = tmp_path / "reward-summary.json"
    reward_summary_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("reward_shaping_smoke"),
                "reward_delta_agent_0": -13.5,
                "reward_delta_agent_1": -13.5,
                "draw_rate_delta": 0.0,
                "idle_rate_delta_agent_0": -0.25,
                "dominant_action_rate_delta_agent_0": -0.1,
                "no_damage_episodes_delta": -1,
                "low_engagement_episodes_delta": -1,
                "damage_events_delta_agent_0": 2,
                "strategy_issue_count": 15,
                "indexed_artifact_count": 11,
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"reward_shaping_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "reward_shaping_smoke"
    assert entry["summary"] == {
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "idle_rate_delta_agent_0": -0.25,
        "dominant_action_rate_delta_agent_0": -0.1,
        "no_damage_episodes_delta": -1,
        "low_engagement_episodes_delta": -1,
        "damage_events_delta_agent_0": 2,
        "strategy_issue_count": 15,
        "indexed_artifact_count": 11,
    }


def test_build_artifact_index_summarizes_long_run_artifact_smoke_artifacts(tmp_path):
    smoke_path = tmp_path / "artifact-smoke-summary.json"
    smoke_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_artifact_smoke"),
                "run_id": "artifact-smoke",
                "status_blocked_reason": "latest_launcher_not_executed",
                "status_missing_evidence": ["train_exitcode"],
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "health_artifact_scope_dir": "/tmp/evals/artifact-smoke",
                "self_play_sampling_preflight_state": "failed",
                "status_self_play_sampling_preflight": {
                    "passed": False,
                    "failed_checks": ["historical_samples_meet_minimum"],
                },
                "indexed_artifact_counts": {
                    "long_run_manifest": 1,
                    "long_run_status": 1,
                    "league_health": 1,
                },
                "indexed_artifact_count": 3,
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"long_run_artifact_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "long_run_artifact_smoke"
    assert entry["summary"] == {
        "run_id": "artifact-smoke",
        "status_blocked_reason": "latest_launcher_not_executed",
        "status_missing_evidence": ["train_exitcode"],
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "health_artifact_scope_dir": "/tmp/evals/artifact-smoke",
        "self_play_sampling_preflight_state": "failed",
        "self_play_sampling_preflight_passed": False,
        "self_play_sampling_preflight_failed_checks": [
            "historical_samples_meet_minimum"
        ],
        "indexed_artifact_count": 3,
        "indexed_long_run_manifest_count": 1,
        "indexed_long_run_status_count": 1,
        "indexed_league_health_count": 1,
    }


def test_run_artifact_index_can_save_manifest(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    (artifact_dir / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )
    output_dir = tmp_path / "indexes"

    run_artifact_index(
        str(artifact_dir),
        recursive=False,
        output_dir=str(output_dir),
        output_label="artifact manifest",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_artifact-manifest.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved artifact index to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "artifact_index",
        "schema_version": 1,
    }
    assert saved["artifact_counts"] == {"promotion_audit": 1}


def test_build_artifact_index_summarizes_exit_code_sidecars(tmp_path):
    (tmp_path / "promotion-audit.exitcode").write_text("0\n")
    (tmp_path / "long-run-check.exitcode").write_text("1\n")

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"exit_code": 2}
    summaries = {
        Path(entry["path"]).name: entry["summary"]
        for entry in index["artifacts"]
        if entry["artifact_type"] == "exit_code"
    }
    assert summaries == {
        "long-run-check.exitcode": {"exit_code": 1, "passed": False, "raw": "1"},
        "promotion-audit.exitcode": {"exit_code": 0, "passed": True, "raw": "0"},
    }


def test_build_artifact_index_summarizes_shell_script_sidecars(tmp_path):
    launcher = tmp_path / "long-run-launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n")
    launcher.chmod(0o755)

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"shell_script": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "shell_script"
    assert entry["summary"] == {
        "line_count": 2,
        "executable": True,
        "starts_with_shebang": True,
    }


def test_build_artifact_index_summarizes_command_logs(tmp_path):
    log_path = tmp_path / "train.out"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(30)) + "\n")

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"command_log": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "command_log"
    assert entry["summary"]["tail_truncated"] is False
    assert entry["summary"]["tail_byte_limit"] == 8192
    assert entry["summary"]["tail_lines"] == [f"line {idx}" for idx in range(10, 30)]


def test_build_artifact_index_redacts_command_log_secrets(tmp_path):
    log_path = tmp_path / "train.out"
    log_path.write_text(
        "\n".join(
            [
                "api_key=abc123",
                "TOKEN: xyz789",
                "OPENAI_API_KEY=sk-local",
                "AWS_SECRET_ACCESS_KEY = aws-secret",
                "MY_TOKEN: custom-token",
                "client_secret='quoted-secret'",
                "Authorization: Bearer opaque-token",
                "Authorization: Basic opaque-basic-token",
                "Cookie: session=abc123",
                "DATABASE_URL=postgres://user:db-pass@localhost/db",
                "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----abc",
                "password = swordfish",
                "python script.py --api-key abc123 --safe value",
                "python script.py --api-key=abc123 --safe value",
                '{"token":"json-token","safe":"value"}',
                "safe line",
            ]
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "command_log"
    assert entry["summary"]["tail_lines"] == [
        "api_key=<redacted>",
        "TOKEN: <redacted>",
        "OPENAI_API_KEY=<redacted>",
        "AWS_SECRET_ACCESS_KEY = <redacted>",
        "MY_TOKEN: <redacted>",
        "client_secret=<redacted>",
        "Authorization: Bearer <redacted>",
        "Authorization: Basic <redacted>",
        "Cookie: <redacted>",
        "DATABASE_URL=<redacted>",
        "PRIVATE_KEY=<redacted>",
        "password = <redacted>",
        "python script.py --api-key <redacted> --safe value",
        "python script.py --api-key=<redacted> --safe value",
        '{"token":"<redacted>","safe":"value"}',
        "safe line",
    ]


def test_build_artifact_index_skips_symlinked_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside_log = tmp_path / "outside.out"
    outside_log.write_text("secret_token=outside\n")
    symlink_path = artifact_dir / "train.out"
    try:
        symlink_path.symlink_to(outside_log)
    except OSError:
        return

    index = build_artifact_index(artifact_dir)

    assert index["artifact_counts"] == {}
    assert index["artifacts"] == []


def test_build_strategy_report_skips_symlinked_json_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside_json = tmp_path / "outside.json"
    outside_json.write_text(json.dumps(_eval_summary("outside", draw_rate=1.0)) + "\n")
    symlink_path = artifact_dir / "eval.json"
    try:
        symlink_path.symlink_to(outside_json)
    except OSError:
        return

    report = build_strategy_report(artifact_dir, recursive=True)

    assert report["scanned_artifacts"] == 0
    assert report["skipped_artifacts"] == []
    assert report["issue_count"] == 0


def test_run_analyze_replay_can_save_indexable_artifact(tmp_path, capsys):
    replay_path = tmp_path / "episode_0007.json"
    replay_path.write_text(
        json.dumps(
            {
                "episode_id": 7,
                "winner": "agent_0",
                "length": 3,
                "map_name": "split",
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1,
                        "damage_dealt": 10,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": 10,
                    },
                },
                "frames": [
                    {
                        "tick": 3,
                        "map_name": "split",
                        "agents": {
                            "agent_0": {"hp": 30},
                            "agent_1": {"hp": 20},
                        },
                    }
                ],
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "evals"

    run_analyze_replay(
        str(replay_path),
        output_dir=str(output_dir),
        output_label="sample replay",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_sample-replay.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)

    assert "Saved replay analysis to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "replay_analysis",
        "schema_version": 1,
    }
    assert saved["flags"]["no_damage"] is False
    assert index["artifact_counts"] == {"replay_analysis": 1}
    assert index["artifacts"][0]["summary"]["flags"]["no_damage"] is False


def test_build_replay_analysis_batch_selects_representative_buckets(tmp_path):
    _write_replay(
        tmp_path / "agent0_win.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
    )
    _write_replay(
        tmp_path / "agent1_win.json",
        episode_id=2,
        winner="agent_1",
        damage_dealt=10,
        map_name="split",
    )
    _write_replay(
        tmp_path / "draw_no_damage.json",
        episode_id=3,
        winner="draw",
        damage_dealt=0,
        map_name="tower",
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["artifact"] == {
        "artifact_type": "replay_analysis_batch",
        "schema_version": 1,
    }
    assert batch["scanned_replays"] == 3
    assert batch["selected_count"] == 3
    assert {
        bucket: batch["bucket_counts"][bucket]
        for bucket in (
            "agent_0_win",
            "agent_1_win",
            "draw",
            "combat",
            "no_damage",
            "no_attacks",
        )
    } == {
        "agent_0_win": 1,
        "agent_1_win": 1,
        "draw": 1,
        "combat": 1,
        "no_damage": 1,
        "no_attacks": 1,
    }
    assert batch["bucket_counts"]["combat_map:classic"] == 1
    assert batch["bucket_counts"]["combat_map:split"] == 1
    selected_for = {
        bucket
        for item in batch["selected"]
        for bucket in item["selected_for"]
    }
    assert {
        "agent_0_win",
        "agent_1_win",
        "draw",
        "combat",
        "no_damage",
        "no_attacks",
        "combat_map:classic",
        "combat_map:split",
    } == selected_for


def test_build_replay_analysis_batch_selects_combat_samples_per_map(tmp_path):
    _write_replay(
        tmp_path / "flat_combat.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
        map_name="flat",
    )
    _write_replay(
        tmp_path / "classic_combat.json",
        episode_id=2,
        winner="agent_0",
        damage_dealt=10,
        map_name="classic",
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["selected_count"] == 2
    selected_for = {
        bucket
        for item in batch["selected"]
        for bucket in item["selected_for"]
    }
    assert "combat_map:flat" in selected_for
    assert "combat_map:classic" in selected_for


def test_build_replay_analysis_batch_selects_action_collapse_samples(tmp_path):
    replay_path = tmp_path / "idle_heavy.json"
    replay_path.write_text(
        json.dumps(
            {
                "episode_id": 5,
                "winner": "draw",
                "length": 3,
                "map_name": "flat",
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1,
                        "damage_dealt": 10,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": 10,
                    },
                },
                "frames": [
                    {"tick": 0, "map_name": "flat"},
                    {
                        "tick": 1,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                    {
                        "tick": 2,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                    {
                        "tick": 3,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                ],
            }
        )
        + "\n"
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["bucket_counts"]["idle_agent_0"] == 1
    assert batch["bucket_counts"]["dominant_action_agent_0"] == 1
    assert {
        "idle_agent_0",
        "dominant_action_agent_0",
    }.issubset(set(batch["selected"][0]["selected_for"]))


def test_run_analyze_replay_dir_saves_selected_artifacts_and_batch(
    tmp_path,
    capsys,
):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_replay(
        replay_dir / "agent0_win.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
    )
    _write_replay(
        replay_dir / "draw_no_damage.json",
        episode_id=2,
        winner="draw",
        damage_dealt=0,
    )
    output_dir = tmp_path / "evals"

    run_analyze_replay_dir(
        str(replay_dir),
        samples_per_bucket=1,
        output_dir=str(output_dir),
        output_label="sampled-replays",
    )

    stdout = capsys.readouterr().out
    index = build_artifact_index(output_dir)

    assert "Saved replay analysis batch to" in stdout
    assert index["artifact_counts"] == {
        "replay_analysis": 2,
        "replay_analysis_batch": 1,
    }
    batch_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "replay_analysis_batch"
    )
    assert batch_entry["summary"]["selected_count"] == 2
    assert batch_entry["summary"]["bucket_counts"]["no_damage"] == 1
    assert batch_entry["summary"]["bucket_counts"]["no_attacks"] == 1


def test_build_artifact_index_summarizes_strategy_report_skips(tmp_path):
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "scanned_artifacts": 1,
        "skipped_artifacts": [
            {
                "path": str(tmp_path / "bad-suite.json"),
                "relative_path": "bad-suite.json",
                "artifact_type": "suite",
                "reason": "ValueError: bad suite metric",
            }
        ],
        "weakness_count": 0,
        "weaknesses": [],
    }
    (tmp_path / "strategy.json").write_text(json.dumps(strategy_report) + "\n")

    index = build_artifact_index(tmp_path)

    [strategy_entry] = [
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "strategy_report"
    ]
    assert strategy_entry["summary"]["skipped_artifact_count"] == 1


def test_build_artifact_index_summarizes_invalid_matchup_metrics(tmp_path):
    strategy_report = {
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
            {"scope": "candidate:candidate", "metric": "draw_rate"},
        ],
        "scanned_artifacts": 2,
        "weakness_count": 0,
        "weaknesses": [],
    }
    (tmp_path / "strategy.json").write_text(json.dumps(strategy_report) + "\n")

    index = build_artifact_index(tmp_path)

    [strategy_entry] = [
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "strategy_report"
    ]
    summary = strategy_entry["summary"]
    assert summary["invalid_matchup_metric_count"] == 2
    assert summary["candidate_invalid_matchup_metric_count"] == 1
    assert summary["invalid_matchup_metrics"] == [
        "episodes",
        "win_rate_agent_0",
    ]


def test_artifact_index_contains_path_matches_absolute_and_relative_entries(tmp_path):
    artifact_path = tmp_path / "promotion.json"
    artifact_path.write_text(json.dumps(_promotion_audit_summary()) + "\n")
    index = build_artifact_index(tmp_path)

    assert artifact_index_contains_path(index, artifact_path)
    assert artifact_index_contains_path(index, str(artifact_path))

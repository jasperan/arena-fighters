import json

from scripts.reward_shaping_smoke import build_smoke_summary, write_smoke_summary


def test_build_smoke_summary_reads_expected_artifacts(tmp_path):
    (tmp_path / "20260504T000000Z_idle-default.json").write_text("{}\n")
    (tmp_path / "20260504T000001Z_idle-anti-stall.json").write_text("{}\n")
    (tmp_path / "20260504T000002Z_idle-reward-compare.json").write_text(
        json.dumps(
            {
                "deltas": {
                    "avg_rewards.agent_0": -13.5,
                    "avg_rewards.agent_1": -13.5,
                    "draw_rate": 0.0,
                    "behavior.avg_idle_rate.agent_0": 0.0,
                    "behavior.avg_dominant_action_rate.agent_0": 0.0,
                    "behavior.no_damage_episodes": 0,
                    "behavior.low_engagement_episodes": 0,
                    "behavior.damage_events.agent_0": 0,
                }
            }
        )
        + "\n"
    )
    (tmp_path / "20260504T000003Z_strategy-report.json").write_text(
        json.dumps({"issue_count": 15}) + "\n"
    )
    (tmp_path / "20260504T000004Z_artifact-index.json").write_text(
        json.dumps({"index_config": {"artifact_count": 5}}) + "\n"
    )

    summary = build_smoke_summary(tmp_path)

    assert summary == {
        "artifact": {"artifact_type": "reward_shaping_smoke", "schema_version": 1},
        "output_dir": str(tmp_path),
        "default_eval": str(tmp_path / "20260504T000000Z_idle-default.json"),
        "anti_stall_eval": str(tmp_path / "20260504T000001Z_idle-anti-stall.json"),
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "idle_rate_delta_agent_0": 0.0,
        "dominant_action_rate_delta_agent_0": 0.0,
        "no_damage_episodes_delta": 0,
        "low_engagement_episodes_delta": 0,
        "damage_events_delta_agent_0": 0,
        "strategy_issue_count": 15,
        "indexed_artifact_count": 5,
    }


def test_write_smoke_summary_creates_parent_dirs(tmp_path):
    summary = {
        "artifact": {"artifact_type": "reward_shaping_smoke", "schema_version": 1},
        "reward_delta_agent_0": -1.0,
    }
    path = tmp_path / "nested" / "reward-summary.json"

    written = write_smoke_summary(summary, path)

    assert written == path
    assert json.loads(path.read_text()) == summary

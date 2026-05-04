import json

from scripts.reward_shaping_smoke import build_smoke_summary


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
        "output_dir": str(tmp_path),
        "default_eval": str(tmp_path / "20260504T000000Z_idle-default.json"),
        "anti_stall_eval": str(tmp_path / "20260504T000001Z_idle-anti-stall.json"),
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "strategy_issue_count": 15,
        "indexed_artifact_count": 5,
    }

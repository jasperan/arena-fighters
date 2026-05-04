from dataclasses import replace
from datetime import datetime, timezone
import json

import numpy as np

from arena_fighters.config import (
    DUCK,
    IDLE,
    JUMP,
    MELEE,
    MOVE_RIGHT,
    SHOOT_DIAG_DOWN,
    SHOOT_DIAG_UP,
    Config,
)
from arena_fighters.env import ArenaFightersEnv
from arena_fighters.evaluation import (
    AggressivePolicy,
    artifact_metadata,
    RandomPolicy,
    ScriptedPolicy,
    EvasivePolicy,
    IdlePolicy,
    compare_eval_summaries,
    evaluate_baseline_suite,
    evaluate_matchup,
    evaluate_pairwise_suite,
    infer_winner,
    make_builtin_policy,
    gate_eval_comparison,
    gate_rank_summary,
    rank_baseline_suites,
    run_episode,
    score_baseline_suite,
    update_elo_ratings,
    validate_artifact,
    write_eval_summary,
)


def _short_cfg() -> Config:
    cfg = Config()
    return replace(cfg, arena=replace(cfg.arena, max_ticks=5))


def test_artifact_metadata_records_type_and_schema_version():
    assert artifact_metadata("rank") == {
        "artifact_type": "rank",
        "schema_version": 1,
    }


def test_validate_artifact_rejects_missing_or_wrong_type():
    assert validate_artifact({"artifact": artifact_metadata("eval")}, "eval") is True

    for summary, expected_message in (
        ({}, "Missing artifact metadata"),
        ({"artifact": artifact_metadata("rank")}, "Expected eval artifact"),
        (
            {"artifact": {"artifact_type": "eval", "schema_version": 999}},
            "Unsupported artifact schema version",
        ),
    ):
        try:
            validate_artifact(summary, "eval")
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("expected artifact validation to fail")


def test_run_episode_returns_metrics():
    cfg = _short_cfg()
    result = run_episode(
        cfg,
        RandomPolicy(np.random.default_rng(1)),
        RandomPolicy(np.random.default_rng(2)),
        seed=123,
    )

    assert result["winner"] in {"agent_0", "agent_1", "draw"}
    assert 1 <= result["length"] <= cfg.arena.max_ticks
    assert set(result["action_counts"]) == {"agent_0", "agent_1"}
    assert set(result["damage_dealt"]) == {"agent_0", "agent_1"}
    assert set(result["event_totals"]) == {"agent_0", "agent_1"}
    assert set(result["behavior"]["idle_rate"]) == {"agent_0", "agent_1"}
    assert set(result["behavior"]["dominant_action_rate"]) == {"agent_0", "agent_1"}
    assert isinstance(result["behavior"]["no_damage"], bool)


def test_run_episode_counts_timeout_as_draw_despite_shaped_rewards():
    cfg = replace(Config(), arena=replace(Config().arena, max_ticks=1))

    result = run_episode(cfg, IdlePolicy(), EvasivePolicy(), seed=123)

    assert result["winner"] == "draw"
    assert result["rewards"]["agent_0"] < result["rewards"]["agent_1"]
    assert result["behavior"]["low_engagement"] is True


def test_infer_winner_uses_hp_for_terminations():
    state = {
        "agents": {
            "agent_0": {"hp": 1},
            "agent_1": {"hp": 0},
        }
    }

    winner = infer_winner(
        state,
        terminations={"agent_0": True, "agent_1": True},
        truncations={"agent_0": False, "agent_1": False},
        final_rewards={"agent_0": -1.0, "agent_1": 1.0},
    )

    assert winner == "agent_0"


def test_evaluate_matchup_aggregates_episode_metrics():
    cfg = _short_cfg()
    summary = evaluate_matchup(
        cfg,
        RandomPolicy(np.random.default_rng(1)),
        RandomPolicy(np.random.default_rng(2)),
        episodes=3,
        seed=100,
    )

    assert summary["episodes"] == 3
    assert sum(summary["wins"].values()) == 3
    assert 1 <= summary["avg_length"] <= cfg.arena.max_ticks
    assert sum(summary["action_counts"]["agent_0"].values()) > 0
    assert sum(summary["action_counts"]["agent_1"].values()) > 0
    assert np.isclose(sum(summary["action_distribution"]["agent_0"].values()), 1.0)
    assert set(summary["event_totals"]) == {"agent_0", "agent_1"}
    assert summary["per_map"]["classic"]["episodes"] == 3
    assert set(summary["avg_rewards"]) == {"agent_0", "agent_1"}
    assert set(summary["per_map"]["classic"]["avg_rewards"]) == {
        "agent_0",
        "agent_1",
    }
    assert sum(summary["per_map"]["classic"]["action_counts"]["agent_0"].values()) > 0
    assert np.isclose(
        sum(summary["per_map"]["classic"]["action_distribution"]["agent_0"].values()),
        1.0,
    )
    assert set(summary["per_map"]["classic"]["damage_dealt"]) == {
        "agent_0",
        "agent_1",
    }
    assert set(summary["per_map"]["classic"]["event_totals"]) == {
        "agent_0",
        "agent_1",
    }
    assert set(summary["per_map"]["classic"]["behavior"]["avg_idle_rate"]) == {
        "agent_0",
        "agent_1",
    }
    assert "damage_events" in summary["per_map"]["classic"]["behavior"]
    assert "avg_idle_rate" in summary["behavior"]
    assert "avg_dominant_action_rate" in summary["behavior"]
    assert "low_engagement_episodes" in summary["behavior"]


def test_evaluate_matchup_reports_randomized_maps():
    cfg = _short_cfg()
    cfg = replace(
        cfg,
        arena=replace(
            cfg.arena,
            randomize_maps=True,
            map_choices=("classic", "flat"),
        ),
    )
    summary = evaluate_matchup(
        cfg,
        RandomPolicy(np.random.default_rng(1)),
        RandomPolicy(np.random.default_rng(2)),
        episodes=4,
        seed=100,
    )

    assert set(summary["per_map"]).issubset({"classic", "flat"})
    assert sum(metrics["episodes"] for metrics in summary["per_map"].values()) == 4
    assert all("no_damage_episodes" in metrics for metrics in summary["per_map"].values())
    assert all("action_counts" in metrics for metrics in summary["per_map"].values())
    assert all(
        "action_distribution" in metrics for metrics in summary["per_map"].values()
    )
    assert all("behavior" in metrics for metrics in summary["per_map"].values())


def test_scripted_policy_uses_melee_when_adjacent():
    cfg = _short_cfg()
    env = ArenaFightersEnv(config=cfg)
    obs, _ = env.reset()
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_1"].x = 11
    env._agent_states["agent_1"].y = 18

    action = ScriptedPolicy().act("agent_0", obs["agent_0"], env)

    assert action == MELEE


def test_make_builtin_policy_supports_baseline_archetypes():
    assert isinstance(make_builtin_policy("idle"), IdlePolicy)
    assert isinstance(make_builtin_policy("scripted"), ScriptedPolicy)
    assert isinstance(make_builtin_policy("aggressive"), AggressivePolicy)
    assert isinstance(make_builtin_policy("evasive"), EvasivePolicy)


def test_aggressive_policy_uses_diagonal_shots_for_vertical_targets():
    cfg = _short_cfg()
    env = ArenaFightersEnv(config=cfg)
    obs, _ = env.reset()
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_0"].shoot_cd = 0
    env._agent_states["agent_1"].x = 14
    env._agent_states["agent_1"].y = 15

    assert AggressivePolicy().act("agent_0", obs["agent_0"], env) == SHOOT_DIAG_UP

    env._agent_states["agent_1"].y = 19
    assert AggressivePolicy().act("agent_0", obs["agent_0"], env) == SHOOT_DIAG_DOWN


def test_aggressive_policy_closes_distance_to_face_target():
    cfg = _short_cfg()
    env = ArenaFightersEnv(config=cfg)
    obs, _ = env.reset()
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = -1
    env._agent_states["agent_1"].x = 20
    env._agent_states["agent_1"].y = 18

    assert AggressivePolicy().act("agent_0", obs["agent_0"], env) == MOVE_RIGHT


def test_idle_policy_always_idles():
    cfg = _short_cfg()
    env = ArenaFightersEnv(config=cfg)
    obs, _ = env.reset()

    assert IdlePolicy().act("agent_0", obs["agent_0"], env) == IDLE


def test_evasive_policy_ducks_or_jumps_under_pressure():
    cfg = _short_cfg()
    env = ArenaFightersEnv(config=cfg)
    obs, _ = env.reset()

    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_1"].x = 20
    env._agent_states["agent_1"].y = 18
    assert EvasivePolicy().act("agent_0", obs["agent_0"], env) == DUCK

    env._agent_states["agent_1"].x = 11
    assert EvasivePolicy().act("agent_0", obs["agent_0"], env) == JUMP


def test_write_eval_summary_creates_timestamped_json(tmp_path):
    summary = {"episodes": 2, "wins": {"agent_0": 1, "agent_1": 1, "draw": 0}}
    timestamp = datetime(2026, 5, 4, 3, 0, 0, tzinfo=timezone.utc)

    path = write_eval_summary(
        summary,
        tmp_path,
        label="anti stall / classic",
        timestamp=timestamp,
    )

    assert path.name == "20260504T030000Z_anti-stall-classic.json"
    assert json.loads(path.read_text()) == summary


def test_compare_eval_summaries_reports_metric_deltas():
    before = {
        "eval_config": {"reward_preset": "default"},
        "win_rate_agent_0": 0.25,
        "draw_rate": 0.5,
        "avg_length": 100.0,
        "avg_rewards": {"agent_0": -1.0, "agent_1": -1.0},
        "damage_dealt": {"agent_0": 10, "agent_1": 20},
        "action_distribution": {
            "agent_0": {IDLE: 0.75, MELEE: 0.25},
            "agent_1": {IDLE: 0.25, MELEE: 0.75},
        },
        "behavior": {
            "avg_idle_rate": {"agent_0": 0.3, "agent_1": 0.1},
            "avg_dominant_action_rate": {"agent_0": 0.6, "agent_1": 0.5},
            "damage_events": {"agent_0": 1, "agent_1": 2},
            "no_damage_episodes": 4,
            "low_engagement_episodes": 3,
        },
        "per_map": {
            "classic": {
                "episodes": 2,
                "win_rate_agent_0": 0.25,
                "win_rate_agent_1": 0.25,
                "draw_rate": 0.5,
                "avg_length": 100.0,
                "avg_rewards": {"agent_0": -1.0, "agent_1": -1.0},
                "damage_dealt": {"agent_0": 10, "agent_1": 20},
                "action_distribution": {
                    "agent_0": {IDLE: 0.75, MELEE: 0.25},
                    "agent_1": {IDLE: 0.25, MELEE: 0.75},
                },
                "behavior": {
                    "avg_idle_rate": {"agent_0": 0.3, "agent_1": 0.1},
                    "avg_dominant_action_rate": {"agent_0": 0.6, "agent_1": 0.5},
                    "damage_events": {"agent_0": 1, "agent_1": 2},
                    "no_damage_episodes": 2,
                    "low_engagement_episodes": 1,
                },
                "no_damage_episodes": 2,
                "low_engagement_episodes": 1,
            }
        },
    }
    after = {
        "eval_config": {"reward_preset": "anti_stall"},
        "win_rate_agent_0": 0.5,
        "draw_rate": 0.25,
        "avg_length": 80.0,
        "avg_rewards": {"agent_0": -5.0, "agent_1": -5.0},
        "damage_dealt": {"agent_0": 18, "agent_1": 12},
        "action_distribution": {
            "agent_0": {IDLE: 0.25, MELEE: 0.75},
            "agent_1": {IDLE: 0.5, MELEE: 0.5},
        },
        "behavior": {
            "avg_idle_rate": {"agent_0": 0.2, "agent_1": 0.1},
            "avg_dominant_action_rate": {"agent_0": 0.4, "agent_1": 0.5},
            "damage_events": {"agent_0": 3, "agent_1": 1},
            "no_damage_episodes": 1,
            "low_engagement_episodes": 0,
        },
        "per_map": {
            "classic": {
                "episodes": 2,
                "win_rate_agent_0": 0.5,
                "win_rate_agent_1": 0.25,
                "draw_rate": 0.25,
                "avg_length": 80.0,
                "avg_rewards": {"agent_0": -5.0, "agent_1": -5.0},
                "damage_dealt": {"agent_0": 18, "agent_1": 12},
                "action_distribution": {
                    "agent_0": {IDLE: 0.25, MELEE: 0.75},
                    "agent_1": {IDLE: 0.5, MELEE: 0.5},
                },
                "behavior": {
                    "avg_idle_rate": {"agent_0": 0.2, "agent_1": 0.1},
                    "avg_dominant_action_rate": {"agent_0": 0.4, "agent_1": 0.5},
                    "damage_events": {"agent_0": 3, "agent_1": 1},
                    "no_damage_episodes": 1,
                    "low_engagement_episodes": 0,
                },
                "no_damage_episodes": 1,
                "low_engagement_episodes": 0,
            }
        },
    }

    comparison = compare_eval_summaries(before, after)

    assert comparison["artifact"] == {
        "artifact_type": "comparison",
        "schema_version": 1,
    }
    assert comparison["before_config"]["reward_preset"] == "default"
    assert comparison["after_config"]["reward_preset"] == "anti_stall"
    assert comparison["deltas"]["win_rate_agent_0"] == 0.25
    assert comparison["deltas"]["avg_rewards.agent_0"] == -4.0
    assert comparison["deltas"][f"action_distribution.agent_0.{IDLE}"] == -0.5
    assert comparison["deltas"][f"action_distribution.agent_0.{MELEE}"] == 0.5
    assert comparison["deltas"]["behavior.no_damage_episodes"] == -3.0
    assert comparison["per_map"]["classic"]["deltas"]["avg_length"] == -20.0
    assert comparison["per_map"]["classic"]["deltas"]["avg_rewards.agent_0"] == -4.0
    assert comparison["per_map"]["classic"]["deltas"]["damage_dealt.agent_0"] == 8.0
    assert (
        comparison["per_map"]["classic"]["deltas"][
            f"action_distribution.agent_0.{IDLE}"
        ]
        == -0.5
    )
    assert np.isclose(
        comparison["per_map"]["classic"]["deltas"][
            "behavior.avg_idle_rate.agent_0"
        ],
        -0.1,
    )
    assert (
        comparison["per_map"]["classic"]["deltas"]["behavior.damage_events.agent_0"]
        == 2.0
    )


def test_evaluate_baseline_suite_runs_maps_and_opponents():
    cfg = _short_cfg()

    def agent0_factory(seed):
        return RandomPolicy(np.random.default_rng(seed))

    suite = evaluate_baseline_suite(
        cfg=cfg,
        agent0_policy_factory=agent0_factory,
        agent0_label="random",
        opponents=("idle", "evasive"),
        maps=("classic", "flat"),
        episodes=1,
        seed=10,
    )

    assert suite["suite_config"]["opponents"] == ["idle", "evasive"]
    assert suite["suite_config"]["maps"] == ["classic", "flat"]
    assert suite["overview"]["total_matchups"] == 4
    assert suite["overview"]["total_episodes"] == 4
    assert set(suite["matchups"]) == {"classic", "flat"}
    assert set(suite["matchups"]["classic"]) == {"idle", "evasive"}


def test_evaluate_pairwise_suite_runs_forward_and_reverse_matchups():
    cfg = _short_cfg()
    policies = {
        "idle": lambda seed: IdlePolicy(),
        "random": lambda seed: RandomPolicy(np.random.default_rng(seed)),
    }

    suite = evaluate_pairwise_suite(
        cfg=cfg,
        policy_factories=policies,
        maps=("classic",),
        episodes=1,
        seed=20,
    )

    assert suite["overview"]["unordered_pairs"] == 1
    assert suite["overview"]["ordered_matchups"] == 2
    assert suite["overview"]["total_episodes"] == 2
    assert set(suite["standings"][0]) >= {
        "label",
        "rank",
        "score",
        "wins",
        "losses",
        "draws",
        "episodes",
        "elo",
    }
    assert {row["label"] for row in suite["standings"]} == {"idle", "random"}
    assert set(suite["matchups"]["classic"]) == {"idle__vs__random"}


def test_update_elo_ratings_moves_winner_up_and_loser_down():
    ratings = {"winner": 1000.0, "loser": 1000.0}

    update_elo_ratings(ratings, "winner", "loser", score_a=1.0)

    assert round(ratings["winner"], 1) == 1016.0
    assert round(ratings["loser"], 1) == 984.0


def test_score_baseline_suite_combines_wins_and_draws():
    suite = {
        "matchups": {
            "classic": {
                "idle": {
                    "episodes": 2,
                    "win_rate_agent_0": 0.5,
                    "draw_rate": 0.25,
                    "avg_length": 10.0,
                },
                "scripted": {
                    "episodes": 2,
                    "win_rate_agent_0": 0.25,
                    "draw_rate": 0.5,
                    "avg_length": 20.0,
                },
            }
        }
    }

    score = score_baseline_suite(suite)

    assert score["score"] == 0.5625
    assert score["mean_win_rate_agent_0"] == 0.375
    assert score["mean_draw_rate"] == 0.375
    assert score["mean_no_damage_rate"] == 0.0
    assert score["mean_low_engagement_rate"] == 0.0
    assert score["mean_avg_length"] == 15.0
    assert [item["episodes"] for item in score["matchup_scores"]] == [2, 2]


def test_score_baseline_suite_penalizes_low_engagement_draws():
    suite = {
        "matchups": {
            "flat": {
                "idle": {
                    "episodes": 1,
                    "win_rate_agent_0": 0.0,
                    "draw_rate": 1.0,
                    "avg_length": 500.0,
                    "behavior": {
                        "no_damage_episodes": 1,
                        "low_engagement_episodes": 1,
                    },
                }
            }
        }
    }

    score = score_baseline_suite(suite)

    assert score["score"] == 0.0
    assert score["mean_draw_rate"] == 1.0
    assert score["mean_no_damage_rate"] == 1.0
    assert score["mean_low_engagement_rate"] == 1.0


def test_rank_baseline_suites_orders_by_score():
    weaker = {
        "label": "ppo_100",
        "checkpoint": "ppo_100.zip",
        "checkpoint_metadata": {"num_timesteps": 100},
        "suite": {
            "matchups": {
                "classic": {
                    "idle": {
                        "win_rate_agent_0": 0.25,
                        "draw_rate": 0.0,
                        "avg_length": 10.0,
                    }
                }
            }
        },
    }
    stronger = {
        "label": "ppo_200",
        "checkpoint": "ppo_200.zip",
        "checkpoint_metadata": {"num_timesteps": 200},
        "suite": {
            "matchups": {
                "classic": {
                    "idle": {
                        "win_rate_agent_0": 0.5,
                        "draw_rate": 0.5,
                        "avg_length": 8.0,
                    }
                }
            }
        },
    }

    ranking = rank_baseline_suites([weaker, stronger])

    assert ranking["score_config"] == {
        "draw_weight": 0.5,
        "no_damage_penalty": 0.25,
        "low_engagement_penalty": 0.25,
    }
    assert [item["label"] for item in ranking["rankings"]] == ["ppo_200", "ppo_100"]
    assert [item["rank"] for item in ranking["rankings"]] == [1, 2]


def test_gate_rank_summary_fails_low_score_and_low_engagement():
    summary = {
        "rankings": [
            {
                "label": "ppo_stalled",
                "score": 0.0,
                "mean_win_rate_agent_0": 0.0,
                "mean_no_damage_rate": 1.0,
                "mean_low_engagement_rate": 1.0,
            }
        ]
    }

    gate = gate_rank_summary(summary)

    assert gate["artifact"] == {"artifact_type": "rank_gate", "schema_version": 1}
    assert gate["passed"] is False
    assert gate["candidate"]["label"] == "ppo_stalled"
    assert {failure["metric"] for failure in gate["failures"]} == {
        "score",
        "mean_no_damage_rate",
        "mean_low_engagement_rate",
    }


def test_gate_rank_summary_can_limit_draw_rate():
    summary = {
        "rankings": [
            {
                "label": "draw_farmer",
                "score": 0.45,
                "mean_win_rate_agent_0": 0.0,
                "mean_draw_rate": 0.95,
                "mean_no_damage_rate": 0.0,
                "mean_low_engagement_rate": 0.0,
            }
        ]
    }

    gate = gate_rank_summary(summary, max_draw_rate=0.9)

    assert gate["passed"] is False
    assert gate["failures"] == [
        {
            "metric": "mean_draw_rate",
            "value": 0.95,
            "max": 0.9,
            "reason": "above_maximum",
        }
    ]


def test_gate_rank_summary_fails_closed_on_invalid_per_map_score():
    summary = {
        "rankings": [
            {
                "label": "candidate",
                "score": 0.5,
                "mean_win_rate_agent_0": 0.5,
                "mean_no_damage_rate": 0.0,
                "mean_low_engagement_rate": 0.0,
                "matchup_scores": [
                    {"map_name": "classic", "score": 0.25, "episodes": 4},
                    {"map_name": "flat", "score": float("nan"), "episodes": 4},
                    {"map_name": "tower", "score": "not-a-float", "episodes": 4},
                    {"map_name": "split", "episodes": 4},
                ],
            }
        ]
    }

    gate = gate_rank_summary(summary, min_map_score=0.0)

    assert gate["passed"] is False
    assert gate["failures"] == [
        {
            "metric": "per_map_score",
            "value": None,
            "min": 0.0,
            "reason": "invalid_score",
            "invalid_map_scores": [
                {
                    "map_name": "flat",
                    "matchup_index": 1,
                    "score": "nan",
                    "reason": "invalid_score",
                },
                {
                    "map_name": "tower",
                    "matchup_index": 2,
                    "score": "not-a-float",
                    "reason": "invalid_score",
                },
                {
                    "map_name": "split",
                    "matchup_index": 3,
                    "score": None,
                    "reason": "missing_score",
                },
            ],
            "low_score_maps": [],
            "per_map_scores": [
                {
                    "map_name": "classic",
                    "mean_score": 0.25,
                    "matchup_count": 1,
                    "episode_count": 4,
                }
            ],
        }
    ]


def test_gate_rank_summary_can_require_head_to_head_rating():
    summary = {
        "rankings": [
            {
                "label": "candidate",
                "score": 0.5,
                "mean_win_rate_agent_0": 0.5,
                "mean_no_damage_rate": 0.0,
                "mean_low_engagement_rate": 0.0,
            }
        ],
        "head_to_head": {
            "standings": [
                {
                    "label": "candidate",
                    "elo": 990.0,
                    "score": 0.25,
                }
            ]
        },
    }

    gate = gate_rank_summary(
        summary,
        min_head_to_head_elo=1000.0,
        min_head_to_head_score=0.5,
    )

    assert gate["passed"] is False
    assert {failure["metric"] for failure in gate["failures"]} == {
        "head_to_head.elo",
        "head_to_head.score",
    }


def test_gate_eval_comparison_passes_within_default_thresholds():
    comparison = {
        "deltas": {
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "behavior.avg_idle_rate.agent_0": 0.01,
            "behavior.no_damage_episodes": -1.0,
            "behavior.low_engagement_episodes": 0.0,
        }
    }

    gate = gate_eval_comparison(comparison)

    assert gate["artifact"] == {"artifact_type": "gate", "schema_version": 1}
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_gate_eval_comparison_fails_regressions():
    comparison = {
        "deltas": {
            "win_rate_agent_0": -0.2,
            "draw_rate": 0.2,
            "behavior.avg_idle_rate.agent_0": 0.2,
            "behavior.no_damage_episodes": 2.0,
            "behavior.low_engagement_episodes": 1.0,
        }
    }

    gate = gate_eval_comparison(comparison)

    assert gate["passed"] is False
    assert {failure["metric"] for failure in gate["failures"]} == {
        "win_rate_agent_0",
        "draw_rate",
        "behavior.avg_idle_rate.agent_0",
        "behavior.no_damage_episodes",
        "behavior.low_engagement_episodes",
    }


def test_gate_eval_comparison_fails_per_map_regressions():
    comparison = {
        "deltas": {
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "behavior.avg_idle_rate.agent_0": 0.0,
            "behavior.no_damage_episodes": 0.0,
            "behavior.low_engagement_episodes": 0.0,
        },
        "per_map": {
            "tower": {
                "deltas": {
                    "win_rate_agent_0": -0.2,
                    "draw_rate": 0.0,
                    "behavior.avg_idle_rate.agent_0": 0.0,
                    "behavior.no_damage_episodes": 0.0,
                    "behavior.low_engagement_episodes": 0.0,
                }
            }
        },
    }

    gate = gate_eval_comparison(comparison)

    assert gate["passed"] is False
    assert gate["failures"] == [
        {
            "metric": "win_rate_agent_0",
            "delta": -0.2,
            "min_delta": -0.05,
            "reason": "below_min_delta",
            "scope": "per_map:tower",
            "map_name": "tower",
        }
    ]

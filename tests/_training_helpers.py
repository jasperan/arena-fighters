from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap

from arena_fighters.config import Config, IDLE, reward_config_for_preset
from arena_fighters.evaluation import artifact_metadata
from arena_fighters.self_play import OpponentPool
from scripts.train import (
    SelfPlayCallback,
    artifact_index_contains_path,
    build_artifact_index,
    build_long_run_check,
    build_long_run_manifest,
    build_long_run_status,
    build_league_health_report,
    build_replay_analysis_batch,
    build_strategy_report,
    build_training_wrapper,
    candidate_per_map_scores,
    checkpoint_file_sha256,
    checkpoint_metadata_integrity,
    checkpoint_metadata_maps,
    checkpoint_metadata,
    checkpoint_trust_manifest,
    curriculum_metadata,
    discover_checkpoints,
    effective_reward_config,
    load_checkpoint_trust_manifest,
    load_trusted_ppo_checkpoint,
    main,
    missing_required_maps,
    parse_builtin_opponents,
    parse_csv_tuple,
    parse_rank_checkpoints,
    parse_suite_maps,
    rank_evaluation_episode_counts,
    read_checkpoint_metadata,
    run_artifact_index,
    run_audit_summary,
    run_analyze_replay,
    run_analyze_replay_dir,
    run_compare,
    run_eval,
    run_gate,
    run_long_run_check,
    run_long_run_manifest,
    run_long_run_status,
    run_league_health,
    run_promotion_audit,
    run_rank_gate,
    run_suite,
    run_strategy_report,
    summarize_promotion_audit,
    verify_checkpoint_trust,
    write_checkpoint_metadata,
)


class FakeWrapper:
    def __init__(self):
        self.map_pools = []
        self.reward_configs = []

    def set_map_pool(self, map_choices):
        self.map_pools.append(map_choices)

    def set_reward_config(self, reward_config):
        self.reward_configs.append(reward_config)


class FakeLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class FakeModelWithLogger:
    def __init__(self):
        self.logger = FakeLogger()


class FakePredictModel:
    def predict(self, obs, deterministic=True):
        return IDLE, None


def _clean_source_snapshot():
    return {
        "vcs": "git",
        "available": True,
        "commit": "clean-commit",
        "dirty": False,
        "status_short_count": 0,
    }


def _write_replay(
    path: Path,
    *,
    episode_id: int,
    winner: str,
    damage_dealt: int,
    map_name: str = "classic",
) -> None:
    path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "winner": winner,
                "length": 3,
                "map_name": map_name,
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1 if damage_dealt else 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1 if damage_dealt else 0,
                        "damage_dealt": damage_dealt,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": damage_dealt,
                    },
                },
                "frames": [
                    {
                        "tick": 3,
                        "map_name": map_name,
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


def _eval_summary(
    label: str,
    win_rate: float = 0.5,
    draw_rate: float = 0.0,
    idle_rate: float = 0.1,
    no_damage_episodes: int = 0,
    low_engagement_episodes: int = 0,
):
    return {
        "artifact": artifact_metadata("eval"),
        "eval_config": {"label": label},
        "episodes": 4,
        "win_rate_agent_0": win_rate,
        "win_rate_agent_1": 1.0 - win_rate - draw_rate,
        "draw_rate": draw_rate,
        "avg_length": 10.0,
        "avg_rewards": {"agent_0": -1.0, "agent_1": -1.0},
        "damage_dealt": {"agent_0": 2.0, "agent_1": 1.0},
        "behavior": {
            "avg_idle_rate": {"agent_0": idle_rate, "agent_1": 0.1},
            "avg_dominant_action_rate": {"agent_0": 0.4, "agent_1": 0.4},
            "damage_events": {"agent_0": 2, "agent_1": 1},
            "no_damage_episodes": no_damage_episodes,
            "low_engagement_episodes": low_engagement_episodes,
        },
    }


def _rank_summary(
    label: str = "candidate",
    score: float = 0.5,
    win_rate: float = 0.5,
    no_damage_rate: float = 0.0,
    low_engagement_rate: float = 0.0,
):
    return {
        "artifact": artifact_metadata("rank"),
        "rankings": [
            {
                "label": label,
                "score": score,
                "mean_win_rate_agent_0": win_rate,
                "mean_no_damage_rate": no_damage_rate,
                "mean_low_engagement_rate": low_engagement_rate,
            }
        ],
    }


def _promotion_audit_summary(passed: bool = True):
    return {
        "artifact": artifact_metadata("promotion_audit"),
        "passed": passed,
        "rank_artifact_path": "evals/rank.json",
        "rank_gate_artifact_path": "evals/rank-gate.json",
        "ranking_labels": ["candidate", "older"],
        "rules": {"min_score": 0.1},
        "candidate": {
            "label": "candidate",
            "checkpoint": "checkpoints/candidate.zip",
            "rank": 1,
            "score": 0.5,
            "mean_win_rate_agent_0": 0.5,
            "mean_no_damage_rate": 0.0,
            "mean_low_engagement_rate": 0.0,
        },
        "failures": [] if passed else [{"metric": "score", "reason": "below"}],
    }


def _long_run_promotion_audit():
    summary = _promotion_audit_summary()
    summary["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": 0.5, "episodes": 20},
    ]
    summary["rank"] = {
        "artifact": artifact_metadata("rank"),
        "rank_config": {
            "checkpoints": ["checkpoints/candidate.zip"],
            "maps": ["classic", "flat"],
            "opponents": ["idle", "scripted"],
            "episodes_per_matchup": 10,
        },
        "rankings": [
            {
                "label": "candidate",
                "checkpoint": "checkpoints/candidate.zip",
                "rank": 1,
                "score": 0.5,
            }
        ],
        "suites": [
            {
                "label": "candidate",
                "checkpoint": "checkpoints/candidate.zip",
                "suite": {
                    "matchups": {
                        "classic": {
                            "idle": {"episodes": 10},
                            "scripted": {"episodes": 10},
                        },
                        "flat": {
                            "idle": {"episodes": 10},
                            "scripted": {"episodes": 10},
                        },
                    }
                },
            }
        ],
        "head_to_head": {
            "overview": {"total_episodes": 4},
            "standings": [
                {"label": "candidate", "score": 1.0},
                {"label": "older", "score": 0.0},
            ],
            "matchups": {
                "classic": {
                    "candidate__vs__older": {
                        "forward": {"episodes": 1},
                        "reverse": {"episodes": 1},
                    }
                },
                "flat": {
                    "candidate__vs__older": {
                        "forward": {"episodes": 1},
                        "reverse": {"episodes": 1},
                    }
                },
            },
        },
    }
    return summary


def _long_run_strategy_report(
    candidate_issue: bool = False,
    replay_issue: bool = False,
):
    issues = []
    if candidate_issue:
        issues.append(
            {
                "scope": "candidate:candidate",
                "metric": "mean_no_damage_rate",
                "value": 1.0,
            }
        )
    if replay_issue:
        issues.append(
            {
                "scope": "replay:episode_0001",
                "metric": "replay_idle_rate_agent_0",
                "value": 1.0,
                "threshold": 0.75,
            }
        )
    return {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": len(issues),
        "issues": issues,
    }


def _long_run_artifact_index(
    replay_analysis: bool = True,
    replay_maps: tuple[str, ...] = ("classic",),
):
    artifacts = []
    counts = {
        "promotion_audit": 1,
        "strategy_report": 1,
        "rank": 1,
        "rank_gate": 1,
    }
    if replay_analysis:
        counts["replay_analysis"] = len(replay_maps)
        for map_name in replay_maps:
            artifacts.append(
                {
                    "artifact_type": "replay_analysis",
                    "summary": {
                        "map_name": map_name,
                        "flags": {"no_damage": False},
                        "totals": {"damage_dealt": 10},
                    },
                }
            )
    return {
        "artifact": artifact_metadata("artifact_index"),
        "artifact_counts": counts,
        "artifacts": artifacts,
    }


__all__ = [
    'replace',
    'json',
    'os',
    'Path',
    'shlex',
    'subprocess',
    'sys',
    'textwrap',
    'Config',
    'IDLE',
    'reward_config_for_preset',
    'artifact_metadata',
    'OpponentPool',
    'SelfPlayCallback',
    'artifact_index_contains_path',
    'build_artifact_index',
    'build_long_run_check',
    'build_long_run_manifest',
    'build_long_run_status',
    'build_league_health_report',
    'build_replay_analysis_batch',
    'build_strategy_report',
    'build_training_wrapper',
    'candidate_per_map_scores',
    'checkpoint_file_sha256',
    'checkpoint_metadata_integrity',
    'checkpoint_metadata_maps',
    'checkpoint_metadata',
    'checkpoint_trust_manifest',
    'curriculum_metadata',
    'discover_checkpoints',
    'effective_reward_config',
    'load_checkpoint_trust_manifest',
    'load_trusted_ppo_checkpoint',
    'main',
    'missing_required_maps',
    'parse_builtin_opponents',
    'parse_csv_tuple',
    'parse_rank_checkpoints',
    'parse_suite_maps',
    'rank_evaluation_episode_counts',
    'read_checkpoint_metadata',
    'run_artifact_index',
    'run_audit_summary',
    'run_analyze_replay',
    'run_analyze_replay_dir',
    'run_compare',
    'run_eval',
    'run_gate',
    'run_long_run_check',
    'run_long_run_manifest',
    'run_long_run_status',
    'run_league_health',
    'run_promotion_audit',
    'run_rank_gate',
    'run_suite',
    'run_strategy_report',
    'summarize_promotion_audit',
    'verify_checkpoint_trust',
    'write_checkpoint_metadata',
    'FakeWrapper',
    'FakeLogger',
    'FakeModelWithLogger',
    'FakePredictModel',
    '_clean_source_snapshot',
    '_write_replay',
    '_eval_summary',
    '_rank_summary',
    '_promotion_audit_summary',
    '_long_run_promotion_audit',
    '_long_run_strategy_report',
    '_long_run_artifact_index',
]

"""Evaluation helpers for checkpoint and baseline matchups."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from arena_fighters.config import (
    DUCK,
    IDLE,
    JUMP,
    MELEE,
    MOVE_LEFT,
    MOVE_RIGHT,
    NUM_ACTIONS,
    SHOOT_FORWARD,
    SHOOT_DIAG_DOWN,
    SHOOT_DIAG_UP,
    Config,
)
from arena_fighters.env import ArenaFightersEnv


DEFAULT_GATE_RULES = {
    "win_rate_agent_0": {"min_delta": -0.05},
    "draw_rate": {"max_delta": 0.05},
    "behavior.avg_idle_rate.agent_0": {"max_delta": 0.05},
    "behavior.no_damage_episodes": {"max_delta": 0.0},
    "behavior.low_engagement_episodes": {"max_delta": 0.0},
}

ARTIFACT_SCHEMA_VERSION = 1


def artifact_metadata(artifact_type: str) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }


def validate_artifact(
    summary: dict[str, Any],
    expected_type: str,
    expected_schema_version: int = ARTIFACT_SCHEMA_VERSION,
) -> bool:
    artifact = summary.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("Missing artifact metadata")
    artifact_type = artifact.get("artifact_type")
    if artifact_type != expected_type:
        raise ValueError(
            f"Expected {expected_type} artifact, got {artifact_type or 'unknown'}"
        )
    schema_version = artifact.get("schema_version")
    if schema_version != expected_schema_version:
        raise ValueError(
            "Unsupported artifact schema version: "
            f"{schema_version} (expected {expected_schema_version})"
        )
    return True


def mirror_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grid = obs["grid"].copy()
    vector = obs["vector"].copy()

    grid = np.flip(grid, axis=2).copy()
    grid[[1, 2]] = grid[[2, 1]]
    grid[[3, 4]] = grid[[4, 3]]
    vector[0], vector[1] = vector[1], vector[0]

    return {"grid": grid, "vector": vector}


class EvalPolicy(Protocol):
    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        ...


@dataclass
class RandomPolicy:
    rng: np.random.Generator

    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        return int(self.rng.integers(NUM_ACTIONS))


@dataclass
class IdlePolicy:
    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        return IDLE


@dataclass
class ScriptedPolicy:
    """Simple combat baseline for sanity-checking trained checkpoints."""

    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        st = env._agent_states[agent_name]
        other = env._agent_states[env._other(agent_name)]
        dx = other.x - st.x
        dy = other.y - st.y
        target_facing = 1 if dx > 0 else -1

        can_attack = dy == 0 and st.facing == target_facing
        if abs(dx) == 1 and can_attack and st.melee_cd <= 0:
            return MELEE
        if dx != 0 and can_attack and st.shoot_cd <= 0:
            return SHOOT_FORWARD
        if dx < 0:
            return MOVE_LEFT
        if dx > 0:
            return MOVE_RIGHT
        if other.y < st.y and env._on_ground(st):
            return JUMP
        return IDLE


@dataclass
class AggressivePolicy:
    """Pressure baseline that attacks while closing distance across elevations."""

    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        st = env._agent_states[agent_name]
        other = env._agent_states[env._other(agent_name)]
        dx = other.x - st.x
        dy = other.y - st.y
        target_facing = 1 if dx > 0 else -1 if dx < 0 else st.facing
        facing_target = dx == 0 or st.facing == target_facing

        if abs(dx) == 1 and dy == 0 and facing_target and st.melee_cd <= 0:
            return MELEE

        if dx != 0 and facing_target and st.shoot_cd <= 0:
            if dy < 0:
                return SHOOT_DIAG_UP
            if dy > 0:
                return SHOOT_DIAG_DOWN
            return SHOOT_FORWARD

        if dx < 0:
            return MOVE_LEFT
        if dx > 0:
            return MOVE_RIGHT
        if other.y < st.y and env._on_ground(st):
            return JUMP
        return IDLE


@dataclass
class EvasivePolicy:
    """Baseline that prioritizes avoiding direct horizontal pressure."""

    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        st = env._agent_states[agent_name]
        other = env._agent_states[env._other(agent_name)]
        dx = other.x - st.x
        dy = other.y - st.y

        if dy == 0 and abs(dx) > 2:
            return DUCK
        if abs(dx) <= 2 and env._on_ground(st):
            return JUMP
        if dx > 0:
            return MOVE_LEFT
        if dx < 0:
            return MOVE_RIGHT
        return IDLE


@dataclass
class ModelPolicy:
    model: Any
    deterministic: bool = True

    def act(
        self,
        agent_name: str,
        obs: dict[str, np.ndarray],
        env: ArenaFightersEnv,
    ) -> int:
        model_obs = obs
        if agent_name == "agent_1":
            model_obs = mirror_obs(obs)
        action, _ = self.model.predict(model_obs, deterministic=self.deterministic)
        return int(action)


def make_builtin_policy(name: str, seed: int | None = None) -> EvalPolicy:
    if name == "random":
        return RandomPolicy(np.random.default_rng(seed))
    if name == "idle":
        return IdlePolicy()
    if name == "scripted":
        return ScriptedPolicy()
    if name == "aggressive":
        return AggressivePolicy()
    if name == "evasive":
        return EvasivePolicy()
    raise ValueError(f"Unknown built-in policy: {name}")


BUILTIN_POLICY_NAMES = ("random", "scripted", "idle", "aggressive", "evasive")


def infer_winner(
    state: dict[str, Any],
    terminations: dict[str, bool],
    truncations: dict[str, bool],
    final_rewards: dict[str, float],
) -> str:
    if any(truncations.values()):
        return "draw"

    agents = state.get("agents", {})
    hp0 = agents.get("agent_0", {}).get("hp", 0)
    hp1 = agents.get("agent_1", {}).get("hp", 0)
    if any(terminations.values()):
        if hp0 <= 0 and hp1 <= 0:
            return "draw"
        if hp0 <= 0:
            return "agent_1"
        if hp1 <= 0:
            return "agent_0"

    if final_rewards["agent_0"] > final_rewards["agent_1"]:
        return "agent_0"
    if final_rewards["agent_1"] > final_rewards["agent_0"]:
        return "agent_1"
    return "draw"


def run_episode(
    cfg: Config,
    agent0_policy: EvalPolicy,
    agent1_policy: EvalPolicy,
    seed: int | None = None,
) -> dict[str, Any]:
    env = ArenaFightersEnv(config=cfg)
    obs_dict, _ = env.reset(seed=seed)
    action_counts = {
        "agent_0": {i: 0 for i in range(NUM_ACTIONS)},
        "agent_1": {i: 0 for i in range(NUM_ACTIONS)},
    }
    damage_dealt = {"agent_0": 0, "agent_1": 0}
    damage_events = {"agent_0": 0, "agent_1": 0}
    cumulative_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    final_rewards = {"agent_0": 0.0, "agent_1": 0.0}
    final_terminations = {"agent_0": False, "agent_1": False}
    final_truncations = {"agent_0": False, "agent_1": False}

    while env.agents:
        policies = {
            "agent_0": agent0_policy,
            "agent_1": agent1_policy,
        }
        actions = {}
        for agent_name in env.agents:
            action = policies[agent_name].act(agent_name, obs_dict[agent_name], env)
            actions[agent_name] = action
            action_counts[agent_name][action] += 1

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        final_rewards = {name: float(rewards[name]) for name in env.possible_agents}
        for agent_name in env.possible_agents:
            cumulative_rewards[agent_name] += float(rewards[agent_name])
        final_terminations = dict(terminations)
        final_truncations = dict(truncations)

        for agent_name in env.possible_agents:
            events = infos[agent_name]["events"]
            damage_dealt[agent_name] += events["damage_dealt"]
            damage_events[agent_name] += (
                events["projectile_hits"] + events["melee_hits"]
            )

        if any(terminations.values()) or any(truncations.values()):
            break

    total_actions = {
        agent_name: sum(counts.values())
        for agent_name, counts in action_counts.items()
    }
    idle_rate = {
        agent_name: (
            action_counts[agent_name][IDLE] / total if total else 0.0
        )
        for agent_name, total in total_actions.items()
    }
    dominant_action_rate = {
        agent_name: (
            max(action_counts[agent_name].values()) / total if total else 0.0
        )
        for agent_name, total in total_actions.items()
    }
    total_damage = sum(damage_dealt.values())
    no_damage = total_damage == 0
    state = env.get_state()
    episode_length = state["tick"]
    winner = infer_winner(state, final_terminations, final_truncations, final_rewards)

    return {
        "winner": winner,
        "length": episode_length,
        "map_name": state["map_name"],
        "rewards": cumulative_rewards,
        "final_rewards": final_rewards,
        "action_counts": action_counts,
        "damage_dealt": damage_dealt,
        "event_totals": state["episode_events"],
        "behavior": {
            "idle_rate": idle_rate,
            "dominant_action_rate": dominant_action_rate,
            "damage_events": damage_events,
            "no_damage": no_damage,
            "low_engagement": no_damage and episode_length >= cfg.arena.max_ticks,
        },
    }


def action_distribution_from_counts(
    action_counts: dict[str, dict[int, int]],
) -> dict[str, dict[int, float]]:
    distribution = {}
    for agent_name, counts in action_counts.items():
        total = sum(counts.values())
        distribution[agent_name] = {
            action: (count / total if total else 0.0)
            for action, count in counts.items()
        }
    return distribution


def evaluate_matchup(
    cfg: Config,
    agent0_policy: EvalPolicy,
    agent1_policy: EvalPolicy,
    episodes: int,
    seed: int | None = None,
) -> dict[str, Any]:
    wins = {"agent_0": 0, "agent_1": 0, "draw": 0}
    lengths: list[int] = []
    action_counts = {
        "agent_0": {i: 0 for i in range(NUM_ACTIONS)},
        "agent_1": {i: 0 for i in range(NUM_ACTIONS)},
    }
    damage_dealt = {"agent_0": 0, "agent_1": 0}
    damage_events = {"agent_0": 0, "agent_1": 0}
    reward_sums = {"agent_0": 0.0, "agent_1": 0.0}
    idle_rate_sum = {"agent_0": 0.0, "agent_1": 0.0}
    dominant_action_rate_sum = {"agent_0": 0.0, "agent_1": 0.0}
    event_totals = {"agent_0": {}, "agent_1": {}}
    no_damage_episodes = 0
    low_engagement_episodes = 0
    per_map: dict[str, dict[str, Any]] = {}

    for episode_idx in range(episodes):
        episode_seed = None if seed is None else seed + episode_idx
        result = run_episode(cfg, agent0_policy, agent1_policy, seed=episode_seed)
        wins[result["winner"]] += 1
        lengths.append(result["length"])
        map_name = result["map_name"]
        if map_name not in per_map:
            per_map[map_name] = {
                "episodes": 0,
                "wins": {"agent_0": 0, "agent_1": 0, "draw": 0},
                "lengths": [],
                "action_counts": {
                    "agent_0": {i: 0 for i in range(NUM_ACTIONS)},
                    "agent_1": {i: 0 for i in range(NUM_ACTIONS)},
                },
                "damage_dealt": {"agent_0": 0, "agent_1": 0},
                "damage_events": {"agent_0": 0, "agent_1": 0},
                "event_totals": {"agent_0": {}, "agent_1": {}},
                "reward_sums": {"agent_0": 0.0, "agent_1": 0.0},
                "idle_rate_sum": {"agent_0": 0.0, "agent_1": 0.0},
                "dominant_action_rate_sum": {"agent_0": 0.0, "agent_1": 0.0},
                "no_damage_episodes": 0,
                "low_engagement_episodes": 0,
            }
        per_map[map_name]["episodes"] += 1
        per_map[map_name]["wins"][result["winner"]] += 1
        per_map[map_name]["lengths"].append(result["length"])
        if result["behavior"]["no_damage"]:
            no_damage_episodes += 1
            per_map[map_name]["no_damage_episodes"] += 1
        if result["behavior"]["low_engagement"]:
            low_engagement_episodes += 1
            per_map[map_name]["low_engagement_episodes"] += 1
        for agent_name in action_counts:
            for action, count in result["action_counts"][agent_name].items():
                action_counts[agent_name][action] += count
                per_map[map_name]["action_counts"][agent_name][action] += count
            reward_sums[agent_name] += float(result["rewards"][agent_name])
            per_map[map_name]["reward_sums"][agent_name] += float(
                result["rewards"][agent_name]
            )
            damage_dealt[agent_name] += result["damage_dealt"][agent_name]
            per_map[map_name]["damage_dealt"][agent_name] += result["damage_dealt"][
                agent_name
            ]
            for event_name, count in result["event_totals"][agent_name].items():
                event_totals[agent_name][event_name] = (
                    event_totals[agent_name].get(event_name, 0) + count
                )
                per_map[map_name]["event_totals"][agent_name][event_name] = (
                    per_map[map_name]["event_totals"][agent_name].get(event_name, 0)
                    + count
                )
            damage_events[agent_name] += result["behavior"]["damage_events"][agent_name]
            per_map[map_name]["damage_events"][agent_name] += result["behavior"][
                "damage_events"
            ][agent_name]
            idle_rate_sum[agent_name] += result["behavior"]["idle_rate"][agent_name]
            per_map[map_name]["idle_rate_sum"][agent_name] += result["behavior"][
                "idle_rate"
            ][agent_name]
            dominant_action_rate_sum[agent_name] += result["behavior"][
                "dominant_action_rate"
            ][agent_name]
            per_map[map_name]["dominant_action_rate_sum"][agent_name] += result[
                "behavior"
            ]["dominant_action_rate"][agent_name]

    per_map_summary = {}
    for map_name, map_metrics in per_map.items():
        map_episodes = map_metrics["episodes"]
        map_wins = map_metrics["wins"]
        per_map_summary[map_name] = {
            "episodes": map_episodes,
            "wins": map_wins,
            "win_rate_agent_0": (
                map_wins["agent_0"] / map_episodes if map_episodes else 0.0
            ),
            "win_rate_agent_1": (
                map_wins["agent_1"] / map_episodes if map_episodes else 0.0
            ),
            "draw_rate": map_wins["draw"] / map_episodes if map_episodes else 0.0,
            "avg_length": (
                float(np.mean(map_metrics["lengths"]))
                if map_metrics["lengths"] else 0.0
            ),
            "avg_rewards": {
                agent_name: (
                    map_metrics["reward_sums"][agent_name] / map_episodes
                    if map_episodes else 0.0
                )
                for agent_name in map_metrics["reward_sums"]
            },
            "action_counts": map_metrics["action_counts"],
            "action_distribution": action_distribution_from_counts(
                map_metrics["action_counts"]
            ),
            "damage_dealt": map_metrics["damage_dealt"],
            "event_totals": map_metrics["event_totals"],
            "behavior": {
                "avg_idle_rate": {
                    agent_name: (
                        map_metrics["idle_rate_sum"][agent_name] / map_episodes
                        if map_episodes else 0.0
                    )
                    for agent_name in map_metrics["idle_rate_sum"]
                },
                "avg_dominant_action_rate": {
                    agent_name: (
                        map_metrics["dominant_action_rate_sum"][agent_name]
                        / map_episodes
                        if map_episodes else 0.0
                    )
                    for agent_name in map_metrics["dominant_action_rate_sum"]
                },
                "damage_events": map_metrics["damage_events"],
                "no_damage_episodes": map_metrics["no_damage_episodes"],
                "low_engagement_episodes": map_metrics["low_engagement_episodes"],
            },
            "no_damage_episodes": map_metrics["no_damage_episodes"],
            "low_engagement_episodes": map_metrics["low_engagement_episodes"],
        }

    return {
        "episodes": episodes,
        "wins": wins,
        "win_rate_agent_0": wins["agent_0"] / episodes if episodes else 0.0,
        "win_rate_agent_1": wins["agent_1"] / episodes if episodes else 0.0,
        "draw_rate": wins["draw"] / episodes if episodes else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_rewards": {
            agent_name: reward_sums[agent_name] / episodes if episodes else 0.0
            for agent_name in reward_sums
        },
        "action_counts": action_counts,
        "action_distribution": action_distribution_from_counts(action_counts),
        "damage_dealt": damage_dealt,
        "event_totals": event_totals,
        "behavior": {
            "avg_idle_rate": {
                agent_name: idle_rate_sum[agent_name] / episodes if episodes else 0.0
                for agent_name in idle_rate_sum
            },
            "avg_dominant_action_rate": {
                agent_name: (
                    dominant_action_rate_sum[agent_name] / episodes
                    if episodes else 0.0
                )
                for agent_name in dominant_action_rate_sum
            },
            "damage_events": damage_events,
            "no_damage_episodes": no_damage_episodes,
            "low_engagement_episodes": low_engagement_episodes,
        },
        "per_map": per_map_summary,
    }


def evaluate_baseline_suite(
    cfg: Config,
    agent0_policy_factory: Callable[[int | None], EvalPolicy],
    agent0_label: str,
    opponents: tuple[str, ...],
    maps: tuple[str, ...],
    episodes: int,
    seed: int | None = None,
    reward_preset: str = "default",
) -> dict[str, Any]:
    matchups: dict[str, dict[str, Any]] = {}
    win_rates = []

    for map_idx, map_name in enumerate(maps):
        map_cfg = replace(
            cfg,
            arena=replace(cfg.arena, map_name=map_name, randomize_maps=False),
        )
        matchups[map_name] = {}
        for opponent_idx, opponent_name in enumerate(opponents):
            matchup_seed = (
                None if seed is None else seed + map_idx * 10_000 + opponent_idx
            )
            opponent_seed = None if matchup_seed is None else matchup_seed + 100_000
            summary = evaluate_matchup(
                cfg=map_cfg,
                agent0_policy=agent0_policy_factory(matchup_seed),
                agent1_policy=make_builtin_policy(opponent_name, seed=opponent_seed),
                episodes=episodes,
                seed=matchup_seed,
            )
            summary["agent_0_policy"] = agent0_label
            summary["agent_1_policy"] = opponent_name
            summary["eval_config"] = {
                "agent_0_policy": agent0_label,
                "opponent": opponent_name,
                "episodes": episodes,
                "seed": matchup_seed,
                "map_name": map_name,
                "randomize_maps": False,
                "map_choices": [map_name],
                "reward_preset": reward_preset,
            }
            matchups[map_name][opponent_name] = summary
            win_rates.append(summary["win_rate_agent_0"])

    total_matchups = len(maps) * len(opponents)
    return {
        "suite_config": {
            "agent_0_policy": agent0_label,
            "opponents": list(opponents),
            "maps": list(maps),
            "episodes_per_matchup": episodes,
            "seed": seed,
            "reward_preset": reward_preset,
        },
        "overview": {
            "total_matchups": total_matchups,
            "total_episodes": total_matchups * episodes,
            "mean_agent_0_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        },
        "matchups": matchups,
    }


def update_elo_ratings(
    ratings: dict[str, float],
    label_a: str,
    label_b: str,
    score_a: float,
    k_factor: float = 32.0,
) -> None:
    rating_a = ratings[label_a]
    rating_b = ratings[label_b]
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    ratings[label_a] = rating_a + k_factor * (score_a - expected_a)
    ratings[label_b] = rating_b + k_factor * ((1.0 - score_a) - (1.0 - expected_a))


def evaluate_pairwise_suite(
    cfg: Config,
    policy_factories: dict[str, Callable[[int | None], EvalPolicy]],
    maps: tuple[str, ...],
    episodes: int,
    seed: int | None = None,
    initial_elo: float = 1000.0,
    elo_k_factor: float = 32.0,
) -> dict[str, Any]:
    labels = tuple(policy_factories)
    standings = {
        label: {"wins": 0, "losses": 0, "draws": 0, "episodes": 0}
        for label in labels
    }
    elo_ratings = {label: float(initial_elo) for label in labels}
    matchups: dict[str, dict[str, Any]] = {}

    def record(summary: dict[str, Any], agent0_label: str, agent1_label: str) -> None:
        wins = summary["wins"]
        standings[agent0_label]["wins"] += wins["agent_0"]
        standings[agent0_label]["losses"] += wins["agent_1"]
        standings[agent0_label]["draws"] += wins["draw"]
        standings[agent0_label]["episodes"] += summary["episodes"]
        standings[agent1_label]["wins"] += wins["agent_1"]
        standings[agent1_label]["losses"] += wins["agent_0"]
        standings[agent1_label]["draws"] += wins["draw"]
        standings[agent1_label]["episodes"] += summary["episodes"]
        for _ in range(wins["agent_0"]):
            update_elo_ratings(
                elo_ratings,
                agent0_label,
                agent1_label,
                score_a=1.0,
                k_factor=elo_k_factor,
            )
        for _ in range(wins["draw"]):
            update_elo_ratings(
                elo_ratings,
                agent0_label,
                agent1_label,
                score_a=0.5,
                k_factor=elo_k_factor,
            )
        for _ in range(wins["agent_1"]):
            update_elo_ratings(
                elo_ratings,
                agent0_label,
                agent1_label,
                score_a=0.0,
                k_factor=elo_k_factor,
            )

    pair_idx = 0
    for map_idx, map_name in enumerate(maps):
        map_cfg = replace(
            cfg,
            arena=replace(cfg.arena, map_name=map_name, randomize_maps=False),
        )
        matchups[map_name] = {}
        for idx_a, label_a in enumerate(labels):
            for idx_b in range(idx_a + 1, len(labels)):
                label_b = labels[idx_b]
                base_seed = (
                    None if seed is None else seed + map_idx * 1_000_000 + pair_idx * 10_000
                )
                forward = evaluate_matchup(
                    cfg=map_cfg,
                    agent0_policy=policy_factories[label_a](base_seed),
                    agent1_policy=policy_factories[label_b](
                        None if base_seed is None else base_seed + 1
                    ),
                    episodes=episodes,
                    seed=base_seed,
                )
                reverse_seed = None if base_seed is None else base_seed + 5_000
                reverse = evaluate_matchup(
                    cfg=map_cfg,
                    agent0_policy=policy_factories[label_b](reverse_seed),
                    agent1_policy=policy_factories[label_a](
                        None if reverse_seed is None else reverse_seed + 1
                    ),
                    episodes=episodes,
                    seed=reverse_seed,
                )
                record(forward, label_a, label_b)
                record(reverse, label_b, label_a)
                matchup_key = f"{label_a}__vs__{label_b}"
                matchups[map_name][matchup_key] = {
                    "agent_0_policy": label_a,
                    "agent_1_policy": label_b,
                    "forward": forward,
                    "reverse": reverse,
                }
                pair_idx += 1

    standings_rows = []
    for label, row in standings.items():
        episodes_played = row["episodes"]
        score = (
            (row["wins"] + 0.5 * row["draws"]) / episodes_played
            if episodes_played else 0.0
        )
        standings_rows.append(
            {
                "label": label,
                **row,
                "score": score,
                "elo": elo_ratings[label],
                "win_rate": row["wins"] / episodes_played if episodes_played else 0.0,
                "draw_rate": row["draws"] / episodes_played if episodes_played else 0.0,
                "loss_rate": row["losses"] / episodes_played if episodes_played else 0.0,
            }
        )
    standings_rows.sort(
        key=lambda item: (-item["elo"], -item["score"], -item["win_rate"], item["label"])
    )
    for rank, row in enumerate(standings_rows, start=1):
        row["rank"] = rank

    total_pairs = len(labels) * (len(labels) - 1) // 2
    return {
        "head_to_head_config": {
            "policies": list(labels),
            "maps": list(maps),
            "episodes_per_side": episodes,
            "seed": seed,
            "initial_elo": initial_elo,
            "elo_k_factor": elo_k_factor,
        },
        "overview": {
            "policies": len(labels),
            "maps": len(maps),
            "unordered_pairs": total_pairs,
            "ordered_matchups": total_pairs * len(maps) * 2,
            "total_episodes": total_pairs * len(maps) * episodes * 2,
        },
        "standings": standings_rows,
        "matchups": matchups,
    }


def score_baseline_suite(
    suite: dict[str, Any],
    draw_weight: float = 0.5,
    no_damage_penalty: float = 0.25,
    low_engagement_penalty: float = 0.25,
) -> dict[str, Any]:
    matchup_scores = []

    for map_name, opponents in suite.get("matchups", {}).items():
        for opponent_name, summary in opponents.items():
            episodes = int(summary.get("episodes", 0))
            win_rate = float(summary.get("win_rate_agent_0", 0.0))
            draw_rate = float(summary.get("draw_rate", 0.0))
            avg_length = float(summary.get("avg_length", 0.0))
            behavior = summary.get("behavior", {})
            no_damage_rate = (
                float(behavior.get("no_damage_episodes", 0)) / episodes
                if episodes else 0.0
            )
            low_engagement_rate = (
                float(behavior.get("low_engagement_episodes", 0)) / episodes
                if episodes else 0.0
            )
            score = (
                win_rate
                + draw_weight * draw_rate
                - no_damage_penalty * no_damage_rate
                - low_engagement_penalty * low_engagement_rate
            )
            matchup_scores.append(
                {
                    "map_name": map_name,
                    "opponent": opponent_name,
                    "score": score,
                    "episodes": episodes,
                    "win_rate_agent_0": win_rate,
                    "draw_rate": draw_rate,
                    "no_damage_rate": no_damage_rate,
                    "low_engagement_rate": low_engagement_rate,
                    "avg_length": avg_length,
                }
            )

    return {
        "score": (
            float(np.mean([item["score"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "mean_win_rate_agent_0": (
            float(np.mean([item["win_rate_agent_0"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "mean_draw_rate": (
            float(np.mean([item["draw_rate"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "mean_no_damage_rate": (
            float(np.mean([item["no_damage_rate"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "mean_low_engagement_rate": (
            float(np.mean([item["low_engagement_rate"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "mean_avg_length": (
            float(np.mean([item["avg_length"] for item in matchup_scores]))
            if matchup_scores else 0.0
        ),
        "matchup_scores": matchup_scores,
    }


def rank_baseline_suites(
    entries: list[dict[str, Any]],
    draw_weight: float = 0.5,
    no_damage_penalty: float = 0.25,
    low_engagement_penalty: float = 0.25,
) -> dict[str, Any]:
    rankings = []
    for entry in entries:
        suite_score = score_baseline_suite(
            entry["suite"],
            draw_weight=draw_weight,
            no_damage_penalty=no_damage_penalty,
            low_engagement_penalty=low_engagement_penalty,
        )
        rankings.append(
            {
                "label": entry["label"],
                "checkpoint": entry.get("checkpoint"),
                "checkpoint_metadata": entry.get("checkpoint_metadata"),
                **suite_score,
            }
        )

    rankings.sort(
        key=lambda item: (
            -item["score"],
            -item["mean_win_rate_agent_0"],
            item["mean_draw_rate"],
            item["label"],
        )
    )
    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx

    return {
        "ranking_metric": (
            "mean(win_rate_agent_0 + draw_weight * draw_rate "
            "- no_damage_penalty * no_damage_rate "
            "- low_engagement_penalty * low_engagement_rate)"
        ),
        "score_config": {
            "draw_weight": draw_weight,
            "no_damage_penalty": no_damage_penalty,
            "low_engagement_penalty": low_engagement_penalty,
        },
        "rankings": rankings,
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return repr(value)


def _episode_count(value: Any) -> int:
    try:
        count = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, count)


def ranking_per_map_score_details(
    candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_map: dict[str, dict[str, Any]] = {}
    invalid_scores = []
    for matchup_index, item in enumerate(candidate.get("matchup_scores", [])):
        if not isinstance(item, dict):
            continue
        map_name = item.get("map_name")
        if not map_name:
            continue
        if "score" not in item:
            invalid_scores.append(
                {
                    "map_name": str(map_name),
                    "matchup_index": matchup_index,
                    "score": None,
                    "reason": "missing_score",
                }
            )
            continue
        score = _finite_float(item["score"])
        if score is None:
            invalid_scores.append(
                {
                    "map_name": str(map_name),
                    "matchup_index": matchup_index,
                    "score": _json_safe_value(item.get("score")),
                    "reason": "invalid_score",
                }
            )
            continue
        entry = per_map.setdefault(
            str(map_name),
            {
                "map_name": str(map_name),
                "score_sum": 0.0,
                "matchup_count": 0,
                "episode_count": 0,
            },
        )
        entry["score_sum"] += score
        entry["matchup_count"] += 1
        entry["episode_count"] += _episode_count(item.get("episodes", 0))
    per_map_scores = [
        {
            "map_name": item["map_name"],
            "mean_score": item["score_sum"] / item["matchup_count"],
            "matchup_count": item["matchup_count"],
            "episode_count": item["episode_count"],
        }
        for item in sorted(per_map.values(), key=lambda row: row["map_name"])
        if item["matchup_count"] > 0
    ]
    return per_map_scores, invalid_scores


def ranking_per_map_scores(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    per_map_scores, _ = ranking_per_map_score_details(candidate)
    return per_map_scores


def gate_rank_summary(
    summary: dict[str, Any],
    min_score: float = 0.1,
    min_win_rate: float = 0.0,
    max_draw_rate: float = 0.9,
    max_no_damage_rate: float = 0.75,
    max_low_engagement_rate: float = 0.5,
    min_map_score: float | None = None,
    min_head_to_head_elo: float | None = None,
    min_head_to_head_score: float | None = None,
) -> dict[str, Any]:
    rankings = summary.get("rankings", [])
    rules = {
        "min_score": min_score,
        "min_win_rate": min_win_rate,
        "max_draw_rate": max_draw_rate,
        "max_no_damage_rate": max_no_damage_rate,
        "max_low_engagement_rate": max_low_engagement_rate,
        "min_map_score": min_map_score,
        "min_head_to_head_elo": min_head_to_head_elo,
        "min_head_to_head_score": min_head_to_head_score,
    }
    if not rankings:
        return {
            "passed": False,
            "rules": rules,
            "candidate": None,
            "failures": [{"metric": "rankings", "reason": "missing_rankings"}],
        }

    candidate = rankings[0]
    checks = (
        ("score", float(candidate.get("score", 0.0)), min_score, "min"),
        (
            "mean_win_rate_agent_0",
            float(candidate.get("mean_win_rate_agent_0", 0.0)),
            min_win_rate,
            "min",
        ),
        (
            "mean_draw_rate",
            float(candidate.get("mean_draw_rate", 0.0)),
            max_draw_rate,
            "max",
        ),
        (
            "mean_no_damage_rate",
            float(candidate.get("mean_no_damage_rate", 0.0)),
            max_no_damage_rate,
            "max",
        ),
        (
            "mean_low_engagement_rate",
            float(candidate.get("mean_low_engagement_rate", 0.0)),
            max_low_engagement_rate,
            "max",
        ),
    )
    failures = []
    for metric, value, threshold, direction in checks:
        if direction == "min" and value < threshold:
            failures.append(
                {
                    "metric": metric,
                    "value": value,
                    "min": threshold,
                    "reason": "below_minimum",
                }
            )
        if direction == "max" and value > threshold:
            failures.append(
                {
                    "metric": metric,
                    "value": value,
                    "max": threshold,
                    "reason": "above_maximum",
                }
            )

    if min_map_score is not None:
        per_map_scores, invalid_map_scores = ranking_per_map_score_details(candidate)
        low_score_maps = [
            item for item in per_map_scores if item["mean_score"] < min_map_score
        ]
        if invalid_map_scores:
            failures.append(
                {
                    "metric": "per_map_score",
                    "value": None,
                    "min": min_map_score,
                    "reason": "invalid_score",
                    "invalid_map_scores": invalid_map_scores,
                    "low_score_maps": low_score_maps,
                    "per_map_scores": per_map_scores,
                }
            )
        elif not per_map_scores or low_score_maps:
            failures.append(
                {
                    "metric": "per_map_score",
                    "value": (
                        min(item["mean_score"] for item in low_score_maps)
                        if low_score_maps
                        else None
                    ),
                    "min": min_map_score,
                    "reason": (
                        "below_minimum" if low_score_maps else "missing_map_scores"
                    ),
                    "low_score_maps": low_score_maps,
                    "per_map_scores": per_map_scores,
                }
            )

    if min_head_to_head_elo is not None or min_head_to_head_score is not None:
        standings = summary.get("head_to_head", {}).get("standings", [])
        standing = next(
            (row for row in standings if row.get("label") == candidate.get("label")),
            None,
        )
        if standing is None:
            failures.append(
                {
                    "metric": "head_to_head",
                    "reason": "missing_candidate_standing",
                }
            )
        else:
            if (
                min_head_to_head_elo is not None
                and float(standing.get("elo", 0.0)) < min_head_to_head_elo
            ):
                failures.append(
                    {
                        "metric": "head_to_head.elo",
                        "value": float(standing.get("elo", 0.0)),
                        "min": min_head_to_head_elo,
                        "reason": "below_minimum",
                    }
                )
            if (
                min_head_to_head_score is not None
                and float(standing.get("score", 0.0)) < min_head_to_head_score
            ):
                failures.append(
                    {
                        "metric": "head_to_head.score",
                        "value": float(standing.get("score", 0.0)),
                        "min": min_head_to_head_score,
                        "reason": "below_minimum",
                    }
                )

    return {
        "artifact": artifact_metadata("rank_gate"),
        "passed": not failures,
        "rules": rules,
        "candidate": candidate,
        "failures": failures,
    }


def write_eval_summary(
    summary: dict[str, Any],
    output_dir: str | Path,
    label: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now(timezone.utc)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")
    label_text = _slugify(label or "eval")
    path = output_path / f"{timestamp_text}_{label_text}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def load_eval_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def compare_eval_summaries(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    scalar_paths = (
        ("win_rate_agent_0",),
        ("win_rate_agent_1",),
        ("draw_rate",),
        ("avg_length",),
        ("avg_rewards", "agent_0"),
        ("avg_rewards", "agent_1"),
        ("damage_dealt", "agent_0"),
        ("damage_dealt", "agent_1"),
        ("behavior", "avg_idle_rate", "agent_0"),
        ("behavior", "avg_idle_rate", "agent_1"),
        ("behavior", "avg_dominant_action_rate", "agent_0"),
        ("behavior", "avg_dominant_action_rate", "agent_1"),
        ("behavior", "damage_events", "agent_0"),
        ("behavior", "damage_events", "agent_1"),
        ("behavior", "no_damage_episodes"),
        ("behavior", "low_engagement_episodes"),
    )
    deltas = {
        ".".join(path): _numeric_at(after, path) - _numeric_at(before, path)
        for path in scalar_paths
    }
    deltas.update(_action_distribution_deltas(before, after))

    before_maps = before.get("per_map", {})
    after_maps = after.get("per_map", {})
    per_map = {}
    for map_name in sorted(set(before_maps) | set(after_maps)):
        before_map = before_maps.get(map_name, {})
        after_map = after_maps.get(map_name, {})
        map_deltas = {
            "win_rate_agent_0": _numeric_at(
                after_map, ("win_rate_agent_0",)
            ) - _numeric_at(before_map, ("win_rate_agent_0",)),
            "win_rate_agent_1": _numeric_at(
                after_map, ("win_rate_agent_1",)
            ) - _numeric_at(before_map, ("win_rate_agent_1",)),
            "draw_rate": _numeric_at(after_map, ("draw_rate",)) - _numeric_at(
                before_map, ("draw_rate",)
            ),
            "avg_length": _numeric_at(after_map, ("avg_length",)) - _numeric_at(
                before_map, ("avg_length",)
            ),
            "avg_rewards.agent_0": _numeric_at(
                after_map, ("avg_rewards", "agent_0")
            ) - _numeric_at(before_map, ("avg_rewards", "agent_0")),
            "avg_rewards.agent_1": _numeric_at(
                after_map, ("avg_rewards", "agent_1")
            ) - _numeric_at(before_map, ("avg_rewards", "agent_1")),
            "damage_dealt.agent_0": _numeric_at(
                after_map, ("damage_dealt", "agent_0")
            ) - _numeric_at(before_map, ("damage_dealt", "agent_0")),
            "damage_dealt.agent_1": _numeric_at(
                after_map, ("damage_dealt", "agent_1")
            ) - _numeric_at(before_map, ("damage_dealt", "agent_1")),
            "behavior.avg_idle_rate.agent_0": _numeric_at(
                after_map, ("behavior", "avg_idle_rate", "agent_0")
            ) - _numeric_at(
                before_map, ("behavior", "avg_idle_rate", "agent_0")
            ),
            "behavior.avg_idle_rate.agent_1": _numeric_at(
                after_map, ("behavior", "avg_idle_rate", "agent_1")
            ) - _numeric_at(
                before_map, ("behavior", "avg_idle_rate", "agent_1")
            ),
            "behavior.avg_dominant_action_rate.agent_0": _numeric_at(
                after_map,
                ("behavior", "avg_dominant_action_rate", "agent_0"),
            ) - _numeric_at(
                before_map,
                ("behavior", "avg_dominant_action_rate", "agent_0"),
            ),
            "behavior.avg_dominant_action_rate.agent_1": _numeric_at(
                after_map,
                ("behavior", "avg_dominant_action_rate", "agent_1"),
            ) - _numeric_at(
                before_map,
                ("behavior", "avg_dominant_action_rate", "agent_1"),
            ),
            "behavior.damage_events.agent_0": _numeric_at(
                after_map, ("behavior", "damage_events", "agent_0")
            ) - _numeric_at(before_map, ("behavior", "damage_events", "agent_0")),
            "behavior.damage_events.agent_1": _numeric_at(
                after_map, ("behavior", "damage_events", "agent_1")
            ) - _numeric_at(before_map, ("behavior", "damage_events", "agent_1")),
            "no_damage_episodes": _numeric_at(
                after_map, ("no_damage_episodes",)
            ) - _numeric_at(before_map, ("no_damage_episodes",)),
            "low_engagement_episodes": _numeric_at(
                after_map, ("low_engagement_episodes",)
            ) - _numeric_at(before_map, ("low_engagement_episodes",)),
            "behavior.no_damage_episodes": _numeric_at(
                after_map, ("behavior", "no_damage_episodes")
            ) - _numeric_at(before_map, ("behavior", "no_damage_episodes")),
            "behavior.low_engagement_episodes": _numeric_at(
                after_map, ("behavior", "low_engagement_episodes")
            ) - _numeric_at(before_map, ("behavior", "low_engagement_episodes")),
        }
        map_deltas.update(_action_distribution_deltas(before_map, after_map))
        per_map[map_name] = {
            "episodes": {
                "before": int(before_map.get("episodes", 0)),
                "after": int(after_map.get("episodes", 0)),
            },
            "deltas": map_deltas,
        }

    return {
        "artifact": artifact_metadata("comparison"),
        "before_config": before.get("eval_config", {}),
        "after_config": after.get("eval_config", {}),
        "deltas": deltas,
        "per_map": per_map,
    }


def gate_eval_comparison(
    comparison: dict[str, Any],
    rules: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    rules = rules or DEFAULT_GATE_RULES
    deltas = comparison.get("deltas", {})
    failures = []

    def add_delta_failures(
        metric: str,
        delta: float,
        rule: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        min_delta = rule.get("min_delta")
        max_delta = rule.get("max_delta")
        if min_delta is not None and delta < min_delta:
            failure = {
                "metric": metric,
                "delta": delta,
                "min_delta": min_delta,
                "reason": "below_min_delta",
            }
            if extra:
                failure.update(extra)
            failures.append(failure)
        if max_delta is not None and delta > max_delta:
            failure = {
                "metric": metric,
                "delta": delta,
                "max_delta": max_delta,
                "reason": "above_max_delta",
            }
            if extra:
                failure.update(extra)
            failures.append(failure)

    for metric, rule in rules.items():
        if metric not in deltas:
            failures.append(
                {
                    "metric": metric,
                    "reason": "missing_metric",
                    "rule": rule,
                }
            )
            continue

        add_delta_failures(metric, float(deltas[metric]), rule)

    for map_name, map_summary in comparison.get("per_map", {}).items():
        map_deltas = map_summary.get("deltas", {})
        for metric, rule in rules.items():
            if metric not in map_deltas:
                continue
            add_delta_failures(
                metric,
                float(map_deltas[metric]),
                rule,
                extra={"scope": f"per_map:{map_name}", "map_name": map_name},
            )

    return {
        "artifact": artifact_metadata("gate"),
        "passed": not failures,
        "failures": failures,
        "rules": rules,
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "eval"


def _numeric_at(data: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return 0.0
        value = value[key]
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _action_distribution_deltas(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    before_distribution = before.get("action_distribution", {})
    after_distribution = after.get("action_distribution", {})
    deltas = {}
    for agent_name in sorted(set(before_distribution) | set(after_distribution)):
        before_actions = before_distribution.get(agent_name, {})
        after_actions = after_distribution.get(agent_name, {})
        action_keys = {
            str(action)
            for action in set(before_actions) | set(after_actions)
        }
        for action in sorted(action_keys, key=_action_sort_key):
            deltas[f"action_distribution.{agent_name}.{action}"] = (
                _numeric_key(after_actions, action)
                - _numeric_key(before_actions, action)
            )
    return deltas


def _action_sort_key(action: str) -> tuple[int, int | str]:
    try:
        return (0, int(action))
    except ValueError:
        return (1, action)


def _numeric_key(data: dict[Any, Any], key: str) -> float:
    value = data.get(key)
    if value is None:
        try:
            value = data.get(int(key))
        except ValueError:
            value = None
    if isinstance(value, int | float):
        return float(value)
    return 0.0

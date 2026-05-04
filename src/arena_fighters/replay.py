from __future__ import annotations

import json
from pathlib import Path

from arena_fighters.config import IDLE, NUM_ACTIONS


AGENT_NAMES = ("agent_0", "agent_1")


class ReplayLogger:
    """Logs episode game states to JSON for later replay."""

    def __init__(self, replay_dir: str, save_every_n: int = 100):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n = save_every_n
        self._episode_count = 0

    def save_episode(
        self,
        episode_id: int,
        frames: list[dict],
        winner: str | None,
        length: int,
    ):
        self._episode_count += 1
        if self._episode_count % self.save_every_n != 0:
            return

        data = {
            "episode_id": episode_id,
            "winner": winner,
            "length": length,
            **summarize_replay_frames(frames),
            "frames": frames,
        }
        path = self.replay_dir / f"episode_{episode_id:04d}.json"
        path.write_text(json.dumps(data))


def summarize_replay_frames(frames: list[dict]) -> dict:
    """Return top-level metadata derived from serialized env frames."""
    if not frames:
        action_counts = _empty_action_counts()
        return {
            "map_name": None,
            "event_totals": {},
            "action_counts": action_counts,
            "action_distribution": _action_distribution_from_counts(action_counts),
            "behavior": _action_behavior_from_counts(action_counts),
        }

    final_frame = frames[-1]
    event_totals = final_frame.get("episode_events")
    if event_totals is None:
        event_totals = _sum_step_events(frames)
    action_counts = _sum_frame_actions(frames)

    return {
        "map_name": final_frame.get("map_name", "classic"),
        "event_totals": event_totals,
        "action_counts": action_counts,
        "action_distribution": _action_distribution_from_counts(action_counts),
        "behavior": _action_behavior_from_counts(action_counts),
    }


def _sum_step_events(frames: list[dict]) -> dict:
    totals: dict[str, dict[str, int]] = {}
    for frame in frames:
        for agent_name, events in frame.get("events", {}).items():
            agent_totals = totals.setdefault(agent_name, {})
            for event_name, value in events.items():
                agent_totals[event_name] = agent_totals.get(event_name, 0) + value
    return totals


def _empty_action_counts() -> dict[str, dict[int, int]]:
    return {
        agent_name: {action: 0 for action in range(NUM_ACTIONS)}
        for agent_name in AGENT_NAMES
    }


def _normalize_action_counts(action_counts: object) -> dict[str, dict[int, int]]:
    normalized = _empty_action_counts()
    if not isinstance(action_counts, dict):
        return normalized

    for agent_name, counts in action_counts.items():
        if not isinstance(counts, dict):
            continue
        agent_counts = normalized.setdefault(
            str(agent_name),
            {action: 0 for action in range(NUM_ACTIONS)},
        )
        for action, count in counts.items():
            try:
                action_idx = int(action)
                action_count = int(count)
            except (TypeError, ValueError):
                continue
            if 0 <= action_idx < NUM_ACTIONS:
                agent_counts[action_idx] = action_count
    return normalized


def _sum_frame_actions(frames: list[dict]) -> dict[str, dict[int, int]]:
    action_counts = _empty_action_counts()
    for frame in frames:
        actions = frame.get("actions")
        if not isinstance(actions, dict):
            continue
        for agent_name, action in actions.items():
            try:
                action_idx = int(action)
            except (TypeError, ValueError):
                continue
            if 0 <= action_idx < NUM_ACTIONS:
                action_counts.setdefault(
                    str(agent_name),
                    {idx: 0 for idx in range(NUM_ACTIONS)},
                )
                action_counts[str(agent_name)][action_idx] += 1
    return action_counts


def _action_distribution_from_counts(
    action_counts: dict[str, dict[int, int]],
) -> dict[str, dict[int, float]]:
    distribution = {}
    for agent_name, counts in action_counts.items():
        total = sum(counts.values())
        distribution[agent_name] = {
            action: count / total if total else 0.0
            for action, count in counts.items()
        }
    return distribution


def _action_behavior_from_counts(action_counts: dict[str, dict[int, int]]) -> dict:
    return {
        "avg_idle_rate": {
            agent_name: (
                counts.get(IDLE, 0) / sum(counts.values())
                if sum(counts.values())
                else 0.0
            )
            for agent_name, counts in action_counts.items()
        },
        "avg_dominant_action_rate": {
            agent_name: (
                max(counts.values()) / sum(counts.values())
                if sum(counts.values())
                else 0.0
            )
            for agent_name, counts in action_counts.items()
        },
    }


def analyze_replay(data: dict) -> dict:
    frames = data.get("frames", [])
    frame_summary = summarize_replay_frames(frames)
    metadata = {
        "episode_id": data.get("episode_id"),
        "winner": data.get("winner"),
        "length": data.get("length", len(frames)),
        "map_name": data.get("map_name"),
    }
    if metadata["map_name"] is None and frames:
        metadata["map_name"] = frames[-1].get("map_name", "classic")

    event_totals = data.get("event_totals")
    if event_totals is None:
        event_totals = frame_summary["event_totals"]
    action_counts = _normalize_action_counts(
        data.get("action_counts") or frame_summary["action_counts"]
    )

    terminal_hp = {}
    if frames:
        final_agents = frames[-1].get("agents", {})
        terminal_hp = {
            agent_name: agent_state.get("hp")
            for agent_name, agent_state in final_agents.items()
        }

    totals = _sum_agent_event_totals(event_totals)
    action_behavior = _action_behavior_from_counts(action_counts)
    return {
        **metadata,
        "terminal_hp": terminal_hp,
        "event_totals": event_totals,
        "totals": totals,
        "action_counts": action_counts,
        "action_distribution": _action_distribution_from_counts(action_counts),
        "behavior": action_behavior,
        "flags": {
            "no_damage": totals["damage_dealt"] == 0,
            "no_projectile_hits": totals["projectile_hits"] == 0,
            "no_melee_hits": totals["melee_hits"] == 0,
            "no_shots_fired": totals["shots_fired"] == 0,
            "no_melee_attempts": totals["melee_attempts"] == 0,
            "no_recorded_actions": all(
                sum(counts.values()) == 0 for counts in action_counts.values()
            ),
            "no_attacks": (
                totals["shots_fired"] == 0 and totals["melee_attempts"] == 0
            ),
        },
    }


def _sum_agent_event_totals(event_totals: dict) -> dict[str, int]:
    keys = (
        "shots_fired",
        "melee_attempts",
        "melee_hits",
        "projectile_hits",
        "damage_dealt",
        "damage_taken",
    )
    totals = {key: 0 for key in keys}
    for events in event_totals.values():
        for key in keys:
            totals[key] += int(events.get(key, 0))
    return totals


def load_replay(path: Path) -> dict:
    """Load a replay file and return the parsed data."""
    return json.loads(path.read_text())

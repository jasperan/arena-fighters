from __future__ import annotations

import json
from pathlib import Path


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
        return {"map_name": None, "event_totals": {}}

    final_frame = frames[-1]
    event_totals = final_frame.get("episode_events")
    if event_totals is None:
        event_totals = _sum_step_events(frames)

    return {
        "map_name": final_frame.get("map_name", "classic"),
        "event_totals": event_totals,
    }


def _sum_step_events(frames: list[dict]) -> dict:
    totals: dict[str, dict[str, int]] = {}
    for frame in frames:
        for agent_name, events in frame.get("events", {}).items():
            agent_totals = totals.setdefault(agent_name, {})
            for event_name, value in events.items():
                agent_totals[event_name] = agent_totals.get(event_name, 0) + value
    return totals


def analyze_replay(data: dict) -> dict:
    frames = data.get("frames", [])
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
        event_totals = summarize_replay_frames(frames)["event_totals"]

    terminal_hp = {}
    if frames:
        final_agents = frames[-1].get("agents", {})
        terminal_hp = {
            agent_name: agent_state.get("hp")
            for agent_name, agent_state in final_agents.items()
        }

    totals = _sum_agent_event_totals(event_totals)
    return {
        **metadata,
        "terminal_hp": terminal_hp,
        "event_totals": event_totals,
        "totals": totals,
        "flags": {
            "no_damage": totals["damage_dealt"] == 0,
            "no_projectile_hits": totals["projectile_hits"] == 0,
            "no_melee_hits": totals["melee_hits"] == 0,
            "no_shots_fired": totals["shots_fired"] == 0,
            "no_melee_attempts": totals["melee_attempts"] == 0,
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

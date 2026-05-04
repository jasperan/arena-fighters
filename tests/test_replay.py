import json
import tempfile
from pathlib import Path

from arena_fighters.config import IDLE, MELEE, MOVE_RIGHT
from arena_fighters.replay import (
    ReplayLogger,
    analyze_replay,
    load_replay,
    summarize_replay_frames,
)


def test_logger_saves_and_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReplayLogger(replay_dir=tmpdir, save_every_n=1)
        frames = [
            {"tick": 0, "agents": {"a": {"hp": 100}}, "bullets": []},
            {"tick": 1, "agents": {"a": {"hp": 90}}, "bullets": []},
        ]
        logger.save_episode(episode_id=1, frames=frames, winner="a", length=2)

        replay = load_replay(Path(tmpdir) / "episode_0001.json")
        assert replay["episode_id"] == 1
        assert replay["winner"] == "a"
        assert len(replay["frames"]) == 2


def test_logger_respects_save_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReplayLogger(replay_dir=tmpdir, save_every_n=3)
        frames = [{"tick": 0, "agents": {}, "bullets": []}]

        logger.save_episode(episode_id=1, frames=frames, winner="a", length=1)
        logger.save_episode(episode_id=2, frames=frames, winner="b", length=1)
        logger.save_episode(episode_id=3, frames=frames, winner="a", length=1)

        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 1  # only episode 3 saved (every 3rd)


def test_load_replay_returns_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "episode_id": 42,
            "winner": "agent_0",
            "length": 5,
            "frames": [{"tick": i} for i in range(5)],
        }
        path = Path(tmpdir) / "episode_0042.json"
        path.write_text(json.dumps(data))

        replay = load_replay(path)
        assert replay["episode_id"] == 42
        assert len(replay["frames"]) == 5


def test_logger_saves_replay_event_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReplayLogger(replay_dir=tmpdir, save_every_n=1)
        frames = [
            {
                "tick": 0,
                "map_name": "flat",
                "agents": {},
                "bullets": [],
                "episode_events": {
                    "agent_0": {"shots_fired": 2, "damage_dealt": 10},
                    "agent_1": {"shots_fired": 0, "damage_taken": 10},
                },
            }
        ]

        logger.save_episode(episode_id=7, frames=frames, winner="agent_0", length=1)

        replay = load_replay(Path(tmpdir) / "episode_0007.json")
        assert replay["map_name"] == "flat"
        assert replay["event_totals"]["agent_0"]["shots_fired"] == 2
        assert replay["event_totals"]["agent_1"]["damage_taken"] == 10


def test_logger_saves_replay_action_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReplayLogger(replay_dir=tmpdir, save_every_n=1)
        frames = [
            {"tick": 0, "map_name": "flat", "agents": {}, "bullets": []},
            {
                "tick": 1,
                "map_name": "flat",
                "agents": {},
                "bullets": [],
                "actions": {"agent_0": IDLE, "agent_1": MOVE_RIGHT},
            },
            {
                "tick": 2,
                "map_name": "flat",
                "agents": {},
                "bullets": [],
                "actions": {"agent_0": MELEE, "agent_1": MOVE_RIGHT},
            },
        ]

        logger.save_episode(episode_id=8, frames=frames, winner="draw", length=2)

        replay = load_replay(Path(tmpdir) / "episode_0008.json")
        assert replay["action_counts"]["agent_0"][str(IDLE)] == 1
        assert replay["action_counts"]["agent_0"][str(MELEE)] == 1
        assert replay["action_counts"]["agent_1"][str(MOVE_RIGHT)] == 2
        assert replay["behavior"]["avg_idle_rate"]["agent_0"] == 0.5
        assert replay["behavior"]["avg_dominant_action_rate"]["agent_1"] == 1.0


def test_summarize_replay_frames_sums_step_events_without_episode_totals():
    frames = [
        {
            "tick": 0,
            "map_name": "classic",
            "events": {"agent_0": {"shots_fired": 1}},
        },
        {
            "tick": 1,
            "map_name": "classic",
            "events": {"agent_0": {"shots_fired": 2, "projectile_hits": 1}},
        },
    ]

    summary = summarize_replay_frames(frames)

    assert summary["map_name"] == "classic"
    assert summary["event_totals"]["agent_0"]["shots_fired"] == 3
    assert summary["event_totals"]["agent_0"]["projectile_hits"] == 1


def test_summarize_replay_frames_counts_actions():
    frames = [
        {"tick": 0, "map_name": "classic"},
        {"tick": 1, "map_name": "classic", "actions": {"agent_0": IDLE}},
        {
            "tick": 2,
            "map_name": "classic",
            "actions": {"agent_0": IDLE, "agent_1": MELEE},
        },
    ]

    summary = summarize_replay_frames(frames)

    assert summary["action_counts"]["agent_0"][IDLE] == 2
    assert summary["action_counts"]["agent_1"][MELEE] == 1
    assert summary["action_distribution"]["agent_0"][IDLE] == 1.0
    assert summary["behavior"]["avg_idle_rate"]["agent_0"] == 1.0
    assert summary["behavior"]["avg_dominant_action_rate"]["agent_1"] == 1.0


def test_analyze_replay_reports_terminal_state_and_flags():
    data = {
        "episode_id": 9,
        "winner": "agent_0",
        "length": 2,
        "map_name": "classic",
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
                "tick": 2,
                "agents": {
                    "agent_0": {"hp": 1},
                    "agent_1": {"hp": -9},
                },
            }
        ],
    }

    analysis = analyze_replay(data)

    assert analysis["winner"] == "agent_0"
    assert analysis["terminal_hp"] == {"agent_0": 1, "agent_1": -9}
    assert analysis["totals"]["damage_dealt"] == 10
    assert analysis["flags"]["no_damage"] is False
    assert analysis["flags"]["no_projectile_hits"] is False
    assert analysis["flags"]["no_melee_attempts"] is True
    assert analysis["flags"]["no_attacks"] is False


def test_analyze_replay_reports_action_behavior_from_frames():
    data = {
        "episode_id": 10,
        "winner": "draw",
        "length": 3,
        "map_name": "flat",
        "event_totals": {
            "agent_0": {},
            "agent_1": {},
        },
        "frames": [
            {"tick": 0, "map_name": "flat"},
            {"tick": 1, "map_name": "flat", "actions": {"agent_0": IDLE}},
            {"tick": 2, "map_name": "flat", "actions": {"agent_0": IDLE}},
            {"tick": 3, "map_name": "flat", "actions": {"agent_0": MELEE}},
        ],
    }

    analysis = analyze_replay(data)

    assert analysis["action_counts"]["agent_0"][IDLE] == 2
    assert analysis["action_counts"]["agent_0"][MELEE] == 1
    assert analysis["behavior"]["avg_idle_rate"]["agent_0"] == 2 / 3
    assert analysis["behavior"]["avg_dominant_action_rate"]["agent_0"] == 2 / 3
    assert analysis["flags"]["no_recorded_actions"] is False

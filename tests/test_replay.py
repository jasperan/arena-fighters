import json
import tempfile
from pathlib import Path

from arena_fighters.replay import ReplayLogger, load_replay


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

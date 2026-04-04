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
            "frames": frames,
        }
        path = self.replay_dir / f"episode_{episode_id:04d}.json"
        path.write_text(json.dumps(data))


def load_replay(path: Path) -> dict:
    """Load a replay file and return the parsed data."""
    return json.loads(path.read_text())

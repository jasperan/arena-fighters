"""Self-play wrapper and opponent pool for SB3 single-agent training."""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena_fighters.config import (
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_VECTOR_OBS,
    Config,
    RewardConfig,
)
from arena_fighters.env import ArenaFightersEnv
from arena_fighters.replay import ReplayLogger


class OpponentPool:
    """Stores frozen policy state_dict snapshots for self-play."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._snapshots: list[dict] = []
        self._snapshot_ids: list[int] = []
        self._snapshot_sample_counts: dict[int, int] = {}
        self._next_snapshot_id = 0
        self.last_sample_index: int | None = None
        self.last_sample_id: int | None = None
        self.last_sample_kind: str | None = None
        self.sample_counts = {"latest": 0, "historical": 0}

    def add(self, state_dict: dict) -> None:
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots.append(copy.deepcopy(state_dict))
        self._snapshot_ids.append(snapshot_id)
        self._snapshot_sample_counts[snapshot_id] = 0
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)
            evicted_id = self._snapshot_ids.pop(0)
            self._snapshot_sample_counts.pop(evicted_id, None)

    def sample(self, latest_prob: float = 0.8) -> dict:
        assert not self.is_empty(), "Cannot sample from empty pool"
        if len(self._snapshots) == 1 or random.random() < latest_prob:
            idx = len(self._snapshots) - 1
            kind = "latest"
        else:
            # Pick from all except the latest
            idx = random.randint(0, len(self._snapshots) - 2)
            kind = "historical"
        snapshot_id = self._snapshot_ids[idx]
        self.last_sample_index = idx
        self.last_sample_id = snapshot_id
        self.last_sample_kind = kind
        self.sample_counts[kind] += 1
        self._snapshot_sample_counts[snapshot_id] += 1
        return copy.deepcopy(self._snapshots[idx])

    def is_empty(self) -> bool:
        return len(self._snapshots) == 0

    def stats(self) -> dict:
        total_samples = self.sample_counts["latest"] + self.sample_counts["historical"]
        snapshots = [
            {
                "id": snapshot_id,
                "index": idx,
                "sample_count": self._snapshot_sample_counts[snapshot_id],
                "is_latest": idx == len(self._snapshot_ids) - 1,
            }
            for idx, snapshot_id in enumerate(self._snapshot_ids)
        ]
        return {
            "size": len(self),
            "latest_samples": self.sample_counts["latest"],
            "historical_samples": self.sample_counts["historical"],
            "historical_sample_rate": (
                self.sample_counts["historical"] / total_samples
                if total_samples
                else 0.0
            ),
            "snapshot_ids": list(self._snapshot_ids),
            "oldest_snapshot_id": self._snapshot_ids[0] if self._snapshot_ids else None,
            "latest_snapshot_id": self._snapshot_ids[-1] if self._snapshot_ids else None,
            "snapshots": snapshots,
            "last_sample_index": self.last_sample_index,
            "last_sample_id": self.last_sample_id,
            "last_sample_kind": self.last_sample_kind,
        }

    def __len__(self) -> int:
        return len(self._snapshots)


class SelfPlayWrapper(gym.Env):
    """Wraps the PettingZoo ParallelEnv as a single-agent Gymnasium env for SB3.

    Agent 0 is the training agent. Agent 1 is controlled by a frozen opponent
    policy (sampled from the pool) or acts randomly if the pool is empty.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: Config | None = None,
        opponent_pool: OpponentPool | None = None,
        opponent_policy: Any | None = None,
        replay_logger: ReplayLogger | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config or Config()
        self.opponent_pool = opponent_pool or OpponentPool()
        self.opponent_policy = opponent_policy
        self.replay_logger = replay_logger
        self.render_mode = render_mode

        self._env = ArenaFightersEnv(config=self.cfg, render_mode=render_mode)

        h, w = self.cfg.arena.height, self.cfg.arena.width
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_CHANNELS, h, w),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-1.0, high=1.0, shape=(NUM_VECTOR_OBS,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._opponent_obs: dict[str, np.ndarray] | None = None
        self._opponent_snapshot_loaded = False
        self._episode_id = 0
        self._episode_frames: list[dict[str, Any]] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        self._sample_opponent_from_pool()
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)
        self._episode_id += 1
        if self.replay_logger is not None:
            self._episode_frames = [copy.deepcopy(self._env.get_state())]

        agent0_obs = obs_dict["agent_0"]
        agent1_obs = obs_dict["agent_1"]
        self._opponent_obs = self._mirror_obs(agent1_obs)

        info = dict(info_dict.get("agent_0", {}))
        info["opponent_snapshot_loaded"] = self._opponent_snapshot_loaded
        info["opponent_pool"] = self.opponent_pool.stats()

        return agent0_obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        opp_action = self._get_opponent_action()

        actions = {"agent_0": action, "agent_1": opp_action}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(actions)

        agent0_obs = obs_dict["agent_0"]
        reward = float(rew_dict["agent_0"])
        terminated = bool(term_dict["agent_0"])
        truncated = bool(trunc_dict["agent_0"])
        info = info_dict.get("agent_0", {})

        state = self._env.get_state()
        state["actions"] = {
            agent_name: int(agent_action)
            for agent_name, agent_action in actions.items()
        }
        if self.replay_logger is not None:
            self._episode_frames.append(copy.deepcopy(state))
            if terminated or truncated:
                winner = self._infer_winner(state, term_dict, trunc_dict, rew_dict)
                self.replay_logger.save_episode(
                    episode_id=self._episode_id,
                    frames=self._episode_frames,
                    winner=winner,
                    length=int(state["tick"]),
                )

        # Store mirrored opponent obs for next step
        if not (terminated or truncated):
            agent1_obs = obs_dict["agent_1"]
            self._opponent_obs = self._mirror_obs(agent1_obs)
        else:
            self._opponent_obs = None

        return agent0_obs, reward, terminated, truncated, info

    def _get_opponent_action(self) -> int:
        if self.opponent_policy is not None and self._opponent_obs is not None:
            action, _ = self.opponent_policy.predict(self._opponent_obs, deterministic=False)
            return int(action)
        return self.action_space.sample()

    def _infer_winner(
        self,
        state: dict[str, Any],
        terminations: dict[str, bool],
        truncations: dict[str, bool],
        rewards: dict[str, float],
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

        if rewards["agent_0"] > rewards["agent_1"]:
            return "agent_0"
        if rewards["agent_1"] > rewards["agent_0"]:
            return "agent_1"
        return "draw"

    def _sample_opponent_from_pool(self) -> None:
        """Load one frozen league snapshot into the opponent policy.

        The training loop provides a separate policy copy for the opponent.
        Loading snapshots into that copy keeps historical opponents frozen and
        avoids mutating the live learner.
        """
        self._opponent_snapshot_loaded = False
        if self.opponent_policy is None or self.opponent_pool.is_empty():
            return

        snapshot = self.opponent_pool.sample(
            latest_prob=self.cfg.training.latest_opponent_prob
        )
        target = getattr(self.opponent_policy, "policy", self.opponent_policy)
        if not hasattr(target, "load_state_dict"):
            raise TypeError("opponent_policy must expose load_state_dict or .policy")

        target.load_state_dict(snapshot)
        if hasattr(target, "set_training_mode"):
            target.set_training_mode(False)
        elif hasattr(target, "eval"):
            target.eval()
        self._opponent_snapshot_loaded = True

    def _mirror_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Mirror observation so the opponent sees itself as 'own' agent.

        - Flips grid horizontally
        - Swaps channels 1<->2 (own/opp position)
        - Swaps channels 3<->4 (own/opp bullets)
        - Swaps vector indices 0<->1 (own/opp HP)
        """
        grid = obs["grid"].copy()
        vector = obs["vector"].copy()

        # Flip grid horizontally (along width axis, which is axis 2)
        grid = np.flip(grid, axis=2).copy()

        # Swap own/opp position channels
        grid[[1, 2]] = grid[[2, 1]]

        # Swap own/opp bullet channels
        grid[[3, 4]] = grid[[4, 3]]

        # Swap own/opp HP in vector
        vector[0], vector[1] = vector[1], vector[0]

        return {"grid": grid, "vector": vector}

    def get_state(self) -> dict[str, Any]:
        """Delegate to inner env."""
        return self._env.get_state()

    def set_map_pool(self, map_choices: tuple[str, ...] | None) -> None:
        """Delegate curriculum map-pool updates to the inner env."""
        self._env.set_map_pool(map_choices)

    def set_reward_config(self, reward_config: RewardConfig) -> None:
        """Delegate curriculum reward updates to the inner env."""
        self.cfg = replace(self.cfg, reward=reward_config)
        self._env.set_reward_config(reward_config)

    def render(self) -> str | None:
        return self._env.render()

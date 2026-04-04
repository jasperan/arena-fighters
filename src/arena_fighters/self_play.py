"""Self-play wrapper and opponent pool for SB3 single-agent training."""

from __future__ import annotations

import copy
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arena_fighters.config import NUM_ACTIONS, NUM_CHANNELS, NUM_VECTOR_OBS, Config
from arena_fighters.env import ArenaFightersEnv


class OpponentPool:
    """Stores frozen policy state_dict snapshots for self-play."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._snapshots: list[dict] = []

    def add(self, state_dict: dict) -> None:
        self._snapshots.append(copy.deepcopy(state_dict))
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self, latest_prob: float = 0.8) -> dict:
        assert not self.is_empty(), "Cannot sample from empty pool"
        if len(self._snapshots) == 1 or random.random() < latest_prob:
            return self._snapshots[-1]
        # Pick from all except the latest
        idx = random.randint(0, len(self._snapshots) - 2)
        return self._snapshots[idx]

    def is_empty(self) -> bool:
        return len(self._snapshots) == 0

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
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config or Config()
        self.opponent_pool = opponent_pool or OpponentPool()
        self.opponent_policy = opponent_policy
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

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        obs_dict, info_dict = self._env.reset(seed=seed, options=options)

        agent0_obs = obs_dict["agent_0"]
        agent1_obs = obs_dict["agent_1"]
        self._opponent_obs = self._mirror_obs(agent1_obs)

        return agent0_obs, info_dict.get("agent_0", {})

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

    def render(self) -> str | None:
        return self._env.render()

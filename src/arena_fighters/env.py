"""PettingZoo ParallelEnv for Arena Fighters."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from arena_fighters.config import (
    CH_OPP_BULLETS,
    CH_OPP_POS,
    CH_OWN_BULLETS,
    CH_OWN_FACING,
    CH_OWN_POS,
    CH_PLATFORMS,
    DUCK,
    IDLE,
    JUMP,
    MELEE,
    MOVE_LEFT,
    MOVE_RIGHT,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_VECTOR_OBS,
    PLATFORM_LAYOUT,
    SHOOT_DIAG_DOWN,
    SHOOT_DIAG_UP,
    SHOOT_FORWARD,
    Config,
)


@dataclass
class AgentState:
    x: int
    y: int
    hp: int
    facing: int  # 1 = right, -1 = left
    vy: int = 0  # vertical velocity (negative = upward)
    shoot_cd: int = 0
    melee_cd: int = 0
    duck_ticks: int = 0


@dataclass
class Bullet:
    x: float
    y: float
    dx: int
    dy: int
    owner: str


class ArenaFightersEnv(ParallelEnv):
    """Two-player 2D platform fighter environment."""

    metadata = {"render_modes": ["ansi"], "name": "arena_fighters_v0"}

    def __init__(self, config: Config | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or Config()
        self.render_mode = render_mode

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)

        # Build grid from platform layout
        self._platform_grid = np.zeros(
            (self.cfg.arena.height, self.cfg.arena.width), dtype=np.int8
        )
        for x_start, x_end, y in PLATFORM_LAYOUT:
            self._platform_grid[y, x_start : x_end + 1] = 1

        self._agent_states: dict[str, AgentState] = {}
        self._bullets: list[Bullet] = []
        self._tick = 0
        self._rewards: dict[str, float] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_CHANNELS, self.cfg.arena.height, self.cfg.arena.width),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-1.0, high=1.0, shape=(NUM_VECTOR_OBS,), dtype=np.float32
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        self.agents = list(self.possible_agents)
        self._tick = 0
        self._bullets = []
        self._rewards = {a: 0.0 for a in self.agents}

        hp = self.cfg.agent.start_hp
        self._agent_states = {
            "agent_0": AgentState(x=5, y=18, hp=hp, facing=1),
            "agent_1": AgentState(x=34, y=18, hp=hp, facing=-1),
        }

        obs = {a: self._build_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        self._tick += 1
        self._rewards = {a: 0.0 for a in self.agents}

        # 1) Process actions
        for agent_name, action in actions.items():
            self._process_action(agent_name, action)

        # 2) Apply gravity / physics
        for agent_name in self.agents:
            self._apply_physics(agent_name)

        # 3) Move bullets + check collisions
        self._update_bullets()

        # 4) Decay cooldowns
        for agent_name in self.agents:
            st = self._agent_states[agent_name]
            st.shoot_cd = max(0, st.shoot_cd - 1)
            st.melee_cd = max(0, st.melee_cd - 1)
            st.duck_ticks = max(0, st.duck_ticks - 1)

        # 5) Idle penalty
        for agent_name in self.agents:
            if actions.get(agent_name) == IDLE:
                self._rewards[agent_name] += self.cfg.reward.idle_penalty

        # 6) Check termination
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        for agent_name in self.agents:
            if self._agent_states[agent_name].hp <= 0:
                # This agent died
                terminations[agent_name] = True
                self._rewards[agent_name] += self.cfg.reward.lose
                # Other agent wins
                other = self._other(agent_name)
                terminations[other] = True
                self._rewards[other] += self.cfg.reward.win
                break

        if self._tick >= self.cfg.arena.max_ticks and not any(terminations.values()):
            for a in self.agents:
                truncations[a] = True
                self._rewards[a] += self.cfg.reward.draw

        # 7) Build observations
        obs = {a: self._build_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        # If episode ended, clear agents list
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs, self._rewards, terminations, truncations, infos

    def _process_action(self, agent_name: str, action: int) -> None:
        st = self._agent_states[agent_name]

        if action == MOVE_LEFT:
            st.facing = -1
            new_x = st.x - 1
            if 0 <= new_x < self.cfg.arena.width and not self._is_solid(new_x, st.y):
                st.x = new_x

        elif action == MOVE_RIGHT:
            st.facing = 1
            new_x = st.x + 1
            if 0 <= new_x < self.cfg.arena.width and not self._is_solid(new_x, st.y):
                st.x = new_x

        elif action == JUMP:
            # Can only jump if on ground
            if self._on_ground(st):
                st.vy = -self.cfg.agent.jump_height

        elif action == DUCK:
            st.duck_ticks = self.cfg.agent.duck_duration

        elif action in (SHOOT_FORWARD, SHOOT_DIAG_UP, SHOOT_DIAG_DOWN):
            if st.shoot_cd <= 0:
                dx = st.facing * self.cfg.agent.bullet_speed
                dy = 0
                if action == SHOOT_DIAG_UP:
                    dy = -1
                elif action == SHOOT_DIAG_DOWN:
                    dy = 1
                bx = st.x + st.facing
                by = st.y
                self._bullets.append(
                    Bullet(x=bx, y=by, dx=dx, dy=dy, owner=agent_name)
                )
                st.shoot_cd = self.cfg.agent.shoot_cooldown

        elif action == MELEE:
            if st.melee_cd <= 0:
                target_x = st.x + st.facing
                target_y = st.y
                other = self._other(agent_name)
                ost = self._agent_states[other]
                if ost.x == target_x and ost.y == target_y:
                    dmg = self.cfg.agent.melee_damage
                    ost.hp -= dmg
                    self._rewards[agent_name] += (
                        self.cfg.reward.deal_damage_per_hp * dmg
                    )
                    self._rewards[other] += self.cfg.reward.take_damage_per_hp * dmg
                st.melee_cd = self.cfg.agent.melee_cooldown

    def _apply_physics(self, agent_name: str) -> None:
        st = self._agent_states[agent_name]

        if st.vy < 0:
            # Rising
            new_y = st.y + st.vy  # vy is negative, so this moves up
            # Clamp to top
            new_y = max(0, new_y)
            # Check for solid tiles between current and new position
            if not self._is_solid(st.x, new_y):
                st.y = new_y
            else:
                # Hit ceiling, stop rising
                st.vy = 0
            st.vy += 1  # gravity pulls velocity toward 0 then positive
        elif not self._on_ground(st):
            # Falling: move down 1 tile per tick
            new_y = st.y + 1
            if new_y < self.cfg.arena.height and not self._is_solid(st.x, new_y):
                st.y = new_y
            else:
                # Landed
                st.vy = 0
        else:
            st.vy = 0

    def _update_bullets(self) -> None:
        remaining: list[Bullet] = []
        for b in self._bullets:
            b.x += b.dx
            b.y += b.dy
            bx, by = int(round(b.x)), int(round(b.y))

            # Out of bounds
            if bx < 0 or bx >= self.cfg.arena.width or by < 0 or by >= self.cfg.arena.height:
                continue

            # Hit platform
            if self._is_solid(bx, by):
                continue

            # Check hit on opponent
            hit = False
            for agent_name, st in self._agent_states.items():
                if agent_name == b.owner:
                    continue
                if st.x == bx and st.y == by:
                    # Ducking avoids horizontal bullets only
                    if st.duck_ticks > 0 and b.dy == 0:
                        continue
                    dmg = self.cfg.agent.bullet_damage
                    st.hp -= dmg
                    self._rewards[b.owner] += (
                        self.cfg.reward.deal_damage_per_hp * dmg
                    )
                    self._rewards[agent_name] += (
                        self.cfg.reward.take_damage_per_hp * dmg
                    )
                    hit = True
                    break

            if not hit:
                remaining.append(b)

        self._bullets = remaining

    def _build_obs(self, agent_name: str) -> dict[str, np.ndarray]:
        h, w = self.cfg.arena.height, self.cfg.arena.width
        grid = np.zeros((NUM_CHANNELS, h, w), dtype=np.float32)

        # Channel 0: platforms
        grid[CH_PLATFORMS] = self._platform_grid.astype(np.float32)

        st = self._agent_states[agent_name]
        other = self._other(agent_name)
        ost = self._agent_states[other]

        # Channel 1: own position
        grid[CH_OWN_POS, st.y, st.x] = 1.0

        # Channel 2: opponent position
        grid[CH_OPP_POS, ost.y, ost.x] = 1.0

        # Channel 3/4: bullets
        for b in self._bullets:
            bx, by = int(round(b.x)), int(round(b.y))
            if 0 <= bx < w and 0 <= by < h:
                if b.owner == agent_name:
                    grid[CH_OWN_BULLETS, by, bx] = 1.0
                else:
                    grid[CH_OPP_BULLETS, by, bx] = 1.0

        # Channel 5: own facing direction
        face_x = st.x + st.facing
        if 0 <= face_x < w:
            grid[CH_OWN_FACING, st.y, face_x] = 1.0

        # Vector obs
        vector = np.array(
            [
                st.hp / self.cfg.agent.start_hp,
                ost.hp / self.cfg.agent.start_hp,
                st.shoot_cd / self.cfg.agent.shoot_cooldown,
                st.melee_cd / self.cfg.agent.melee_cooldown,
                st.vy / max(self.cfg.agent.jump_height, 1),
                1.0 if st.duck_ticks > 0 else 0.0,
            ],
            dtype=np.float32,
        )

        return {"grid": grid, "vector": vector}

    def _other(self, agent_name: str) -> str:
        return "agent_1" if agent_name == "agent_0" else "agent_0"

    def _is_solid(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.cfg.arena.width or y < 0 or y >= self.cfg.arena.height:
            return False
        return bool(self._platform_grid[y, x])

    def _on_ground(self, st: AgentState) -> bool:
        below_y = st.y + 1
        if below_y >= self.cfg.arena.height:
            return True
        return self._is_solid(st.x, below_y)

    def _render_ansi(self, score: tuple[int, int] | None = None) -> str:
        h, w = self.cfg.arena.height, self.cfg.arena.width
        st0 = self._agent_states["agent_0"]
        st1 = self._agent_states["agent_1"]
        max_hp = self.cfg.agent.start_hp

        # Box drawing border
        border_w = w + 2
        lines: list[str] = []

        # Title bar
        lines.append(f"\033[1;36m{'ARENA FIGHTERS':^{border_w}}\033[0m")
        lines.append(f"\033[90m{'=' * border_w}\033[0m")

        # Score line (if provided)
        if score is not None:
            score_str = f"Score: \033[1;33m@\033[0m {score[0]}  -  {score[1]} \033[1;35mX\033[0m"
            lines.append(f"  {score_str}")

        # HP bars
        hp0_pct = max(0, st0.hp) / max_hp
        hp1_pct = max(0, st1.hp) / max_hp
        bar_len = 15
        bar0 = "\033[32m" + "█" * int(hp0_pct * bar_len) + "\033[90m" + "░" * (bar_len - int(hp0_pct * bar_len)) + "\033[0m"
        bar1 = "\033[32m" + "█" * int(hp1_pct * bar_len) + "\033[90m" + "░" * (bar_len - int(hp1_pct * bar_len)) + "\033[0m"
        if hp0_pct < 0.3:
            bar0 = "\033[31m" + "█" * int(hp0_pct * bar_len) + "\033[90m" + "░" * (bar_len - int(hp0_pct * bar_len)) + "\033[0m"
        if hp1_pct < 0.3:
            bar1 = "\033[31m" + "█" * int(hp1_pct * bar_len) + "\033[90m" + "░" * (bar_len - int(hp1_pct * bar_len)) + "\033[0m"

        hp_line = f"  \033[1;33m@ {st0.hp:3d}HP\033[0m {bar0}    {bar1} \033[1;35mX {st1.hp:3d}HP\033[0m"
        lines.append(hp_line)

        # Status indicators
        status_parts = [f"  \033[90mTick {self._tick:3d}/{self.cfg.arena.max_ticks}\033[0m"]
        if st0.duck_ticks > 0:
            status_parts.append("\033[33m@ DUCK\033[0m")
        if st1.duck_ticks > 0:
            status_parts.append("\033[33mX DUCK\033[0m")
        if st0.shoot_cd > 0:
            status_parts.append(f"\033[90m@ CD:{st0.shoot_cd}\033[0m")
        if st1.shoot_cd > 0:
            status_parts.append(f"\033[90mX CD:{st1.shoot_cd}\033[0m")
        lines.append("  ".join(status_parts))

        # Top border
        lines.append("\033[90m+" + "-" * w + "+\033[0m")

        # Build display grid
        display = [[" " for _ in range(w)] for _ in range(h)]

        # Platforms
        for y in range(h):
            for x in range(w):
                if self._platform_grid[y, x]:
                    display[y][x] = "\033[90m█\033[0m"

        # Bullets with direction indicators
        for b in self._bullets:
            bx, by = int(round(b.x)), int(round(b.y))
            if 0 <= bx < w and 0 <= by < h:
                if b.dy < 0:
                    char = "/"
                elif b.dy > 0:
                    char = "\\"
                else:
                    char = "-"
                color = "\033[33m" if b.owner == "agent_0" else "\033[35m"
                display[by][bx] = f"{color}{char}\033[0m"

        # Agents with facing indicators
        a0_char = "\033[1;33m@\033[0m" if st0.duck_ticks == 0 else "\033[1;33m_\033[0m"
        a1_char = "\033[1;35mX\033[0m" if st1.duck_ticks == 0 else "\033[1;35m_\033[0m"
        display[st0.y][st0.x] = a0_char
        display[st1.y][st1.x] = a1_char

        for row in display:
            lines.append("\033[90m|\033[0m" + "".join(row) + "\033[90m|\033[0m")

        # Bottom border
        lines.append("\033[90m+" + "-" * w + "+\033[0m")

        # Legend
        lines.append(f"  \033[1;33m@\033[0m Agent 0  \033[1;35mX\033[0m Agent 1  \033[33m-\033[0m bullet  \033[90m█\033[0m platform")

        return "\n".join(lines)

    def render(self, score: tuple[int, int] | None = None) -> str | None:
        if self.render_mode == "ansi":
            return self._render_ansi(score=score)
        return None

    def get_state(self) -> dict[str, Any]:
        """Return serializable dict of current game state for replay logging."""
        return {
            "tick": self._tick,
            "agents": {
                name: {
                    "x": st.x,
                    "y": st.y,
                    "hp": st.hp,
                    "facing": st.facing,
                    "vy": st.vy,
                    "shoot_cd": st.shoot_cd,
                    "melee_cd": st.melee_cd,
                    "duck_ticks": st.duck_ticks,
                }
                for name, st in self._agent_states.items()
            },
            "bullets": [
                {"x": b.x, "y": b.y, "dx": b.dx, "dy": b.dy, "owner": b.owner}
                for b in self._bullets
            ],
        }

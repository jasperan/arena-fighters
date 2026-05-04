"""Tests for ArenaFightersEnv: grid physics, movement, combat mechanics."""

from dataclasses import replace

import numpy as np
import pytest

from arena_fighters.config import (
    DUCK,
    IDLE,
    JUMP,
    MELEE,
    MOVE_LEFT,
    MOVE_RIGHT,
    NUM_CHANNELS,
    NUM_VECTOR_OBS,
    PLATFORM_LAYOUTS,
    SHOOT_DIAG_DOWN,
    SHOOT_DIAG_UP,
    SHOOT_FORWARD,
    Config,
    reward_config_for_preset,
)
from arena_fighters.env import ArenaFightersEnv, Bullet


def _make_env() -> ArenaFightersEnv:
    return ArenaFightersEnv(config=Config())


def _step_idle(env: ArenaFightersEnv):
    """Step with both agents idle."""
    return env.step({"agent_0": IDLE, "agent_1": IDLE})


def _step_one(env: ArenaFightersEnv, agent: str, action: int):
    """Step with one agent acting, other idle."""
    actions = {"agent_0": IDLE, "agent_1": IDLE}
    actions[agent] = action
    return env.step(actions)


# ---------------------------------------------------------------------------
# 1. Basic env creation
# ---------------------------------------------------------------------------
def test_env_creates_two_agents():
    env = _make_env()
    env.reset()
    assert len(env.agents) == 2
    assert "agent_0" in env.agents
    assert "agent_1" in env.agents


# ---------------------------------------------------------------------------
# 2. Observation shape
# ---------------------------------------------------------------------------
def test_observation_shape():
    env = _make_env()
    obs, _ = env.reset()
    for agent in env.possible_agents:
        assert obs[agent]["grid"].shape == (NUM_CHANNELS, 20, 40)
        assert obs[agent]["vector"].shape == (NUM_VECTOR_OBS,)


# ---------------------------------------------------------------------------
# 3. Starting positions
# ---------------------------------------------------------------------------
def test_agents_start_on_ground():
    env = _make_env()
    env.reset()
    st0 = env._agent_states["agent_0"]
    st1 = env._agent_states["agent_1"]
    # agent_0 at (5, 18), agent_1 at (34, 18)
    assert st0.x == 5 and st0.y == 18
    assert st1.x == 34 and st1.y == 18
    # Both should be on ground (y=19 is solid ground)
    assert env._on_ground(st0)
    assert env._on_ground(st1)


def test_can_select_named_map():
    cfg = Config()
    cfg = replace(cfg, arena=replace(cfg.arena, map_name="flat"))
    env = ArenaFightersEnv(config=cfg)
    obs, infos = env.reset()

    assert infos["agent_0"]["map_name"] == "flat"
    assert env.get_state()["map_name"] == "flat"
    assert env._platform_layout == PLATFORM_LAYOUTS["flat"]
    assert obs["agent_0"]["grid"][0, 12, 14] == 0.0


def test_randomized_map_selection_is_seeded():
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(
            cfg.arena,
            randomize_maps=True,
            map_choices=("flat", "tower"),
        ),
    )
    env = ArenaFightersEnv(config=cfg)

    _, infos_a = env.reset(seed=7)
    _, infos_b = env.reset(seed=7)

    assert infos_a["agent_0"]["map_name"] == infos_b["agent_0"]["map_name"]
    assert infos_a["agent_0"]["map_name"] in {"flat", "tower"}


def test_unknown_map_raises_error():
    cfg = Config()
    cfg = replace(cfg, arena=replace(cfg.arena, map_name="missing"))

    with pytest.raises(ValueError, match="Unknown arena map"):
        ArenaFightersEnv(config=cfg)


def test_map_pool_overrides_reset_map_selection():
    env = ArenaFightersEnv(config=Config())
    env.set_map_pool(("flat",))
    _, infos = env.reset(seed=123)

    assert infos["agent_0"]["map_name"] == "flat"
    assert infos["agent_0"]["map_pool"] == ("flat",)
    assert env.get_state()["map_pool"] == ("flat",)


def test_map_pool_validates_choices():
    env = ArenaFightersEnv(config=Config())

    with pytest.raises(ValueError, match="at least one map"):
        env.set_map_pool(())
    with pytest.raises(ValueError, match="Unknown arena map"):
        env.set_map_pool(("missing",))


def test_reward_config_can_be_updated():
    env = ArenaFightersEnv(config=Config())
    anti_stall = reward_config_for_preset("anti_stall")
    env.set_reward_config(anti_stall)
    env.reset()

    _, rewards, _, _, _ = _step_idle(env)

    assert env.cfg.reward == anti_stall
    assert rewards["agent_0"] == pytest.approx(anti_stall.idle_penalty)
    assert rewards["agent_1"] == pytest.approx(anti_stall.idle_penalty)


# ---------------------------------------------------------------------------
# 4-5. Movement
# ---------------------------------------------------------------------------
def test_move_left():
    env = _make_env()
    env.reset()
    start_x = env._agent_states["agent_0"].x
    _step_one(env, "agent_0", MOVE_LEFT)
    assert env._agent_states["agent_0"].x == start_x - 1
    assert env._agent_states["agent_0"].facing == -1


def test_move_right():
    env = _make_env()
    env.reset()
    start_x = env._agent_states["agent_0"].x
    _step_one(env, "agent_0", MOVE_RIGHT)
    assert env._agent_states["agent_0"].x == start_x + 1
    assert env._agent_states["agent_0"].facing == 1


# ---------------------------------------------------------------------------
# 6. Gravity
# ---------------------------------------------------------------------------
def test_gravity_pulls_down():
    env = _make_env()
    env.reset()
    # Place agent in the air (no platform below at y=10)
    env._agent_states["agent_0"].y = 10
    env._agent_states["agent_0"].x = 20  # mid-air, no platform at y=11
    old_y = env._agent_states["agent_0"].y
    _step_idle(env)
    # Agent should have fallen (y increased)
    assert env._agent_states["agent_0"].y > old_y


# ---------------------------------------------------------------------------
# 7. Jump
# ---------------------------------------------------------------------------
def test_jump_gives_upward_velocity():
    env = _make_env()
    env.reset()
    st = env._agent_states["agent_0"]
    assert env._on_ground(st)
    old_y = st.y
    _step_one(env, "agent_0", JUMP)
    # After jump + physics, agent should have moved up
    assert env._agent_states["agent_0"].y < old_y


# ---------------------------------------------------------------------------
# 8. Bounds checking
# ---------------------------------------------------------------------------
def test_cannot_move_outside_bounds():
    env = _make_env()
    env.reset()
    # Place agent at left edge
    env._agent_states["agent_0"].x = 0
    env._agent_states["agent_0"].y = 18
    _step_one(env, "agent_0", MOVE_LEFT)
    assert env._agent_states["agent_0"].x == 0  # clamped


# ---------------------------------------------------------------------------
# 9. Duck
# ---------------------------------------------------------------------------
def test_duck_sets_ducking_state():
    env = _make_env()
    env.reset()
    _step_one(env, "agent_0", DUCK)
    # duck_ticks is set then decremented in same step, so should be duck_duration - 1
    assert env._agent_states["agent_0"].duck_ticks == env.cfg.agent.duck_duration - 1


# ---------------------------------------------------------------------------
# 10. Shoot forward
# ---------------------------------------------------------------------------
def test_shoot_forward_creates_bullet():
    env = _make_env()
    env.reset()
    _step_one(env, "agent_0", SHOOT_FORWARD)
    # After step, bullets have been moved already. Check at least one bullet
    # was created with dy==0. Since bullets move during step, check the bullet
    # list (it may still be alive if it didn't hit anything).
    # Reset and inspect pre-movement state by checking cooldown was set.
    env2 = _make_env()
    env2.reset()
    assert env2._agent_states["agent_0"].shoot_cd == 0
    st = env2._agent_states["agent_0"]
    # Manually call process_action to see bullet creation
    env2._process_action("agent_0", SHOOT_FORWARD)
    assert len(env2._bullets) == 1
    b = env2._bullets[0]
    assert b.dy == 0
    assert b.dx == st.facing * env2.cfg.agent.bullet_speed
    assert b.owner == "agent_0"


# ---------------------------------------------------------------------------
# 11. Shoot diagonal up
# ---------------------------------------------------------------------------
def test_shoot_diag_up_creates_bullet():
    env = _make_env()
    env.reset()
    env._process_action("agent_0", SHOOT_DIAG_UP)
    assert len(env._bullets) == 1
    assert env._bullets[0].dy == -1


# ---------------------------------------------------------------------------
# 12. Shoot diagonal down
# ---------------------------------------------------------------------------
def test_shoot_diag_down_creates_bullet():
    env = _make_env()
    env.reset()
    env._process_action("agent_0", SHOOT_DIAG_DOWN)
    assert len(env._bullets) == 1
    assert env._bullets[0].dy == 1


# ---------------------------------------------------------------------------
# 13. Shoot cooldown
# ---------------------------------------------------------------------------
def test_shoot_cooldown_prevents_firing():
    env = _make_env()
    env.reset()
    _step_one(env, "agent_0", SHOOT_FORWARD)
    cd_after_first = env._agent_states["agent_0"].shoot_cd
    assert cd_after_first > 0
    # Try shooting again immediately
    bullet_count_before = len(env._bullets)
    env._process_action("agent_0", SHOOT_FORWARD)
    # No new bullet because cooldown is active
    assert len(env._bullets) == bullet_count_before


# ---------------------------------------------------------------------------
# 14. Bullet hits opponent
# ---------------------------------------------------------------------------
def test_bullet_hits_opponent():
    env = _make_env()
    env.reset()
    cfg = env.cfg
    # Place agents so bullet reaches opponent in one tick.
    # agent_0 faces right (facing=1), bullet_speed=2.
    # Bullet spawns at (x+1, y), then moves +2 to (x+3, y).
    # Place opponent at x+3.
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_1"].x = 13  # 10 + 1(spawn offset) + 2(speed) = 13
    env._agent_states["agent_1"].y = 18

    hp_before = env._agent_states["agent_1"].hp
    _, _, _, _, infos = _step_one(env, "agent_0", SHOOT_FORWARD)
    assert env._agent_states["agent_1"].hp == hp_before - cfg.agent.bullet_damage
    assert infos["agent_0"]["events"]["shots_fired"] == 1
    assert infos["agent_0"]["events"]["projectile_hits"] == 1
    assert infos["agent_0"]["events"]["damage_dealt"] == cfg.agent.bullet_damage
    assert infos["agent_1"]["events"]["damage_taken"] == cfg.agent.bullet_damage


# ---------------------------------------------------------------------------
# 15. Duck avoids horizontal bullet
# ---------------------------------------------------------------------------
def test_duck_avoids_horizontal_bullet():
    env = _make_env()
    env.reset()
    # Set up same as bullet hit test but opponent is ducking
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_1"].x = 13
    env._agent_states["agent_1"].y = 18
    env._agent_states["agent_1"].duck_ticks = 3  # ducking

    hp_before = env._agent_states["agent_1"].hp
    _step_one(env, "agent_0", SHOOT_FORWARD)
    # Bullet should pass through ducking agent
    assert env._agent_states["agent_1"].hp == hp_before


# ---------------------------------------------------------------------------
# 16. Melee hits adjacent
# ---------------------------------------------------------------------------
def test_melee_hits_adjacent():
    env = _make_env()
    env.reset()
    # Place opponent at x+facing from agent_0
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_1"].x = 11  # x + facing = 11
    env._agent_states["agent_1"].y = 18

    hp_before = env._agent_states["agent_1"].hp
    _, _, _, _, infos = _step_one(env, "agent_0", MELEE)
    assert env._agent_states["agent_1"].hp == hp_before - env.cfg.agent.melee_damage
    assert infos["agent_0"]["events"]["melee_attempts"] == 1
    assert infos["agent_0"]["events"]["melee_hits"] == 1
    assert infos["agent_0"]["events"]["damage_dealt"] == env.cfg.agent.melee_damage
    assert infos["agent_1"]["events"]["damage_taken"] == env.cfg.agent.melee_damage


def test_event_counters_are_cumulative_in_state():
    env = _make_env()
    env.reset()
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_1"].x = 11
    env._agent_states["agent_1"].y = 18

    _step_one(env, "agent_0", MELEE)
    state = env.get_state()

    assert state["events"]["agent_0"]["melee_hits"] == 1
    assert state["episode_events"]["agent_0"]["melee_hits"] == 1


# ---------------------------------------------------------------------------
# 17. Melee misses distant
# ---------------------------------------------------------------------------
def test_melee_misses_distant():
    env = _make_env()
    env.reset()
    # Agents at default positions (far apart)
    hp_before = env._agent_states["agent_1"].hp
    _step_one(env, "agent_0", MELEE)
    assert env._agent_states["agent_1"].hp == hp_before


# ---------------------------------------------------------------------------
# 18. Episode ends on death
# ---------------------------------------------------------------------------
def test_episode_ends_on_death():
    env = _make_env()
    env.reset()
    # Place agents adjacent, set opponent to 1 HP
    env._agent_states["agent_0"].x = 10
    env._agent_states["agent_0"].y = 18
    env._agent_states["agent_0"].facing = 1
    env._agent_states["agent_1"].x = 11
    env._agent_states["agent_1"].y = 18
    env._agent_states["agent_1"].hp = 1

    _, rewards, terms, truncs, _ = _step_one(env, "agent_0", MELEE)
    assert terms["agent_1"] is True
    assert terms["agent_0"] is True
    assert rewards["agent_0"] > 0  # win reward
    assert rewards["agent_1"] < 0  # lose reward
    assert len(env.agents) == 0  # episode over


# ---------------------------------------------------------------------------
# 19. Episode truncates at max ticks
# ---------------------------------------------------------------------------
def test_episode_truncates_at_max_ticks():
    env = _make_env()
    env.reset()
    env._tick = env.cfg.arena.max_ticks - 1
    _, rewards, terms, truncs, _ = _step_idle(env)
    assert truncs["agent_0"] is True
    assert truncs["agent_1"] is True
    assert not terms["agent_0"]
    assert not terms["agent_1"]
    assert rewards["agent_0"] == pytest.approx(
        env.cfg.reward.draw + env.cfg.reward.idle_penalty
    )


def test_anti_stall_adds_no_damage_timeout_penalty():
    cfg = replace(Config(), reward=reward_config_for_preset("anti_stall"))
    env = ArenaFightersEnv(config=cfg)
    env.reset()
    env._tick = env.cfg.arena.max_ticks - 1

    _, rewards, _, truncs, _ = _step_idle(env)

    assert truncs["agent_0"] is True
    assert rewards["agent_0"] == pytest.approx(
        cfg.reward.draw
        + cfg.reward.no_damage_draw_penalty
        + cfg.reward.idle_penalty
    )


def test_no_damage_timeout_penalty_skips_after_damage():
    cfg = replace(Config(), reward=reward_config_for_preset("anti_stall"))
    env = ArenaFightersEnv(config=cfg)
    env.reset()
    env._episode_events["agent_0"]["damage_dealt"] = 1
    env._tick = env.cfg.arena.max_ticks - 1

    _, rewards, _, truncs, _ = _step_idle(env)

    assert truncs["agent_0"] is True
    assert rewards["agent_0"] == pytest.approx(
        cfg.reward.draw + cfg.reward.idle_penalty
    )


# ---------------------------------------------------------------------------
# Extra: render and get_state smoke tests
# ---------------------------------------------------------------------------
def test_render_ansi():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert isinstance(output, str)
    assert "@" in output
    assert "X" in output


def test_get_state_serializable():
    env = _make_env()
    env.reset()
    state = env.get_state()
    assert "tick" in state
    assert "agents" in state
    assert "bullets" in state
    assert state["agents"]["agent_0"]["x"] == 5

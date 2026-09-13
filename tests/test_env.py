"""Tests for ArenaFightersEnv: grid physics, movement, combat mechanics."""

from dataclasses import replace

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
    ArenaConfig,
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
    # Mirror-symmetric spawns on the ground row; the exact columns may be
    # jittered symmetrically by the episode seed.
    assert st0.y == 18 and st1.y == 18
    assert st0.x + st1.x == env.cfg.arena.width - 1
    assert abs(st0.x - 5) <= env.cfg.arena.spawn_jitter
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


def test_double_knockout_is_symmetric_draw():
    """Simultaneous deaths must not hand agent_1 a win reward over agent_0.

    Regression: the termination loop processed only the first dead agent in
    iteration order, so a double KO always paid agent_0 the lose reward and
    agent_1 the win reward (+20 reward gap per episode).
    """
    env = _make_env()
    env.reset()
    env._agent_states["agent_0"].hp = 0
    env._agent_states["agent_1"].hp = 0

    _, rewards, terms, truncs, _ = _step_idle(env)

    assert terms["agent_0"] is True
    assert terms["agent_1"] is True
    assert not any(truncs.values())
    assert rewards["agent_0"] == pytest.approx(rewards["agent_1"])
    assert rewards["agent_0"] == pytest.approx(
        env.cfg.reward.draw + env.cfg.reward.idle_penalty
    )


def test_single_knockout_pays_win_and_lose_rewards():
    env = _make_env()
    env.reset()
    env._agent_states["agent_1"].hp = 0

    _, rewards, terms, truncs, _ = _step_idle(env)

    assert terms["agent_0"] is True
    assert terms["agent_1"] is True
    assert not any(truncs.values())
    assert rewards["agent_0"] == pytest.approx(
        env.cfg.reward.win + env.cfg.reward.idle_penalty
    )
    assert rewards["agent_1"] == pytest.approx(
        env.cfg.reward.lose + env.cfg.reward.idle_penalty
    )


_MIRRORED_ACTION = {
    0: 0,  # IDLE
    1: MOVE_RIGHT,
    2: MOVE_LEFT,
    3: 3,  # JUMP
    4: 4,  # DUCK
    5: 5,  # SHOOT_FORWARD
    6: 6,  # SHOOT_DIAG_UP
    7: 7,  # SHOOT_DIAG_DOWN
    8: 8,  # MELEE
}


@pytest.mark.parametrize("map_name", sorted(PLATFORM_LAYOUTS))
def test_mirrored_action_stream_stays_mirror_symmetric(map_name):
    """Shared-weight self-play relies on mirror symmetry of every map.

    Feeding mirrored action streams from mirrored spawns must keep the match
    an exact mirror: |x0 + x1| stays at the arena center, HPs stay equal,
    facings stay opposite, and vertical state stays identical. Any map or
    physics asymmetry breaks these invariants.
    """
    cfg = replace(Config(), arena=replace(Config().arena, map_name=map_name))
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)
    center = env.cfg.arena.width - 1
    action_stream = [2, 5, 2, 2, 5, 1, 3, 5, 4, 2, 5, 1, 2, 5, 3, 5, 1, 1, 5, 4,
                     2, 5, 2, 3, 5, 1, 2, 5, 2, 5]

    for action_0 in action_stream:
        s0 = env._agent_states["agent_0"]
        s1 = env._agent_states["agent_1"]
        assert s0.x + s1.x == center
        assert s0.hp == s1.hp
        assert s0.facing == -s1.facing
        assert s0.vy == s1.vy

        env.step({"agent_0": action_0, "agent_1": _MIRRORED_ACTION[action_0]})
        if not env.agents:
            break

    s0 = env._agent_states["agent_0"]
    s1 = env._agent_states["agent_1"]
    assert s0.hp == s1.hp


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
    assert (
        state["agents"]["agent_0"]["x"] + state["agents"]["agent_1"]["x"]
        == env.cfg.arena.width - 1
    )


# ---------------------------------------------------------------------------
# Bullet sweep: fast projectiles must not skip tiles on their path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("map_name", sorted(PLATFORM_LAYOUTS))
@pytest.mark.parametrize("offset", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_bullet_hits_stationary_target_at_every_offset(map_name, offset):
    """Regression: bullets moved 2 tiles per tick and only tested the
    destination tile, so shots fired at an even offset from a stationary
    opponent flew straight through it: the target was unhittable, and an
    arena position that granted immunity to horizontal fire existed on every
    map. Sweeping the path makes every reachable offset connect.
    """
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name=map_name)))
    env.reset(seed=0)
    gunner = env._agent_states["agent_0"]
    target = env._agent_states["agent_1"]
    gunner.x, gunner.y, gunner.facing, gunner.shoot_cd = 10, 18, 1, 0
    target.x, target.y = 10 + offset, 18

    for _ in range(12):
        if not env.agents:
            break
        env.step({"agent_0": SHOOT_FORWARD, "agent_1": IDLE})

    assert target.hp <= 0, f"offset {offset} on {map_name} never connected"


def test_bullet_hits_target_behind_the_shot_direction():
    """Mirror of the above for a shooter facing left."""
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name="flat")))
    env.reset(seed=0)
    gunner = env._agent_states["agent_1"]
    target = env._agent_states["agent_0"]
    gunner.x, gunner.y, gunner.facing, gunner.shoot_cd = 30, 18, -1, 0
    target.x, target.y = 26, 18

    for _ in range(12):
        if not env.agents:
            break
        env.step({"agent_0": IDLE, "agent_1": SHOOT_FORWARD})

    assert target.hp <= 0


def test_bullet_stops_on_intermediate_platform_tile():
    """A bullet crossing a solid tile mid-flight is blocked there."""
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name="classic")))
    env.reset(seed=0)
    # classic low platform spans x=1..4 at y=15; start a bullet so that its
    # second substep enters the span.
    env._bullets = [Bullet(x=2.0, y=15.0, dx=2, dy=0, owner="agent_0")]

    env.step({"agent_0": IDLE, "agent_1": IDLE})

    assert env._bullets == []


def test_bullet_keeps_final_position_of_each_tick():
    """Sweeping must not change bullet trajectories or observation slots."""
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name="flat")))
    env.reset(seed=0)
    env._bullets = [Bullet(x=10.0, y=18.0, dx=2, dy=0, owner="agent_0")]

    env.step({"agent_0": IDLE, "agent_1": IDLE})

    assert [(b.x, b.y) for b in env._bullets] == [(12.0, 18.0)]


# ---------------------------------------------------------------------------
# Jump ceiling: rising jumps must not pass through solid tiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("map_name", sorted(PLATFORM_LAYOUTS))
def test_jump_never_crosses_a_solid_tile(map_name):
    """Regression: a jump moved `jump_height` tiles in one tick and only tested
    the destination, so on tower -- whose platforms are exactly jump_height
    apart -- a fighter under a platform jumped straight through it and landed
    on top. Every standing spot on every map must stop rising below the first
    solid tile above it.
    """
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name=map_name)))
    env.reset(seed=0)
    agent = env._agent_states["agent_0"]

    for x in range(env.cfg.arena.width):
        for y in range(env.cfg.arena.height - 1):
            if env._is_solid(x, y) or not env._is_solid(x, y + 1):
                continue  # not a standing position
            agent.x, agent.y, agent.vy = x, y, 0
            env._agent_states["agent_1"].x = 39 - x
            env._agent_states["agent_1"].y = y
            env.step({"agent_0": JUMP, "agent_1": IDLE})
            crossed = [yy for yy in range(agent.y, y) if env._is_solid(x, yy)]
            assert not crossed, (
                f"{map_name}: jumped from ({x},{y}) to y={agent.y} through {crossed}"
            )


def test_jump_bumps_head_and_stays_below_platform():
    """Under a low platform the fighter rises to just below it, then falls."""
    env = ArenaFightersEnv(
        config=Config(arena=replace(Config().arena, map_name="classic"))
    )
    env.reset(seed=0)
    agent = env._agent_states["agent_0"]
    agent.x, agent.y, agent.vy = 2, 18, 0  # low platform (1..4, y=15) overhead

    env.step({"agent_0": JUMP, "agent_1": IDLE})

    assert agent.y == 16  # rose to just below the platform at y=15
    env.step({"agent_0": IDLE, "agent_1": IDLE})
    assert agent.y == 16  # still pinned under the ceiling
    assert agent.vy >= 0  # no upward velocity left after the ceiling stop


def test_tower_platform_cannot_be_mounted_from_directly_below():
    """Platforms are still only reachable from beside them."""
    env = ArenaFightersEnv(config=Config(arena=replace(Config().arena, map_name="tower")))
    env.reset(seed=0)
    agent = env._agent_states["agent_0"]
    agent.x, agent.y, agent.vy = 19, 18, 0  # under the (17..22, y=16) platform

    for _ in range(6):
        env.step({"agent_0": JUMP, "agent_1": IDLE})
        assert agent.y >= 17, "mounted a platform from directly underneath"


# ---------------------------------------------------------------------------
# Spawn jitter: seeded episodes must differ, and stay mirror-symmetric
# ---------------------------------------------------------------------------
def test_spawn_jitter_varies_openings_and_stays_mirror_symmetric():
    """Deterministic policies replay one episode per seed without variety.

    Seeded spawns are shifted symmetrically toward or away from the centre, so
    N-round evaluations sample distinct openings while the match stays a mirror
    image of itself.
    """
    env = ArenaFightersEnv(config=Config(arena=replace(ArenaConfig(), map_name="flat")))
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        a0 = env._agent_states["agent_0"]
        a1 = env._agent_states["agent_1"]
        assert a0.x + a1.x == env.cfg.arena.width - 1
        assert not env._is_solid(a0.x, a0.y) and not env._is_solid(a1.x, a1.y)
        assert env._on_ground(a0) and env._on_ground(a1)
        seen.add(a0.x)
    assert len(seen) > 1, "spawn jitter never changed the opening"


def test_spawn_jitter_is_reproducible_for_a_seed():
    env = ArenaFightersEnv(config=Config(arena=replace(ArenaConfig(), map_name="classic")))
    env.reset(seed=11)
    first = {name: st.x for name, st in env._agent_states.items()}
    env.reset(seed=11)
    assert {name: st.x for name, st in env._agent_states.items()} == first


def test_spawn_jitter_zero_pins_classic_columns():
    env = ArenaFightersEnv(config=Config(arena=replace(ArenaConfig(), spawn_jitter=0)))
    for seed in range(5):
        env.reset(seed=seed)
        assert env._agent_states["agent_0"].x == 5
        assert env._agent_states["agent_1"].x == 34


# ---------------------------------------------------------------------------
# Passive-distance shaping (engagement preset)
# ---------------------------------------------------------------------------
def test_far_penalty_is_zero_when_close_and_scales_with_distance():
    cfg = Config(
        arena=replace(ArenaConfig(), map_name="flat"),
        reward=reward_config_for_preset("engagement"),
    )
    assert cfg.reward.far_distance == 6 and cfg.reward.far_penalty_per_tile == 0.005
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)
    a0, a1 = env._agent_states["agent_0"], env._agent_states["agent_1"]

    # 20 tiles apart: 14 excess tiles * 0.005, plus the idle penalty both pay.
    a0.x, a1.x = 5, 25
    _, rewards, _, _, _ = _step_idle(env)
    expected = -0.005 * (20 - 6) + cfg.reward.idle_penalty
    assert rewards["agent_0"] == pytest.approx(expected)
    assert rewards["agent_1"] == pytest.approx(expected)

    # Inside the free radius nothing extra is charged.
    env.reset(seed=0)
    a0, a1 = env._agent_states["agent_0"], env._agent_states["agent_1"]
    a0.x, a1.x = 10, 14
    _, rewards, _, _, _ = _step_idle(env)
    assert rewards["agent_0"] == pytest.approx(cfg.reward.idle_penalty)


def test_presets_without_shaping_are_unaffected():
    env = ArenaFightersEnv(config=Config(arena=replace(ArenaConfig(), map_name="flat")))
    env.reset(seed=0)
    env._agent_states["agent_0"].x = 2
    env._agent_states["agent_1"].x = 30

    _, rewards, _, _, _ = _step_idle(env)

    assert rewards["agent_0"] == pytest.approx(Config().reward.idle_penalty)


# ---------------------------------------------------------------------------
# Duck cooldown: capping how long a turtle can stay bullet-proof
# ---------------------------------------------------------------------------
def _duck_uptime(duck_cooldown: int, ticks: int = 12) -> int:
    """How many incoming horizontal bullets a dedicated ducker blocks.

    Measured functionally (block a bullet vs get hit) rather than by reading
    ``duck_ticks`` after the step, because the counter decays inside the same
    step and hides the second covered tick.
    """
    cfg = Config(
        arena=replace(ArenaConfig(), map_name="flat"),
        agent=replace(Config().agent, duck_cooldown=duck_cooldown),
    )
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)
    ducker = env._agent_states["agent_0"]
    blocked = 0
    for _ in range(ticks):
        # Fresh cushion each tick so a leaking shot registers as damage without
        # ending the episode (start_hp is 1).
        ducker.hp = 25
        env._bullets = [
            Bullet(x=float(ducker.x - 2), y=float(ducker.y), dx=2, dy=0, owner="agent_1")
        ]
        env.step({"agent_0": DUCK, "agent_1": IDLE})
        if ducker.hp == 25:
            blocked += 1
    return blocked


def test_duck_cooldown_halves_duck_cover():
    """duck_duration 2 + cooldown 2 covers two of every four ticks, so half of
    the incoming horizontal fire gets through: ducking can no longer be held."""
    assert _duck_uptime(duck_cooldown=2) == 6


def test_duck_cooldown_zero_keeps_continuous_cover():
    """Default behaviour is unchanged: alternate ducks block every shot."""
    assert _duck_uptime(duck_cooldown=0) == 12


def test_duck_cooldown_is_symmetric_and_blocks_back_to_back_ducks():
    cfg = Config(
        arena=replace(ArenaConfig(), map_name="flat"),
        agent=replace(Config().agent, duck_cooldown=3),
    )
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)

    env.step({"agent_0": DUCK, "agent_1": DUCK})
    assert env._agent_states["agent_0"].duck_ticks > 0
    assert env._agent_states["agent_1"].duck_ticks > 0

    # A second duck in a row is refused for both agents.
    env.step({"agent_0": DUCK, "agent_1": DUCK})
    assert env._agent_states["agent_0"].duck_ticks == 0
    assert env._agent_states["agent_1"].duck_ticks == 0
    assert env._agent_states["agent_0"].duck_cooldown_ticks > 0
    assert env._agent_states["agent_1"].duck_cooldown_ticks > 0


def test_far_penalty_knobs_scale_with_configuration():
    """The distance penalty is configurable so a run can test whether paying
    for passivity actually changes where a policy fights."""
    cfg = Config(
        arena=replace(ArenaConfig(), map_name="flat"),
        reward=replace(
            reward_config_for_preset("default"),
            far_distance=4,
            far_penalty_per_tile=0.02,
        ),
    )
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)
    env._agent_states["agent_0"].x = 5
    env._agent_states["agent_1"].x = 15  # 10 tiles, 6 excess

    _, rewards, _, _, _ = _step_idle(env)

    assert rewards["agent_0"] == pytest.approx(-0.02 * 6 + cfg.reward.idle_penalty)


def test_set_duck_cooldown_changes_behaviour_mid_episode():
    """Schedule support: the recovery window can be relaxed without rebuilding
    the environment. Measured functionally (shots blocked), because duck_ticks
    decays inside the step and reads 0 on every other covered tick."""
    cfg = Config(
        arena=replace(ArenaConfig(), map_name="flat"),
        agent=replace(Config().agent, duck_cooldown=3),
    )
    env = ArenaFightersEnv(config=cfg)
    env.reset(seed=0)
    ducker = env._agent_states["agent_0"]

    def blocked_shots(ticks: int) -> int:
        blocked = 0
        for _ in range(ticks):
            ducker.hp = 25
            env._bullets = [
                Bullet(
                    x=float(ducker.x - 2),
                    y=float(ducker.y),
                    dx=2,
                    dy=0,
                    owner="agent_1",
                )
            ]
            env.step({"agent_0": DUCK, "agent_1": IDLE})
            if ducker.hp == 25:
                blocked += 1
        return blocked

    with_cooldown = blocked_shots(8)
    env.set_duck_cooldown(0)
    without_cooldown = blocked_shots(8)

    assert with_cooldown < without_cooldown
    # Relaxing the config does not cancel the timer already running, so the
    # first two ticks are still refused and six of eight shots are blocked.
    assert without_cooldown == 6

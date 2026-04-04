from arena_fighters.config import Config
from arena_fighters.env import ArenaFightersEnv, Bullet


def test_ansi_render_contains_agents():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "@" in output
    assert "X" in output


def test_ansi_render_contains_hp():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "HP:" in output
    assert "100" in output


def test_ansi_render_contains_platforms():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "=" in output


def test_ansi_render_shows_bullets():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._bullets.append(Bullet(x=20, y=10, dx=2, dy=0, owner="agent_0"))
    output = env.render()
    assert "*" in output


def test_ansi_render_shows_tick():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._tick = 42
    output = env.render()
    assert "Tick 42" in output

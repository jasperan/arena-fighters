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
    assert "1HP" in output


def test_ansi_render_contains_platforms():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "platform" in output  # legend text


def test_ansi_render_shows_bullets():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._bullets.append(Bullet(x=20, y=10, dx=2, dy=0, owner="agent_0"))
    output = env.render()
    assert "-" in output  # horizontal bullet char


def test_ansi_render_shows_tick():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._tick = 42
    output = env.render()
    assert "42" in output

from arena_fighters.config import (
    Config,
    PLATFORM_LAYOUT,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_VECTOR_OBS,
)


def test_config_defaults():
    cfg = Config()
    assert cfg.arena.width == 40
    assert cfg.arena.height == 20
    assert cfg.agent.start_hp == 1
    assert cfg.training.total_timesteps == 10_000_000


def test_platform_layout_within_bounds():
    cfg = Config()
    for x_start, x_end, y in PLATFORM_LAYOUT:
        assert 0 <= x_start <= x_end < cfg.arena.width
        assert 0 <= y < cfg.arena.height


def test_action_count():
    assert NUM_ACTIONS == 9


def test_observation_constants():
    assert NUM_CHANNELS == 6
    assert NUM_VECTOR_OBS == 6

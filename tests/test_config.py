from dataclasses import replace

from arena_fighters.config import (
    CH_OPP_BULLETS,
    CH_OPP_POS,
    CH_OWN_BULLETS,
    CH_OWN_FACING,
    CH_OWN_POS,
    CH_PLATFORMS,
    Config,
    CURRICULUMS,
    PLATFORM_LAYOUT,
    PLATFORM_LAYOUTS,
    REWARD_PRESETS,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_VECTOR_OBS,
    VEC_DUCKING,
    VEC_MELEE_COOLDOWN,
    VEC_OPP_HP,
    VEC_OWN_HP,
    VEC_SHOOT_COOLDOWN,
    VEC_VERTICAL_VELOCITY,
    curriculum_for_name,
    curriculum_stage_for_step,
    reward_config_for_preset,
    validate_curriculum,
)


def test_config_defaults():
    cfg = Config()
    assert cfg.arena.width == 40
    assert cfg.arena.height == 20
    assert cfg.agent.start_hp == 1
    assert cfg.training.total_timesteps == 10_000_000


def test_platform_layout_within_bounds():
    cfg = Config()
    for layout in PLATFORM_LAYOUTS.values():
        for x_start, x_end, y in layout:
            assert 0 <= x_start <= x_end < cfg.arena.width
            assert 0 <= y < cfg.arena.height


def test_all_map_layouts_are_mirror_symmetric():
    """Every map must map onto itself under horizontal mirroring.

    The shared self-play policy plays agent_1 through mirrored observations,
    so an asymmetric map (regression: `classic` had off-center low/mid
    platforms) breaks the invariant the whole shared-weight design relies on.
    """
    width = Config().arena.width
    for map_name, layout in PLATFORM_LAYOUTS.items():
        cells = {
            (x, y)
            for x_start, x_end, y in layout
            for x in range(x_start, x_end + 1)
        }
        mirrored = {(width - 1 - x, y) for x, y in cells}
        assert mirrored == cells, f"map '{map_name}' is not mirror-symmetric"


def test_default_platform_layout_alias():
    assert PLATFORM_LAYOUT == list(PLATFORM_LAYOUTS["classic"])


def test_default_map_choices_exist():
    cfg = Config()
    assert cfg.arena.map_name in PLATFORM_LAYOUTS
    for map_name in cfg.arena.map_choices:
        assert map_name in PLATFORM_LAYOUTS


def test_default_reward_preset_matches_config_default():
    assert reward_config_for_preset("default") == Config().reward


def test_anti_stall_reward_preset_is_more_engagement_weighted():
    default_reward = REWARD_PRESETS["default"]
    anti_stall = reward_config_for_preset("anti-stall")

    assert anti_stall.draw < default_reward.draw
    assert anti_stall.no_damage_draw_penalty < default_reward.no_damage_draw_penalty
    assert anti_stall.idle_penalty < default_reward.idle_penalty
    assert anti_stall.deal_damage_per_hp > default_reward.deal_damage_per_hp
    assert anti_stall.take_damage_per_hp < default_reward.take_damage_per_hp


def test_unknown_reward_preset_raises_error():
    import pytest

    with pytest.raises(ValueError, match="Unknown reward preset"):
        reward_config_for_preset("missing")


def test_curriculum_definitions_are_valid():
    for curriculum_name in CURRICULUMS:
        assert validate_curriculum(curriculum_name) is True


def test_curriculum_stage_selection_by_step():
    assert curriculum_stage_for_step("map_progression", 0).name == "flat_intro"
    assert (
        curriculum_stage_for_step("map_progression", 250_000).name
        == "classic_duel"
    )
    assert (
        curriculum_stage_for_step("map_progression", 1_500_000).name
        == "mixed_routes"
    )
    assert (
        curriculum_stage_for_step("map_progression", 3_000_000).name
        == "full_map_pool"
    )


def test_curriculum_stage_maps_and_rewards_exist():
    stages = curriculum_for_name("map_progression")
    for stage in stages:
        for map_name in stage.map_choices:
            assert map_name in PLATFORM_LAYOUTS
        assert stage.reward_preset in {"default", "anti_stall"}


def test_action_count():
    assert NUM_ACTIONS == 9


def test_observation_constants():
    assert (
        CH_PLATFORMS,
        CH_OWN_POS,
        CH_OPP_POS,
        CH_OWN_BULLETS,
        CH_OPP_BULLETS,
        CH_OWN_FACING,
    ) == tuple(range(NUM_CHANNELS))
    assert NUM_CHANNELS == 6
    assert (
        VEC_OWN_HP,
        VEC_OPP_HP,
        VEC_SHOOT_COOLDOWN,
        VEC_MELEE_COOLDOWN,
        VEC_VERTICAL_VELOCITY,
        VEC_DUCKING,
    ) == tuple(range(NUM_VECTOR_OBS))
    assert NUM_VECTOR_OBS == 6


def test_entropy_coefficient_defaults_to_zero_and_is_configurable():
    """Sampled-action entropy can stay high while the greedy action collapses,
    so the entropy bonus is an explicit, recorded training knob."""
    assert Config().training.ent_coef == 0.0
    cfg = Config(training=replace(Config().training, ent_coef=0.02))
    assert cfg.training.ent_coef == 0.02

"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


def test_self_play_callback_applies_curriculum_stages():
    wrapper = FakeWrapper()
    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=OpponentPool(),
        cfg=replace(
            Config(),
            training=replace(Config().training, curriculum_name="map_progression"),
        ),
        curriculum_name="map_progression",
    )

    callback.num_timesteps = 0
    callback._apply_curriculum()
    callback.num_timesteps = 250_000
    callback._apply_curriculum()

    assert wrapper.map_pools == [
        ("flat",),
        ("flat", "classic"),
    ]
    assert wrapper.reward_configs == [
        reward_config_for_preset("default"),
        reward_config_for_preset("default"),
    ]


def test_self_play_callback_applies_curriculum_reward_presets():
    wrapper = FakeWrapper()
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )
    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=OpponentPool(),
        cfg=cfg,
        curriculum_name="map_progression",
    )

    callback.num_timesteps = 0
    callback._apply_curriculum()
    callback.num_timesteps = 1_000_000
    callback._apply_curriculum()

    assert wrapper.map_pools == [
        ("flat",),
        ("classic", "split"),
    ]
    assert wrapper.reward_configs == [
        reward_config_for_preset("default"),
        reward_config_for_preset("anti_stall"),
    ]


def test_build_training_wrapper_wires_replay_logger(tmp_path):
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            replay_save_interval=7,
            opponent_pool_seed=123,
        ),
    )

    wrapper, pool = build_training_wrapper(cfg, str(tmp_path))
    expected_pool = OpponentPool(max_size=cfg.training.opponent_pool_size, seed=123)
    for i in range(5):
        pool.add({"weight": i})
        expected_pool.add({"weight": i})

    assert isinstance(pool, OpponentPool)
    assert [pool.sample(latest_prob=0.4)["weight"] for _ in range(10)] == [
        expected_pool.sample(latest_prob=0.4)["weight"] for _ in range(10)
    ]
    assert wrapper.replay_logger is not None
    assert wrapper.replay_logger.replay_dir == tmp_path
    assert wrapper.replay_logger.save_every_n == 7


def test_self_play_callback_records_opponent_pool_stats():
    pool = OpponentPool()
    pool.add({"weight": 0})
    pool.add({"weight": 1})
    pool.sample(latest_prob=1.0)
    pool.sample(latest_prob=0.0)
    callback = SelfPlayCallback(
        wrapper=FakeWrapper(),
        opponent_pool=pool,
        cfg=Config(),
    )
    model = FakeModelWithLogger()
    callback.model = model

    callback._record_self_play_stats()

    assert model.logger.records == {
        "self_play/opponent_pool_size": 2,
        "self_play/latest_opponent_samples": 1,
        "self_play/historical_opponent_samples": 1,
        "self_play/historical_sample_rate": 0.5,
        "self_play/latest_opponent_snapshot_id": 1,
        "self_play/last_opponent_snapshot_id": 0,
        "self_play/last_sample_was_historical": 1.0,
    }


def test_curriculum_metadata_records_active_stage():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    metadata = curriculum_metadata(cfg, step=1_500_000)

    assert metadata["name"] == "map_progression"
    assert metadata["stage"]["name"] == "mixed_routes"
    assert metadata["active_map_pool"] == ["classic", "split"]


def test_effective_reward_config_uses_curriculum_stage_reward():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    assert effective_reward_config(cfg, 0) == reward_config_for_preset("default")
    assert effective_reward_config(cfg, 1_000_000) == reward_config_for_preset(
        "anti_stall"
    )

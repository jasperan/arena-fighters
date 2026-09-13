"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403

import numpy as np
import pytest


def test_self_play_callback_records_action_collapse_telemetry():
    callback = SelfPlayCallback(
        wrapper=FakeWrapper(),
        opponent_pool=OpponentPool(),
        cfg=Config(),
    )
    model = FakeModelWithLogger()
    callback.model = model
    callback.num_timesteps = 0
    callback.locals = {"actions": np.array([4, 4, 4, 5])}

    callback._on_step()
    callback._record_action_stats()

    records = model.logger.records
    assert records["self_play/dominant_action"] == 4
    assert records["self_play/dominant_action_share"] == pytest.approx(0.75)
    assert records["self_play/action_samples"] == 4
    assert 0.0 < records["self_play/action_entropy"] < 1.0
    assert all(count == 0 for count in callback._action_counts)


def test_self_play_callback_action_entropy_is_zero_when_policy_collapses():
    callback = SelfPlayCallback(
        wrapper=FakeWrapper(),
        opponent_pool=OpponentPool(),
        cfg=Config(),
    )
    model = FakeModelWithLogger()
    callback.model = model
    callback.num_timesteps = 0
    callback.locals = {"actions": np.array([4, 4, 4, 4, 4, 4])}

    callback._on_step()
    callback._record_action_stats()

    records = model.logger.records
    assert records["self_play/action_entropy"] == pytest.approx(0.0)
    assert records["self_play/dominant_action_share"] == pytest.approx(1.0)


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
        "self_play/scripted_opponent_samples": 0,
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


def test_duck_cooldown_schedule_relaxes_at_the_configured_step():
    """The cooldown that produced real movement at 100K faded by 1M, so the
    schedule keeps it while the policy forms and releases it later."""
    wrapper = FakeWrapper()
    cfg = Config()
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, duck_cooldown=3),
        training=replace(cfg.training, duck_cooldown_until=250_000),
    )
    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=OpponentPool(),
        cfg=cfg,
    )
    callback.model = FakeModelWithLogger()

    callback.num_timesteps = 100_000
    callback._apply_duck_cooldown_schedule()
    assert wrapper.duck_cooldowns == []
    assert cfg.agent.duck_cooldown == 3

    callback.num_timesteps = 250_000
    callback._apply_duck_cooldown_schedule()
    callback._apply_duck_cooldown_schedule()  # idempotent

    assert wrapper.duck_cooldowns == [0]
    assert wrapper.cfg.agent.duck_cooldown == 0


def test_duck_cooldown_stages_apply_in_order_once_each():
    """Staged release: 3 -> 2 -> 1 -> 0, each stage firing exactly once."""
    wrapper = FakeWrapper()
    cfg = Config()
    cfg = replace(
        cfg,
        agent=replace(cfg.agent, duck_cooldown=3),
        training=replace(
            cfg.training,
            duck_cooldown_stages=((200_000, 2), (350_000, 1), (500_000, 0)),
        ),
    )
    callback = SelfPlayCallback(wrapper=wrapper, opponent_pool=OpponentPool(), cfg=cfg)
    callback.model = FakeModelWithLogger()

    for step in (0, 199_999, 200_000, 349_999, 350_000, 499_999, 500_000, 900_000):
        callback.num_timesteps = step
        callback._apply_duck_cooldown_schedule()

    assert wrapper.duck_cooldowns == [2, 1, 0]
    assert wrapper.cfg.agent.duck_cooldown == 0
    assert callback.duck_cooldown_stage == {"step": 500_000, "cooldown": 0}

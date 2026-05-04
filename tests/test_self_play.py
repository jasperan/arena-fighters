from dataclasses import replace

import numpy as np
from arena_fighters.config import Config, IDLE, MOVE_LEFT, reward_config_for_preset
from arena_fighters.replay import ReplayLogger, load_replay
from arena_fighters.self_play import SelfPlayWrapper, OpponentPool


class RecordingPolicy:
    def __init__(self):
        self.loaded_snapshots = []
        self.action = IDLE
        self.training_mode = True

    def load_state_dict(self, state_dict):
        self.loaded_snapshots.append(dict(state_dict))
        self.action = state_dict["action"]

    def set_training_mode(self, mode):
        self.training_mode = mode

    def predict(self, obs, deterministic=False):
        return self.action, None


def test_opponent_pool_add_and_sample():
    pool = OpponentPool(max_size=5)
    for i in range(5):
        pool.add({"weight": i})
    assert len(pool) == 5
    snapshot = pool.sample(latest_prob=1.0)
    assert snapshot["weight"] == 4


def test_opponent_pool_evicts_oldest():
    pool = OpponentPool(max_size=3)
    for i in range(5):
        pool.add({"weight": i})
    assert len(pool) == 3
    assert pool._snapshots[0]["weight"] == 2
    assert pool.stats()["snapshot_ids"] == [2, 3, 4]


def test_opponent_pool_random_sample():
    pool = OpponentPool(max_size=10)
    for i in range(10):
        pool.add({"weight": i})
    np.random.seed(42)
    samples = [pool.sample(latest_prob=0.0)["weight"] for _ in range(20)]
    assert len(set(samples)) > 1


def test_opponent_pool_tracks_latest_and_historical_samples():
    pool = OpponentPool(max_size=3)
    pool.add({"weight": 0})
    pool.add({"weight": 1})
    pool.add({"weight": 2})

    latest = pool.sample(latest_prob=1.0)
    historical = pool.sample(latest_prob=0.0)

    assert latest["weight"] == 2
    assert historical["weight"] in {0, 1}
    assert pool.last_sample_index in {0, 1}
    stats = pool.stats()
    assert stats["size"] == 3
    assert stats["latest_samples"] == 1
    assert stats["historical_samples"] == 1
    assert stats["historical_sample_rate"] == 0.5
    assert stats["snapshot_ids"] == [0, 1, 2]
    assert stats["oldest_snapshot_id"] == 0
    assert stats["latest_snapshot_id"] == 2
    assert stats["last_sample_index"] == pool.last_sample_index
    assert stats["last_sample_id"] in {0, 1}
    assert stats["last_sample_kind"] == "historical"
    assert stats["snapshots"] == [
        {
            "id": 0,
            "index": 0,
            "sample_count": 1 if stats["last_sample_id"] == 0 else 0,
            "is_latest": False,
        },
        {
            "id": 1,
            "index": 1,
            "sample_count": 1 if stats["last_sample_id"] == 1 else 0,
            "is_latest": False,
        },
        {
            "id": 2,
            "index": 2,
            "sample_count": 1,
            "is_latest": True,
        },
    ]


def test_opponent_pool_sample_returns_frozen_snapshot_copy():
    pool = OpponentPool(max_size=3)
    pool.add({"nested": {"weight": 1}})

    sampled = pool.sample(latest_prob=1.0)
    sampled["nested"]["weight"] = 99

    assert pool.sample(latest_prob=1.0)["nested"]["weight"] == 1


def test_self_play_wrapper_is_single_agent():
    cfg = Config()
    wrapper = SelfPlayWrapper(config=cfg)
    obs, _ = wrapper.reset()
    assert "grid" in obs
    assert "vector" in obs
    assert hasattr(wrapper, "observation_space")
    assert hasattr(wrapper, "action_space")


def test_self_play_wrapper_step():
    cfg = Config()
    wrapper = SelfPlayWrapper(config=cfg)
    obs, _ = wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_self_play_wrapper_saves_training_replay_on_episode_end(tmp_path):
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(cfg.arena, max_ticks=1, map_name="flat"),
    )
    replay_logger = ReplayLogger(replay_dir=str(tmp_path), save_every_n=1)
    wrapper = SelfPlayWrapper(config=cfg, replay_logger=replay_logger)

    wrapper.reset(seed=123)
    _, _, terminated, truncated, _ = wrapper.step(IDLE)

    assert terminated or truncated
    [replay_path] = tmp_path.glob("episode_0001.json")
    replay = load_replay(replay_path)
    assert replay["episode_id"] == 1
    assert replay["winner"] == "draw"
    assert replay["length"] == 1
    assert replay["map_name"] == "flat"
    assert len(replay["frames"]) == 2
    assert replay["frames"][-1]["actions"]["agent_0"] == IDLE
    assert replay["action_counts"]["agent_0"][str(IDLE)] == 1
    assert replay["event_totals"]["agent_0"]["damage_dealt"] == 0


def test_self_play_wrapper_loads_opponent_pool_snapshot_on_reset():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, latest_opponent_prob=1.0),
    )
    pool = OpponentPool(max_size=3)
    pool.add({"action": MOVE_LEFT})
    policy = RecordingPolicy()
    wrapper = SelfPlayWrapper(
        config=cfg,
        opponent_pool=pool,
        opponent_policy=policy,
    )

    _, info = wrapper.reset()

    assert policy.loaded_snapshots == [{"action": MOVE_LEFT}]
    assert policy.training_mode is False
    assert wrapper._opponent_snapshot_loaded is True
    assert info["opponent_snapshot_loaded"] is True
    assert info["opponent_pool"]["last_sample_kind"] == "latest"
    assert info["opponent_pool"]["last_sample_id"] == 0
    assert info["opponent_pool"]["latest_snapshot_id"] == 0
    assert info["opponent_pool"]["latest_samples"] == 1


def test_self_play_wrapper_uses_loaded_snapshot_policy_for_opponent_action():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, latest_opponent_prob=1.0),
    )
    pool = OpponentPool(max_size=3)
    pool.add({"action": MOVE_LEFT})
    policy = RecordingPolicy()
    wrapper = SelfPlayWrapper(
        config=cfg,
        opponent_pool=pool,
        opponent_policy=policy,
    )

    wrapper.reset()
    start_x = wrapper._env._agent_states["agent_1"].x
    wrapper.step(IDLE)

    assert wrapper._env._agent_states["agent_1"].x == start_x - 1


def test_self_play_wrapper_delegates_map_pool_updates():
    wrapper = SelfPlayWrapper(config=Config())
    wrapper.set_map_pool(("flat",))

    _, info = wrapper.reset(seed=123)

    assert info["map_name"] == "flat"
    assert wrapper.get_state()["map_pool"] == ("flat",)


def test_self_play_wrapper_delegates_reward_config_updates():
    wrapper = SelfPlayWrapper(config=Config())
    anti_stall = reward_config_for_preset("anti_stall")

    wrapper.set_reward_config(anti_stall)

    assert wrapper.cfg.reward == anti_stall
    assert wrapper._env.cfg.reward == anti_stall

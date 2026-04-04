import numpy as np
from arena_fighters.config import Config
from arena_fighters.self_play import SelfPlayWrapper, OpponentPool


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


def test_opponent_pool_random_sample():
    pool = OpponentPool(max_size=10)
    for i in range(10):
        pool.add({"weight": i})
    np.random.seed(42)
    samples = [pool.sample(latest_prob=0.0)["weight"] for _ in range(20)]
    assert len(set(samples)) > 1


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

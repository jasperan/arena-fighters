import tempfile
from pathlib import Path

from stable_baselines3 import PPO

from arena_fighters.config import Config
from arena_fighters.network import ArenaFeaturesExtractor
from arena_fighters.self_play import SelfPlayWrapper, OpponentPool


def test_full_training_loop():
    """Train for a few steps and verify checkpoint is saved."""
    cfg = Config()
    opponent_pool = OpponentPool(max_size=5)
    wrapper = SelfPlayWrapper(config=cfg, opponent_pool=opponent_pool)

    policy_kwargs = dict(
        features_extractor_class=ArenaFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "MultiInputPolicy",
        wrapper,
        policy_kwargs=policy_kwargs,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
    )

    model.learn(total_timesteps=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model"
        model.save(str(path))
        loaded = PPO.load(str(path))
        assert loaded is not None

    wrapper.close()


def test_opponent_pool_integration():
    """Verify opponent pool works during training."""
    cfg = Config()
    pool = OpponentPool(max_size=3)
    wrapper = SelfPlayWrapper(config=cfg, opponent_pool=pool)

    policy_kwargs = dict(
        features_extractor_class=ArenaFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "MultiInputPolicy",
        wrapper,
        policy_kwargs=policy_kwargs,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        verbose=0,
    )

    model.learn(total_timesteps=64)

    # Add a snapshot and verify the wrapper can use it
    pool.add(model.policy.state_dict())
    assert len(pool) == 1

    wrapper.opponent_policy = model.policy
    obs, _ = wrapper.reset()
    obs, reward, term, trunc, info = wrapper.step(0)
    assert isinstance(reward, float)

    wrapper.close()

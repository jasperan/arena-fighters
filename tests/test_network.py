import numpy as np
import torch as th
from gymnasium import spaces

from arena_fighters.config import Config, NUM_CHANNELS, NUM_VECTOR_OBS
from arena_fighters.network import ArenaFeaturesExtractor


def _make_obs_space(cfg: Config) -> spaces.Dict:
    h, w = cfg.arena.height, cfg.arena.width
    return spaces.Dict({
        "grid": spaces.Box(low=0, high=1, shape=(NUM_CHANNELS, h, w), dtype=np.float32),
        "vector": spaces.Box(low=0, high=1, shape=(NUM_VECTOR_OBS,), dtype=np.float32),
    })


def test_extractor_output_shape():
    cfg = Config()
    obs_space = _make_obs_space(cfg)
    extractor = ArenaFeaturesExtractor(obs_space, features_dim=128)
    batch = {
        "grid": th.randn(4, NUM_CHANNELS, cfg.arena.height, cfg.arena.width),
        "vector": th.randn(4, NUM_VECTOR_OBS),
    }
    out = extractor(batch)
    assert out.shape == (4, 128)


def test_extractor_features_dim():
    cfg = Config()
    obs_space = _make_obs_space(cfg)
    extractor = ArenaFeaturesExtractor(obs_space, features_dim=128)
    assert extractor.features_dim == 128


def test_extractor_gradients_flow():
    cfg = Config()
    obs_space = _make_obs_space(cfg)
    extractor = ArenaFeaturesExtractor(obs_space, features_dim=128)
    batch = {
        "grid": th.randn(2, NUM_CHANNELS, cfg.arena.height, cfg.arena.width),
        "vector": th.randn(2, NUM_VECTOR_OBS),
    }
    out = extractor(batch)
    loss = out.sum()
    loss.backward()
    for p in extractor.parameters():
        assert p.grad is not None

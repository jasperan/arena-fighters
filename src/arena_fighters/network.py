import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ArenaFeaturesExtractor(BaseFeaturesExtractor):
    """CNN for grid observation + MLP for vector observation."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        grid_space = observation_space["grid"]
        vector_space = observation_space["vector"]
        n_channels = grid_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy forward pass
        with th.no_grad():
            sample = th.as_tensor(grid_space.sample()[None]).float()
            cnn_out_size = self.cnn(sample).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(cnn_out_size, 128),
            nn.ReLU(),
        )

        vector_dim = vector_space.shape[0]
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 32),
            nn.ReLU(),
        )

        # 128 (cnn) + 32 (vector) = 160 -> features_dim
        self.merge = nn.Sequential(
            nn.Linear(160, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        grid_features = self.cnn_linear(self.cnn(observations["grid"]))
        vector_features = self.vector_mlp(observations["vector"])
        merged = th.cat([grid_features, vector_features], dim=1)
        return self.merge(merged)

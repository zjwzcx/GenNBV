from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


class Hybrid_Encoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        encoder_param=None,
        net_param=None,
        visual_input_shape=None,
        state_input_shape=None,
    ):
        assert encoder_param is not None, "Need parameters !"
        assert net_param is not None, "Need parameters !"
        assert isinstance(visual_input_shape, List) or isinstance(visual_input_shape, Tuple), "Use tuple or list"
        assert isinstance(state_input_shape, List) or isinstance(state_input_shape, Tuple), "Use tuple or list"
        self.image_channel = visual_input_shape[0]  # buffer_size
        self.image_shape = visual_input_shape[1:]
        self.state_input_shape = state_input_shape
        feature_dim = net_param["append_hidden_shapes"][-1]
        net_param["append_hidden_shapes"].pop()
        super(Hybrid_Encoder, self).__init__(observation_space, feature_dim)

        self.naive_encoder_grid = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.output_layer_grid = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256, bias=True),   # feature size after 3D CNN, grid_size = 20
            nn.ReLU(inplace=True),
        )

        self.naive_encoder_action = nn.Sequential(
            nn.Linear(in_features=2400, out_features=256, bias=True),   # action_size (positional embedding) * buffer_size
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
        )

    def positional_encoding(self, positions, freqs=2):
        """
        Params:
            positions: [num_env, buffer_size, action_size]

        Return:
            pts: [num_env, buffer_size, 4 * action_size]
        """
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # [2]
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1], ))  # [num_env, buffer_size, 2*action_size]
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [num_env, buffer_size*action_size + 1*X*Y*Z + k*H*W]
        "value = value.permute(0, 4, 1, 2, 3).reshape(value.shape[0], -1)" in wrapper
        """
        num_env = observations.shape[0]

        # action
        action_input = observations[:, :self.state_input_shape[0]]  # [num_env, buffer_size*action_size]
        action_input = action_input.view(num_env, -1, 6)
        action_input = self.positional_encoding(action_input).view(num_env, -1)
        feature_action = self.naive_encoder_action(action_input)    # [num_env, 256]

        # grid
        grid_input = observations[:, self.state_input_shape[0]:self.state_input_shape[0]+8000]     # [num_env, 1*X*Y*Z]
        grid_input = grid_input.reshape(num_env, 1, 20, 20, 20)

        feature_grid = self.naive_encoder_grid(grid_input).reshape(num_env, -1) # [num_env, hidden_layer_size]
        feature_grid = self.output_layer_grid(feature_grid)  # [num_env, 256]

        feature_hybrid = self.output_layer(torch.cat((feature_action, feature_grid), dim=-1))  # [num_env, 256*3] -> [num_env, 256]

        return feature_hybrid

"""A fully-convolutional masked policy for the build env.

The action space factors as ``(building, col, row, rotation)``, and two of those
four dimensions *are* the grid. A conventional policy flattens the feature map and
runs a linear layer to one logit per action - here that is a
``Linear(65536, 45056)``, three billion parameters. Using convolution we can shrink the head
to roughly 128,749.
"""

import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

DEFAULT_DILATIONS = (1, 2, 4, 8)


class ShapezCNN(BaseFeaturesExtractor):
    """Conv trunk that preserves the grid."""

    def __init__(self, observation_space, channels=64, dilations=DEFAULT_DILATIONS):
        n_input, height, width = observation_space.shape
        super().__init__(observation_space, features_dim=channels * height * width)

        self.map_shape = (channels, height, width)

        layers = []
        in_channels = n_input
        for dilation in dilations:
            layers += [
                nn.Conv2d(
                    in_channels, channels, kernel_size=3,
                    padding=dilation, dilation=dilation,
                ),
                nn.ReLU(),
            ]
            in_channels = channels
        self.cnn = nn.Sequential(*layers)

    def forward(self, observations):
        return self.cnn(observations).flatten(1)


class SpatialActionHead(nn.Module):
    """One logit per ``(building, col, row, rotation)``, from a 1x1 conv."""

    def __init__(self, map_shape, n_buildings, n_rotations):
        super().__init__()
        self.map_shape = tuple(map_shape)
        self.n_buildings = n_buildings
        self.n_rotations = n_rotations
        self.conv = nn.Conv2d(map_shape[0], n_buildings * n_rotations, kernel_size=1)

    def forward(self, latent):
        batch = latent.shape[0]
        spatial = latent.view(batch, *self.map_shape)

        logits = self.conv(spatial)
        height, width = logits.shape[-2:]
        logits = logits.view(batch, self.n_buildings, self.n_rotations, height, width)

        # (building, rotation, row, col) -> (building, col, row, rotation)
        return logits.permute(0, 1, 4, 3, 2).reshape(batch, -1)


class PooledValueHead(nn.Module):
    """Global-average-pool the feature map, then one linear."""

    def __init__(self, map_shape):
        super().__init__()
        self.map_shape = tuple(map_shape)
        self.linear = nn.Linear(map_shape[0], 1)

    def forward(self, latent):
        spatial = latent.view(latent.shape[0], *self.map_shape)
        return self.linear(spatial.mean(dim=(2, 3)))


class SpatialMaskablePolicy(MaskableActorCriticPolicy):
    """``MaskableActorCriticPolicy`` with conv heads instead of linear ones."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_buildings,
        n_rotations,
        feature_channels=64,
        dilations=DEFAULT_DILATIONS,
        **kwargs,
    ):
        self.n_buildings = n_buildings
        self.n_rotations = n_rotations
        self.feature_channels = feature_channels
        self.dilations = tuple(dilations)

        _channels, height, width = observation_space.shape
        expected = n_buildings * width * height * n_rotations
        if int(action_space.n) != expected:
            raise ValueError(
                f"action space has {action_space.n} actions but a "
                f"{n_buildings}-building, {width}x{height}, {n_rotations}-rotation "
                f"factorisation needs {expected}; policy_kwargs and the env disagree"
            )

        kwargs["features_extractor_class"] = ShapezCNN
        kwargs["features_extractor_kwargs"] = {
            "channels": feature_channels,
            "dilations": self.dilations,
        }
        # Identity latents: the heads read the feature map directly, so an MLP
        # between trunk and head would only flatten away the geometry they need.
        kwargs["net_arch"] = []

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        """Build the heads."""
        self._build_mlp_extractor()

        map_shape = self.features_extractor.map_shape
        self.action_net = SpatialActionHead(map_shape, self.n_buildings, self.n_rotations)
        self.value_net = PooledValueHead(map_shape)

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            n_buildings=self.n_buildings,
            n_rotations=self.n_rotations,
            feature_channels=self.feature_channels,
            dilations=self.dilations,
        )
        return data


def flat_action_index(n_buildings, width, height, n_rotations, building, col, row, rotation):
    """The flat action index for a placement - the ordering, spelled out once."""
    return int(
        np.ravel_multi_index(
            (building, col, row, rotation), (n_buildings, width, height, n_rotations)
        )
    )

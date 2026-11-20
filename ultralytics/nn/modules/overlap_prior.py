"""Image-predicted overlap priors; forward inference never takes GT labels."""

import math

import torch
import torch.nn as nn


class OverlapPriorHead(nn.Module):
    """Two pointwise layers predict one P3 logit map; no labels enter forward()."""

    def __init__(self, channels, hidden=16, initial_probability=0.05):
        super().__init__()
        if hidden < 1 or not 0 < initial_probability < 1:
            raise ValueError("Invalid prior head width or initial probability")
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.layers[-1].weight, std=1e-3)
        nn.init.constant_(self.layers[-1].bias, math.log(initial_probability / (1 - initial_probability)))

    def forward(self, feature):
        return self.layers(feature)

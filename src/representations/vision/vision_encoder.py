from __future__ import annotations

from ..base import TrafficEncoder
import torch.nn as nn

class VisionTrafficEncoder(TrafficEncoder):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        z = self.net(x)
        return z.view(x.size(0), -1)
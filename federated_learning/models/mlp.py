import torch
from torch import nn


class TabularMLP(nn.Module):
    """MLP for tabular IDS-style inputs."""

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

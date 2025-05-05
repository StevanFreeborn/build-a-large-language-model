"""Feed Forward Module"""

from gelu import GELU
from torch import nn


class FeedForward(nn.Module):
    """Feed Forward Module"""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

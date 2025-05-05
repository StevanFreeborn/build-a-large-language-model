"""An implementation of the Gaussian Error Linear Unit (GELU) activation function."""

import torch
from torch import nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit (GELU) activation function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass for GELU activation."""
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

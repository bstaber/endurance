"""Minimal encoder and decoder blocks with attention and feed-forward layers."""

from typing import Optional

import torch
import torch.nn as nn

from mini_transformers.attention import MultiheadAttention


class Encoder(nn.Module):
    """Transformer encoder block.

    Args:
        input_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimension of the feed-forward layer.
    """

    def __init__(self, input_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attention_layer = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, input_dim)
        )

        self.normalization1 = nn.LayerNorm(input_dim)
        self.normalization2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass of the encoder block."""
        attention_out = self.attention_layer(x, mask)[0]
        x = x + attention_out
        x = self.normalization1(x)

        linear_out = self.linear(x)
        x = x + linear_out
        x = self.normalization2(x)
        return x

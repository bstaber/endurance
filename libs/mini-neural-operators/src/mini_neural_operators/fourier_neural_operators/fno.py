"""Minimal Fourier Neural Operator."""

import torch
import torch.nn as nn

from mini_neural_operators.fourier_neural_operators.spectral_convolutions import (
    SpectralConv1d,
)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dim: Hidden dimension.
        num_modes: Number of Fourier modes to use.
        num_layers: Number of Fourier layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_modes: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.lifting = nn.Linear(in_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, out_dim)
        self.fourier_layers = nn.ModuleList(
            [
                SpectralConv1d(hidden_dim, hidden_dim, num_modes)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the 1D FNO."""
        x = self.lifting(x)
        for layer in self.fourier_layers:
            # Simplified residual connection
            x = layer(x) + x
            x = torch.relu(x)
        x = self.projection(x)
        return x

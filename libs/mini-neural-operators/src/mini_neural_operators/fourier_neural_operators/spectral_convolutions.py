"""Spectral convolutions layers for 1D, 2D and 3D data."""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and inverse FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """Initialize the layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of Fourier modes to keep.
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes, 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the layer parameters."""
        self.weights = self.scale * torch.rand(
            self.in_channels, self.out_channels, self.modes, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch, in_channels, n).

        Returns:
            Output tensor of shape (batch, out_channels, n).
        """
        batch_size = x.shape[0]
        n_grid = x.shape[-1]

        # Apply real FFT
        xf = torch.fft.rfft(x)

        # Create a null vector
        out_convolution = torch.zeros(
            batch_size,
            self.out_channels,
            n_grid // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        # Apply spectral convolution to the first modes
        complex_weights = torch.view_as_complex(self.weights)
        out_convolution[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", xf[:, :, : self.modes], complex_weights
        )

        # Inverse transform
        h = torch.fft.irfft(out_convolution, n=n_grid)
        return h

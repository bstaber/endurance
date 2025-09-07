"""Module that implements the basic RNN cell."""

import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    """A minimal RNN cell implemented from scratch.

    h_t = tanh(Wx x_t + Wh h_{t-1} + b)

    Args:
        input_size: number of input features per time step
        hidden_size: hidden state size
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the values of the RNN parameters."""
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.zeros_(self.bh)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """One reccurent step.

        Args:
            x_t: (B, input_size)
            h_prev: (B, hidden_size)

        Returns:
            h_t: (B, hidden_size)
        """
        return torch.tanh(x_t @ self.Wx.T + h_prev @ self.Wh.T + self.bh)

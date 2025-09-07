"""A minimal GRU layer implemented from scratch using PyTorch."""

import torch
import torch.nn as nn


class GRULayer(nn.Module):
    """A minimal GRU cell implemented from scratch.

    Introduced to mitigate vanishing gradients in RNNs by learning to control how much
    of the past is remember vs updated.

    It adds two gates: update gate (z_t) and reset gate (r_t). GRU blends the hidden state and
    new candidate state using the update gate.

    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    z_t = sigmoid(x_t W_xz^T + h_{t-1} W_hz^T + b_z)
    r_t = sigmoid(x_t W_xr^T + h_{t-1} W_hr^T + b_r)
    n_t = tanh(x_t W_xn^T + r_t ⊙ (h_{t-1} W_hn^T) + b_n)

    Shapes:
      x_t: (B, input_size)
      h_{t-1}: (B, hidden_size)
      h_t: (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combine gates for efficiency: [r, z, n] (order is arbitrary but consistent)
        self.W_x = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

        # Separate biases (bias_ih and bias_hh)
        self.b_x = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_h = nn.Parameter(torch.zeros(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.zeros_(self.b_x)
        nn.init.zeros_(self.b_h)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GRU cell.

        Args:
            x_t: input at time t, shape (B, input_size)
            h_prev: previous hidden state, shape (B, hidden_size)

        Returns:
            h_t: new hidden state, shape (B, hidden_size)
        """
        # x/h projections
        gates_x = x_t @ self.W_x.T + self.b_x  # (B, 3H)
        gates_h = h_prev @ self.W_h.T + self.b_h  # (B, 3H)

        # split into r, z, n parts
        r_x, z_x, n_x = gates_x.chunk(3, dim=-1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=-1)

        r = torch.sigmoid(r_x + r_h)
        z = torch.sigmoid(z_x + z_h)

        n = torch.tanh(n_x + r * n_h)
        h_t = (1.0 - z) * n + z * h_prev
        return h_t

"""A minimal LSTM layer implemented from scratch using PyTorch."""

import torch
import torch.nn as nn


class LSTMLayer(nn.Module):
    """A minimal LSTM cell implemented from scratch.

    The LSTM adds a new vector, the cell state (c_t), to help mitigate vanishing gradients,
    and that represents long-term memory.

    The usual hidden state (h_t) now represents short-term memory.

    At each step, we decide;
        - what to forget from the previous cell state (forget gate, f_t)
        - what new information to add (input gate, i_t and cell candidate, g_t)
        - what to output (output gate, o_t)

    The cell state is updated as:
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

    The hidden state is then:
        h_t = o_t ⊙ tanh(c_t)


    Equations (per time step):
        [i, f, g, o] = x_t @ W_x^T + h_{t-1} @ W_h^T + b_x + b_h
        i_t = sigmoid(i)                      (input gate)
        f_t = sigmoid(f)                      (forget gate)
        g_t = tanh(g)                         (cell candidate)
        o_t = sigmoid(o)                      (output gate)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       (cell state)
        h_t = o_t ⊙ tanh(c_t)                 (hidden state)

    Shapes:
        x_t:    (B, input_size)
        h_prev: (B, hidden_size)
        c_prev: (B, hidden_size)
        h_t:    (B, hidden_size)
        c_t:    (B, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combine all 4 gates for efficiency: [i, f, g, o] along the last dim
        self.W_x = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_h = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))

        # Separate biases like PyTorch (bias_ih and bias_hh)
        self.b_x = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_h = nn.Parameter(torch.zeros(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters following PyTorch's default LSTM initialization."""
        nn.init.xavier_uniform_(self.W_x)
        nn.init.orthogonal_(self.W_h)
        nn.init.zeros_(self.b_x)
        nn.init.zeros_(self.b_h)

        with torch.no_grad():
            H = self.hidden_size
            # gate order: [i, f, g, o]
            self.b_x[H : 2 * H].fill_(1.0)
            self.b_h[H : 2 * H].fill_(1.0)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step.

        Args:
            x_t:    Input at current time step, shape (B, input_size)
            h_prev: Hidden state from previous time step, shape (B, hidden_size)
            c_prev: Cell state from previous time step, shape (B, hidden_size)

        Returns:
            h_t:    Updated hidden state, shape (B, hidden_size)
            c_t:    Updated cell state, shape (B, hidden_size)
        """
        gates = x_t @ self.W_x.T + self.b_x + h_prev @ self.W_h.T + self.b_h  # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)  # cell candidate
        o = torch.sigmoid(o)  # output gate
        c_t = f * c_prev + i * g  # new cell state
        h_t = o * torch.tanh(c_t)  # new hidden state
        return h_t, c_t  # (B, H), (B, H)

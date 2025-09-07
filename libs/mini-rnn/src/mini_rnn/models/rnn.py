"""Recurrent neural network."""

from typing import Optional

import torch
import torch.nn as nn

from mini_rnn.layers.rnn import RNNLayer


class SimpleRNN(nn.Module):
    """Unrolled RNN with an output head."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        layers: list[RNNLayer] = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(RNNLayer(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_all_outputs: bool = True,
    ) -> torch.Tensor:
        """Forward method.

        Args:
            x: Input sequence given as a tensor of shape (B, T, input_size)
            h0: Initial hidden state given as a tensor of shape (B, hidden_size) or None -> zeros
            return_all_outputs: Returns the outputs at all time steps if True

        Returns:
            (Y, h_T):
            Y: (B, T, output_size) if return_all_outputs else (B, output_size)
            h_T: (B, hidden_size)
        """
        B, T, _ = x.shape
        H, out_size = self.hidden_size, self.out.out_features

        if T == 0:
            raise ValueError("SimpleRNN received a sequence with T=0.")

        # Get initial hidden state
        if h0 is None:
            h = x.new_zeros(B, H)
        else:
            if h0.shape != (B, H):
                raise ValueError(
                    f"h0 must have shape (B, H)=({B}, {H}), got {tuple(h0.shape)}"
                )
            if h0.device != x.device or h0.dtype != x.dtype:
                h0 = h0.to(device=x.device, dtype=x.dtype)
            h = h0

        if return_all_outputs:
            Y = x.new_empty(B, T, out_size)
            for t in range(T):
                inp = x[:, t, :]
                for cell in self.layers:
                    h = cell(inp, h)
                    inp = h
                Y[:, t, :] = self.out(inp)
            return Y
        else:
            last_y = x.new_empty(B, out_size)
            for t in range(T):
                inp = x[:, t, :]
                for cell in self.layers:
                    h = cell(inp, h)
                    inp = h
                last_y = self.out(inp)
            return last_y

"""A minimal GRU model implemented from scratch using PyTorch."""

from typing import Optional

import torch
import torch.nn as nn

from mini_rnn.layers.gru import GRULayer


class SimpleGRU(nn.Module):
    """Unrolled GRU with stacked layers and an output head.

    Args:
      input_size: features per time step
      hidden_size: hidden units per layer
      output_size: output features
      num_layers: number of stacked GRU layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers: list[GRULayer] = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(GRULayer(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_all_outputs: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the GRU model.

        Args:
            x: input sequence, shape (B, T, input_size)
            h0: initial hidden states, shape (B, hidden_size)
            return_all_outputs: if True, return outputs for all time steps,
                                else return only the last output.
        """
        B, T, _ = x.shape
        H, out_size = self.hidden_size, self.out.out_features

        if T == 0:
            raise ValueError("SimpleGRU received a sequence with T=0.")

        if h0 is None:
            h = x.new_zeros(B, H)
        else:
            if h0.shape != (B, H):
                raise ValueError(f"h0 must have shape (B, {H}), got {tuple(h0.shape)}")
            if h0.device != x.device or h0.dtype != x.dtype:
                h0 = h0.to(device=x.device, dtype=x.dtype)
            h = h0

        if return_all_outputs:
            Y = x.new_empty(B, T, out_size)
            for t in range(T):
                inp = x[:, t, :]
                for cell in self.layers:
                    h = cell(inp, h)  # (B, H)
                    inp = h
                Y[:, t, :] = self.out(inp)  # (B, O)
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

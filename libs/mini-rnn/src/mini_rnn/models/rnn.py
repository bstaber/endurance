"""Recurrent neural network."""

from typing import Optional

import torch
import torch.nn as nn

from mini_rnn.layers.rnn import RNNLayer


class SimpleRNN(nn.Module):
    """Unrolled RNN with an output head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.cell = RNNLayer(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_all_outputs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method.

        Args:
            x: Tensor of shape (B, T, input_size)
            h0: Tensor of shape (B, hidden_size) or None -> zeros
            return_all_outputs: Returns the outputs at all time steps if True

        Returns:
            (Y, h_T):
            Y: (B, T, output_size) if return_all_outputs else (B, output_size)
            h_T: (B, hidden_size)
        """
        B, T, _ = x.shape
        H = self.cell.hidden_size
        out_size = self.out.out_features

        if T == 0:
            raise ValueError("SimpleRNN received a sequence with T=0.")

        if h0 is None:
            h = x.new_zeros(B, H)
        else:
            if h0.shape != (B, H):
                raise ValueError(f"h0 must have shape (B, H)=({B}, {H}), got {tuple(h0.shape)}")
            if h0.device != x.device or h0.dtype != x.dtype:
                h0 = h0.to(device=x.device, dtype=x.dtype)
            h = h0

        if return_all_outputs:
            Y = x.new_empty(B, T, out_size)
            for t in range(T):
                h = self.cell(x[:, t, :], h)     # (B, H)
                Y[:, t, :] = self.out(h)         # (B, O)
            return Y, h
        else:
            last_y = torch.empty(B, out_size)
            for t in range(T):
                h = self.cell(x[:, t, :], h)
                last_y = self.out(h)
            return last_y, h

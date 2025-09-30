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
            x: (B, T, input_size)
            h0: (B, H) for layer 0 only, or (num_layers, B, H); None -> zeros
            return_all_outputs: if True returns outputs at all time steps

        Returns:
            Y: (B, T, output_size) if return_all_outputs else (B, output_size)
            # NOTE: Your docstring mentioned returning h_T as well, but this
            # function currently returns only Y. If you want h_T, see comment below.
        """
        B, T, _ = x.shape
        H, out_size = self.hidden_size, self.out.out_features

        if T == 0:
            raise ValueError("SimpleRNN received a sequence with T=0.")

        if h0 is None:
            h_list = [x.new_zeros(B, H) for _ in range(self.num_layers)]
        else:
            if h0.dim() == 2 and h0.shape == (B, H):
                # apply to layer 0; others zero
                h_list = [h0.to(device=x.device, dtype=x.dtype)]
                h_list += [x.new_zeros(B, H) for _ in range(self.num_layers - 1)]
            elif h0.dim() == 3 and h0.shape == (self.num_layers, B, H):
                h_list = [
                    h0[l_idx].to(device=x.device, dtype=x.dtype)
                    for l_idx in range(self.num_layers)
                ]
            else:
                raise ValueError(
                    f"h0 must have shape (B, H)=({B}, {H}) or "
                    f"({self.num_layers}, B, H)=({self.num_layers}, {B}, {H}), "
                    f"got {tuple(h0.shape)}"
                )

        if return_all_outputs:
            Y = x.new_empty(B, T, out_size)
            for t in range(T):
                inp = x[:, t, :]
                for l_idx, cell in enumerate(self.layers):
                    h_new = cell(inp, h_list[l_idx])  # use layer-l previous hidden
                    h_list[l_idx] = h_new  # update layer-l hidden for next time step
                    inp = h_new  # feed to next layer
                Y[:, t, :] = self.out(inp)
            return Y
        else:
            last_y = x.new_empty(B, out_size)
            for t in range(T):
                inp = x[:, t, :]
                for l_idx, cell in enumerate(self.layers):
                    h_new = cell(inp, h_list[l_idx])
                    h_list[l_idx] = h_new
                    inp = h_new
                last_y = self.out(inp)
            return last_y

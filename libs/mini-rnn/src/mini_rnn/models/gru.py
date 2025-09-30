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
            h0: initial hidden states, shape (B, H) for layer 0 only,
                or (num_layers, B, H); None -> zeros
            return_all_outputs: if True, return outputs for all time steps,
                                else return only the last output.

        Returns:
            Y: (B, T, output_size) if return_all_outputs else (B, output_size)
            # If you also want final hidden states, see comment at the end.
        """
        B, T, _ = x.shape
        H, out_size = self.hidden_size, self.out.out_features

        if T == 0:
            raise ValueError("SimpleGRU received a sequence with T=0.")

        if h0 is None:
            h_list = [x.new_zeros(B, H) for _ in range(self.num_layers)]
        else:
            if h0.dim() == 2 and h0.shape == (B, H):
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
                    h_list[l_idx] = h_new  # update layer-l hidden
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

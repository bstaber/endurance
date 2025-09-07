"""A minimal LSTM-based RNN model with stacked layers and output head."""

from typing import Optional

import torch
import torch.nn as nn

from mini_rnn.layers.lstm import LSTMLayer


class SimpleLSTM(nn.Module):
    """Unrolled LSTM with stacked layers and an output head.

    Args:
        input_size:  features per time step
        hidden_size: hidden units per layer
        output_size: output features
        num_layers:  number of stacked LSTM layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers: list[LSTMLayer] = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(LSTMLayer(in_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, input_size)
        h0: Optional[torch.Tensor] = None,  # (B, L, H) or None
        c0: Optional[torch.Tensor] = None,  # (B, L, H) or None
        return_all_outputs: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the stacked LSTM layers.

        Args:
            x: input sequence, shape (B, T, input_size)
            h0: initial hidden states for all layers, shape (B, L, H)
            c0: initial cell states for all layers, shape (B, L, H)
            return_all_outputs: if True, return outputs at all time steps; else only last output

        Returns:
            outputs: if return_all_outputs is True, shape (B, T, output_size); else (B, output_size)
            (h_T, c_T): final hidden and cell states for all layers, each of shape (B, L, H)
        """
        B, T, _ = x.shape
        L, H, O = self.num_layers, self.hidden_size, self.out.out_features
        if T == 0:
            raise ValueError("SimpleLSTM received a sequence with T=0.")

        def _prep(state: Optional[torch.Tensor]) -> list[torch.Tensor]:
            if state is None:
                return [x.new_zeros(B, H) for _ in range(L)]
            if state.shape != (B, L, H):
                raise ValueError(
                    f"state must have shape (B, {L}, {H}), got {tuple(state.shape)}"
                )
            if state.device != x.device or state.dtype != x.dtype:
                state = state.to(device=x.device, dtype=x.dtype)
            return [state[:, l, :] for l in range(L)]

        h_list = _prep(h0)
        c_list = _prep(c0)

        if return_all_outputs:
            Y = x.new_empty(B, T, O)
            for t in range(T):
                inp = x[:, t, :]
                for l, cell in enumerate(self.layers):
                    h_new, c_new = cell(inp, h_list[l], c_list[l])
                    h_list[l], c_list[l] = h_new, c_new
                    inp = h_new
                Y[:, t, :] = self.out(inp)
            h_T = torch.stack(h_list, dim=1)  # (B, L, H)
            c_T = torch.stack(c_list, dim=1)  # (B, L, H)
            return Y, (h_T, c_T)
        else:
            last_y = x.new_empty(B, O)
            for t in range(T):
                inp = x[:, t, :]
                for l, cell in enumerate(self.layers):
                    h_new, c_new = cell(inp, h_list[l], c_list[l])
                    h_list[l], c_list[l] = h_new, c_new
                    inp = h_new
                last_y = self.out(inp)
            h_T = torch.stack(h_list, dim=1)
            c_T = torch.stack(c_list, dim=1)
            return last_y, (h_T, c_T)

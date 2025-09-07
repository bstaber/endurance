"""Base reccurent model defined by a protocol."""

from typing import Optional

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    """A reccurent model that processes sequences.

    The model should take input of shape (B, T, input_size) and return output
    of shape (B, T, output_size) or (B, output_size) depending on whether
    return_all_outputs is True or False.

    The model should also return the final hidden state of shape (B, hidden_size).
    """

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_all_outputs: bool = True,
    ) -> torch.Tensor:
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
        ...

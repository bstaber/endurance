import math

import torch.nn as nn


class LoRALayer(nn.Module):
    """Implement a standalone LoRA layer."""

    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        self.B = nn.Linear(in_features, r, bias=False)
        self.A = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))
        nn.init.zeros_(self.A.weight)

    def forward(self, x):
        """Forward pass through the LoRA Linear layer."""
        return self.scaling * self.A(self.B(x))


class LoRALinear(nn.Module):
    """Linear layer with LoRA.."""

    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_layer = LoRALayer(in_features, out_features, r)

    def forward(self, x):
        """Forward pass through the LoRA Linear layer."""
        return self.base(x) + self.lora_layer(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from pinn_lora.linear import LoRALinear


class BaseMLP(nn.Module):
    """Base MLP model used in LoRA experiments."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = layers

    def forward(self, x):
        """Forward pass through the MLP."""
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))
        x = self.layers[-1](x)
        return x


class LoRAMLP(nn.Module):
    """MLP model with LoRA layers."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        base_state_dict,
        r=4,
        alpha=1.0,
    ):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(LoRALinear(input_dim, hidden_dim, r=r, alpha=alpha))
        for _ in range(num_layers - 2):
            layers.append(LoRALinear(hidden_dim, hidden_dim, r=r, alpha=alpha))
        layers.append(LoRALinear(hidden_dim, output_dim, r=r, alpha=alpha))
        self.layers = layers

        with torch.no_grad():
            for name, param in base_state_dict.items():
                if "weight" in name:
                    layer_idx = int(name.split(".")[1])
                    self.layers[layer_idx].base.weight.copy_(param)
                elif "bias" in name:
                    layer_idx = int(name.split(".")[1])
                    self.layers[layer_idx].base.bias.copy_(param)

        for module in self.modules():
            if hasattr(module, "base") and isinstance(module.base, nn.Linear):
                for p in module.base.parameters():
                    p.requires_grad_(False)

    def forward(self, x):
        """Forward pass through the LoRA MLP."""
        for layer in self.layers[:-1]:
            x = F.tanh(layer(x))
        x = self.layers[-1](x)
        return x

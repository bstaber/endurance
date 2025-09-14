"""Burgers' equation PDE residual computation using PyTorch."""

import torch
import torch.nn as nn
from torch.autograd import grad


def burgers_pde(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor, nu: float
) -> torch.Tensor:
    """Compute the Burgers' equation residual."""
    xs = torch.cat([x, t], dim=1)  # (N, 2)
    u = model(xs)  # (N, 1)

    # first derivatives
    u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]

    # second derivative
    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


if __name__ == "__main__":
    """Example usage of the burgers_pde function."""
    import torch.nn as nn

    # Simple feedforward neural network
    model = nn.Sequential(
        nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 1)
    )

    # Sample input points (x, t)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)

    # Viscosity
    nu = 0.01 / torch.pi

    # Compute the PDE residual
    residual = burgers_pde(model, x, t, nu)
    print(residual)

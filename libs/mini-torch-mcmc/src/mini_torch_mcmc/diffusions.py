"""Stochastic Gradient MCMC algorithms."""

import math

import torch
from torch.optim import Optimizer


class SGMCMCBase(Optimizer):
    """Base class for SGMCMC algorithms.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr: float, weight_decay: float = 0.0, **kwargs) -> None:
        if lr <= 0:
            raise ValueError("step size (lr) must be > 0")
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            generator=kwargs.pop("generator", None),
            **kwargs,
        )
        super().__init__(params, defaults)

    def init_state(self, p: torch.Tensor, state: dict, group: dict):
        """Initialize state for parameter p."""
        raise NotImplementedError

    def update_fn(self, p: torch.Tensor, d_p: torch.Tensor, state: dict, group: dict):
        """Update function for parameter p."""
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single MCMC step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["iteration"] = 0
                    self.init_state(p, state, group)
                state["iteration"] += 1

                d_p = p.grad

                # L2 prior inside gradient
                if wd != 0.0:
                    d_p = d_p.add(p, alpha=wd)

                # delegate to kernel update
                self.update_fn(p, d_p, state, group)

        return loss


class SGLD(SGMCMCBase):
    """Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        temperature (float, optional): temperature (default: 1.0)
            (higher temperature -> more noise)
        generator (torch.Generator, optional): random number generator for noise
            (default: None)
    """

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        *,
        temperature: float = 1.0,
        generator=None,
    ):
        super().__init__(
            params=params, lr=lr, weight_decay=weight_decay, generator=generator
        )
        self.temperature = float(temperature)

    def init_state(self, p, state, group):
        """Initialize state for parameter p.

        No state to initialize for SGLD.
        """
        pass

    def update_fn(self, p, d_p, _state, group):
        """Update function for parameter p.

        Args:
            p (torch.Tensor): parameter to update
            d_p (torch.Tensor): gradient of the parameter
            state (dict): state of the parameter
            group (dict): parameter group
        """
        lr = group["lr"]
        # drift
        p.add_(d_p, alpha=-0.5 * lr)
        # diffusion
        noise_std = math.sqrt(lr * self.temperature)
        gen = group.get("generator", None)
        p.add_(torch.randn_like(p, generator=gen), alpha=noise_std)

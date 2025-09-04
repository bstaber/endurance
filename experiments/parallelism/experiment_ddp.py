"""Example of usage of torchrun with DDP."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    """Example that uses DDP to perform data parallelism during training."""
    backend = (
        "nccl"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else "gloo"
    )
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    _world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model = nn.Linear(
        10,
        1,
    ).to(device)

    ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    _ = torch.manual_seed(1234 + rank)
    for step in range(5):
        x = torch.randn(32, 10, device=device)
        target = torch.randn(32, 1, device=device)

        opt.zero_grad(set_to_none=True)
        y = ddp_model(x)
        loss = loss_fn(y, target)
        loss.backward()
        opt.step()

        if rank == 0:
            print(f"[DDP](step {step}): loss = {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

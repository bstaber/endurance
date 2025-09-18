"""Minimal example of using FSDP for distributed training."""

import os
import time

import torch
import torch.distributed as dist
from common import SimpleCNN, get_loader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def main():
    """Train the SimpleCNN model on MNIST using FSDP for distributed training."""
    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    # Choose device based on local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize model, optimizer, and data loader
    model = SimpleCNN().to(device)
    model = FSDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    loader = get_loader(batch_size=32, workers=2)

    # Run a few epochs
    for epoch in range(10):
        model.train()
        start_time = time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            opt.zero_grad()
            preds = model(x)
            loss = torch.nn.functional.cross_entropy(preds, y)
            loss.backward()
            opt.step()
        end_time = time.time()
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s"
        )

    # Destroy process group at the end
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

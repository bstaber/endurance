"""Example of usage of DTensor (sharding + gather)."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def main():
    """Example that creates a mesh of devices, distribute a tensor, and perform some basic operations."""
    backend = (
        "nccl"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1
        else "gloo"
    )
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device_type = "cpu"
        device = torch.device("cpu")

    # Create a 1D mesh of size = world_size
    mesh = init_device_mesh(device_type, (world_size,))

    # Global tensor (same on each rank just for demo)
    global_t = torch.arange(8, dtype=torch.float32, device=device)

    # Shard along dim 0 accross the mesh
    dt = distribute_tensor(global_t, mesh, placements=[Shard(0)])

    # Show each rank's local shard
    local = dt.to_local()
    print(f"[DTensor][rank {rank}] local shard: {local.tolist()}")

    # Do some computation
    dt2 = dt * 10 + 1

    # Gather/replicate
    dt_full = dt2.redistribute(mesh, placements=[Replicate()])
    full_local = dt_full.to_local()

    dist.destroy_process_group()

    if rank == 0:
        print(f"[DTensor][rank 0] full after compute: {full_local.tolist()}")


if __name__ == "__main__":
    main()

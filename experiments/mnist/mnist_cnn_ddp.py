"""Minimal example of training a CNN on MNIST using PyTorch DDP."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """A simple CNN for MNIST classification."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # 28x28 -> 28x28
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after 2x2 max pooling twice
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        """Forward pass of the CNN."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def init_ddp():
    """Initialize DDP environment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def close_ddp():
    """Clean up DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, world_size, local_rank = init_ddp()

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        sampler=test_sampler,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )

    # model/opt
    model = SimpleCNN().to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # only rank 0 prints/saves
    is_main = rank == 0

    for epoch in range(1, 100 + 1):
        train_sampler.set_epoch(epoch)  # reshuffle each epoch
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            # local stats
            b = y.size(0)
            running_loss += loss.item() * b
            running_acc += (torch.argmax(logits, dim=1) == y).sum().item()
            n += b

        # reduce train stats across processes
        t_loss = torch.tensor([running_loss], device=device)
        t_acc = torch.tensor([running_acc], device=device)
        t_n = torch.tensor([n], device=device, dtype=torch.float32)

        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_n, op=dist.ReduceOp.SUM)
        train_loss_epoch = (t_loss / t_n).item()
        train_acc_epoch = (t_acc / t_n).item()

        # eval (distributed)
        model.eval()
        e_loss, e_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                b = y.size(0)
                e_loss += loss.item() * b
                e_acc += (torch.argmax(logits, dim=1) == y).sum().item()
                m += b
        e_loss_t = torch.tensor([e_loss], device=device)
        e_acc_t = torch.tensor([e_acc], device=device)
        m_t = torch.tensor([m], device=device, dtype=torch.float32)
        dist.all_reduce(e_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(e_acc_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(m_t, op=dist.ReduceOp.SUM)
        val_loss = (e_loss_t / m_t).item()
        val_acc = (e_acc_t / m_t).item()

        if is_main:
            print(
                f"[Epoch {epoch}] train_loss={train_loss_epoch:.4f} "
                f"train_acc={train_acc_epoch:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )

    close_ddp()

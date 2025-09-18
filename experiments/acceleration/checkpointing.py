"""Minimal example of using checkpointing for memory efficiency."""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from common import get_loader


class NetCP(nn.Module):
    """A simple CNN for MNIST classification with checkpointing."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1), nn.ReLU()
        )
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass with checkpointing."""
        x = cp.checkpoint(self.block, x, use_reentrant=False)
        x = F.adaptive_avg_pool2d(x, (12, 12))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    """Train the SimpleCNN model on MNIST using checkpointing for memory efficiency."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and data loader
    model = NetCP().to(device)
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
            loss = F.cross_entropy(preds, y)
            loss.backward()
            opt.step()
        end_time = time.time()
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s"
        )


if __name__ == "__main__":
    main()

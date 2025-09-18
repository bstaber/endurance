"""Baseline training script using FP32 precision."""

import time

import torch
from common import SimpleCNN, get_loader


def main():
    """Train the SimpleCNN model on MNIST using FP32 precision."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and data loader
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = get_loader(batch_size=32, workers=2)

    # Run a few epochs
    for epoch in range(10):
        model.train()
        # time the epoch
        start_time = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            preds = model(x)
            loss = torch.nn.functional.cross_entropy(preds, y)
            loss.backward()
            opt.step()
        end_time = time.time()
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s"
        )


if __name__ == "__main__":
    main()

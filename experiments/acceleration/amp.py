"""Minimal example of using AMP for mixed precision training."""

import time

import torch
import torch.nn.functional as F
from common import SimpleCNN, get_loader


def main():
    """Train the SimpleCNN model on MNIST using AMP for mixed precision."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and data loader
    model = SimpleCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    scaler = torch.GradScaler(device.type)
    loader = get_loader(batch_size=32, workers=2)

    # Run a few epochs
    for epoch in range(10):
        model.train()
        start_time = time.time()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device == "cuda" else torch.bfloat16,
            ):
                preds = model(x)
                loss = F.cross_entropy(preds, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        end_time = time.time()
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s"
        )


if __name__ == "__main__":
    main()

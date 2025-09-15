"""Minimal MNIST example with CNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


if __name__ == "__main__":
    """Train the CNN on MNIST."""

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms and data loaders
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
    )

    # Initialize model, optimizer, and loss function
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        train_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                correct += (torch.argmax(logits, dim=1) == y).sum().item()

        train_loss /= len(train_loader)  # loss over all batches
        correct /= len(train_loader.dataset)  # accuracy over all samples

        model.eval()
        test_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)

                b = y.size(0)
                test_loss_sum += loss.item() * b
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += b

        test_loss = test_loss_sum / total
        test_acc = correct / total

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={test_loss:.4f} val_acc={test_acc:.4f}"
        )

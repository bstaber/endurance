import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """A simple CNN for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_loader(batch_size: int, workers: int):
    """Get a DataLoader for the MNIST dataset."""
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    return loader


if __name__ == "__main__":
    # Example usage
    model = SimpleCNN()
    loader = get_loader(batch_size=64, workers=2)
    for images, labels in loader:
        outputs = model(images)
        print(outputs.shape)  # Should be [64, 10]
        break

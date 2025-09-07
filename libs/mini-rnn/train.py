"""Trainer entrpy point for recurrent models."""

import logging

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from mini_rnn.models.base import RNNModel
from omegaconf import DictConfig
from torch.utils.data import DataLoader

logger = logging.getLogger("RNN Trainer")


class RandomSequenceDataset(torch.utils.data.IterableDataset):
    """Generates random (x, y) pairs where y is a delayed copy of x.

    Purely for smoke-testing the training loop.
    """

    def __init__(
        self, input_size: int, seq_len: int, num_batches: int, batch_size: int
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        """Y is X delayed by one time step, with zeros at the end."""
        for _ in range(self.num_batches):
            x = torch.randn(self.batch_size, self.seq_len, self.input_size)
            y = torch.roll(x, shifts=-1, dims=1)
            y[:, -1, :] = 0.0
            yield x, y


def _make_loader(cfg: DictConfig) -> torch.utils.data.DataLoader:
    ds = RandomSequenceDataset(
        input_size=cfg.data.input_size,
        seq_len=cfg.data.seq_len,
        num_batches=cfg.data.num_batches,
        batch_size=cfg.data.batch_size,
    )
    return torch.utils.data.DataLoader(ds, batch_size=None)


def _train_one_epoch(
    model: RNNModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()  # pyright: ignore[reportUnknownMemberType]
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x, return_all_outputs=True)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    """Entry point that trains a recurrent model."""
    device = torch.device(cfg.trainer.device)

    model: RNNModel = instantiate(cfg.model).to(device)
    criterion = instantiate(cfg.loss)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Data
    loader = _make_loader(cfg)

    # Train
    for epoch in range(1, cfg.trainer.epochs + 1):
        loss = _train_one_epoch(model, loader, criterion, optimizer, device)
        logger.info(f"[epoch {epoch:03d}] loss={loss:.4f}")

    # Optionally save
    if cfg.trainer.save_path:
        torch.save(model.state_dict(), cfg.trainer.save_path)
        logger.info(f"Saved weights to: {cfg.trainer.save_path}")


if __name__ == "__main__":
    run()

"""Darcy example taken from physics-nemo (without distributed training)."""

import torch
import torch.nn as nn
import torch.optim as optim
from physicsnemo.datapipes.benchmarks.darcy import Darcy2D
from physicsnemo.models.fno import FNO


def main():
    """Main function to run the Darcy example."""
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and training setup
    model = FNO(
        in_channels=1,
        out_channels=1,
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=9,
        decoder_layer_size=32,
        decoder_layers=1,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.85**step
    )

    # Dataloader
    # This dataloader generates data on the fly.
    # It relies on NVIDIA's warp library to solve the Darcy equation.
    normaliser = {"permeability": (1.25, 0.75), "darcy": (4.52e-2, 2.79e-2)}
    dataloader = Darcy2D(normaliser=normaliser, batch_size=4, resolution=16)

    # Dummy training loop
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        # Dummy loop over 10 batches
        # The dataloader is a generator, so we use zip to limit the number of batches
        for _, batch in zip(range(10), dataloader):
            # Standard training step
            # In the origin example, they use the StaticCaptureTraining decorator instead of implement this manually
            optimizer.zero_grad()
            x_batch = batch["permeability"]
            y_batch = model(x_batch)
            loss = loss_fn(y_batch, batch["darcy"])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= 10

        # In the original example, they also use the taticCaptureEvaluateNoGrad decorator instead of this manual implementation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for _, batch in zip(range(10), dataloader):
                x_batch = batch["permeability"]
                y_batch = model(x_batch)
                loss = loss_fn(y_batch, batch["darcy"])
                val_loss += loss.item()
            val_loss /= 10
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:12.4f}, Val Loss: {val_loss:12.4f}"
        )


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("Darcy example")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()

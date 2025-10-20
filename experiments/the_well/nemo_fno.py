"""Experiment script for training FNO on the turbulent radiative layer 2D case taken from The Well."""

import logging

import einops
import torch
import torch.optim as optim
from physicsnemo.models.fno import FNO
from the_well.data import WellDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def main():
    """Main function to set up dataset, dataloader, model, and train."""
    base_path = "/mnt/c/Users/brian/Downloads/the_well/datasets"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "turbulent_radiative_layer_2D"
    dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=name,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=3,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = FNO(
        in_channels=4,
        out_channels=4,
        decoder_layers=1,
        decoder_layer_size=32,
        decoder_activation_fn="silu",
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=17,
        padding=8,
        padding_type="constant",
        activation_fn="gelu",
        coord_features=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.GradScaler(device=device.type)

    for epoch in range(10):
        loss_train = 0.0
        for batch in dataloader:
            t_max = batch["output_fields"].shape[1]
            assert t_max > 1, (
                "t_max must be greater than 1 for auto-regressive training"
            )

            batch["input_fields"] = batch["input_fields"].to(device)
            y: torch.Tensor = batch["output_fields"]
            y = y.to(device)

            y_preds: list[torch.Tensor] = []
            model.train()
            optimizer.zero_grad()
            for t in range(t_max):
                # process batch
                x: torch.Tensor = einops.rearrange(
                    batch["input_fields"], "b t h w c -> b (t c) h w"
                )
                x = x.to(device)

                # forward pass
                y_pred = model(x)

                # rearrange y_pred to have shape (b, 1, h, w, c)
                y_pred = einops.rearrange(y_pred, "b c ... -> b 1 ... c")
                y_preds.append(y_pred)

                # append prediction to input for next step and remove oldest input
                if t < t_max - 1:
                    batch["input_fields"] = torch.cat(
                        (batch["input_fields"][:, 1:], y_pred), dim=1
                    )
            y_preds_concat: torch.Tensor = torch.cat(y_preds, dim=1)
            loss = torch.nn.functional.mse_loss(y_preds_concat, y)

            # backward pass with gradient scaling
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
            scaler.step(optimizer)
            scaler.update()

            loss_train += loss.item()
        loss_train /= len(dataloader)

        logger.info(f"Epoch {epoch + 1}, Loss: {loss_train}")


if __name__ == "__main__":
    main()

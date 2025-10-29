"""Train a Fourier Neural Operator (FNO) model on the "The Well" dataset."""

import logging

import torch
from einops import rearrange
from neuralop.models.fno import FNO2d
from the_well.data import WellDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main function to set up dataset, dataloader, model, and train."""
    base_path = "/mnt/c/Users/brian/Downloads/the_well/datasets"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = "turbulent_radiative_layer_2D"

    dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name=name,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
    )
    num_fields: int = dataset.metadata.n_fields
    print(f"Number of fields: {num_fields}")
    print(f"Train dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4
    )

    model = FNO2d(
        n_modes_height=16,
        n_modes_width=16,
        hidden_channels=128,
        in_channels=4 * num_fields,
        out_channels=1 * num_fields,
    ).to(device)
    print(
        f"Model has {count_parameters(model) / 1e6:.2f} million trainable parameters."
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Perform 10 steps of training
    for iteration in range(10):
        batch = next(iter(dataloader))
        # Shape: (batch_size, T_in, H, W, F)
        input_fields = batch["input_fields"].to(device)
        input_fields = rearrange(input_fields, "b t_in h w f -> b (t_in f) h w")
        # Shape: (batch_size, T_out, H, W, F)
        output_fields = batch["output_fields"].to(device)
        output_fields = rearrange(output_fields, "b t_out h w f -> b (t_out f) h w")

        preds = model(input_fields)
        loss = torch.nn.functional.mse_loss(preds, output_fields)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"Iteration {iteration} | Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

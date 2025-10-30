"""Script to read the Car CFD Dataset."""

import logging

logging.basicConfig(level=logging.INFO)

import mlflow
import torch
import torch.nn.functional as F
from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset
from neuralop.models import GINO

logger = logging.getLogger(__name__)


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: The model to count parameters for.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_batch(batch, device) -> tuple[dict, torch.Tensor]:
    """Process batch to move to device and set input/output keys.

    Args:
        batch: Batch from DataLoader.
        device: Device to move data to.

    Returns:
        input_dict: Dictionary of inputs for the model.
        truth: Ground truth tensor.
    """
    # Move data to device and prepare input/output
    in_p = batch["vertices"].squeeze(0).to(device)
    latent_queries = batch["query_points"].squeeze(0).to(device)
    out_p = batch["vertices"].squeeze(0).to(device)
    f = batch["distance"].to(device)
    truth = batch["press"].squeeze(0).unsqueeze(-1)

    # Adjust output size if necessary
    output_vertices = truth.shape[1]
    if out_p.shape[0] > output_vertices:
        out_p = out_p[:output_vertices, :]
    truth = truth.to(device)

    # Prepare input dictionary
    input_dict = dict(
        input_geom=in_p,
        latent_queries=latent_queries,
        latent_features=f,
        output_queries=out_p,
        x=None,
    )

    return input_dict, truth


def main():
    """Main entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CarCFDDataset(
        root_dir="/mnt/c/Users/brian/cache/car_cfd_dataset/processed-car-pressure-data",
        n_train=-1,
        n_test=-1,
        download=False,
    )

    train_loader = dataset.train_loader(
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = dataset.test_loader(
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    model = GINO(
        in_channels=3,
        out_channels=1,
        latent_feature_channels=1,
    ).to(device)

    logger.info(f"Model has {count_parameters(model)/1e6:.2f} million trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    num_epochs = 10
    mlflow.set_experiment("GINO-CarCFDDataset")
    with mlflow.start_run():
        for epoch in range(num_epochs):
            loss_train = 0.0
            model.train()
            for batch in train_loader:
                input_dict, output = process_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=True):
                    pred = model(**input_dict)
                    loss = F.mse_loss(pred, output)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train /= len(train_loader)
            mlflow.log_metric("train_loss", value=loss_train, step=epoch)
            scheduler.step(loss_train)

            model.eval()
            with torch.no_grad():
                loss_test = 0.0
                for batch in test_loader:
                    input_dict, output = process_batch(batch, device)
                    pred = model(**input_dict)
                    loss = F.mse_loss(pred, output)
                    loss_test += loss.item()
                loss_test /= len(test_loader)
                mlflow.log_metric("test_loss", value=loss_test, step=epoch)

    import matplotlib.pyplot as plt

    vertices = batch["vertices"].squeeze().cpu().numpy()
    press_true = batch["press"].squeeze().cpu().numpy()
    press_pred = pred.detach().cpu().numpy().squeeze()

    # --- translation offset ---
    offset = 1.5  # how far apart to place them

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # True (left)
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2] * 2,
        s=2,
        c=press_true,
        cmap="jet",
        label="Ground truth",
    )

    # Prediction (right) — translated in X
    ax.scatter(
        vertices[:, 0] + offset,
        vertices[:, 1],
        vertices[:, 2] * 2,
        s=2,
        c=press_pred,
        cmap="jet",
        alpha=0.5,
        label="Prediction",
    )

    ax.set_xlim(0, 2 + offset)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.view_init(elev=20, azim=150, roll=0, vertical_axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig("car_cfd_prediction_side_by_side.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

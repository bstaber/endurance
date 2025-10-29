"""Train a Fourier Neural Operator (FNO) model on the "The Well" dataset."""

import logging
from pathlib import Path

import hydra
import mlflow
import torch
from einops import rearrange
from lr_scheduler import LinearWarmupCosineAnnealingLR
from neuralop.models.fno import FNO2d
from omegaconf import DictConfig
from the_well.data import WellDataModule
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

log = logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to set up dataset, dataloader, model, and train."""
    outdir = Path(".").resolve()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # --- data ---
    n_steps_input = 1
    datamodule = WellDataModule(
        well_base_path=cfg.dataloader.base_path,
        well_dataset_name=cfg.dataloader.name,
        n_steps_input=n_steps_input,
        n_steps_output=1,
        use_normalization=True,
        batch_size=cfg.dataloader.batch_size,
        data_workers=cfg.dataloader.num_workers,
    )

    num_fields: int = datamodule.train_dataset.metadata.n_fields
    log.info(f"Number of fields: {num_fields}")

    # --- model ---
    model = FNO2d(
        n_modes_height=cfg.model.n_modes_height,
        n_modes_width=cfg.model.n_modes_width,
        hidden_channels=cfg.model.hidden_channels,
        in_channels=n_steps_input * num_fields,
        out_channels=1 * num_fields,
    ).to(device)
    # if cfg.trainer.channels_last:
    #     model = model.to(memory_format=torch.channels_last)
    log.info(
        f"Model has {count_parameters(model) / 1e6:.2f} million trainable parameters."
    )

    if cfg.trainer.compile:
        try:
            compile_kwargs = dict(
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )

            model = torch.compile(model, **compile_kwargs)
        except Exception as e:
            log.info(f"torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.trainer.lr, weight_decay=1e-4
    )

    # --- amp ---
    use_amp = cfg.trainer.amp and use_cuda
    amp_dtype = torch.float16  # bfloat16 is unsupported by torch.fft
    scaler = torch.GradScaler(enabled=use_amp)

    # --- training step ---
    def train_step(batch):
        """Single training step."""
        non_blocking = True if use_cuda else False
        # Shape: (batch_size, T_in, H, W, F)
        input_fields = batch["input_fields"].to(device, non_blocking=non_blocking)
        input_fields = rearrange(input_fields, "b t_in h w f -> b (t_in f) h w")[
            :, :, :, :256
        ]  # crop to 256 width
        # Shape: (batch_size, T_out, H, W, F)
        output_fields = batch["output_fields"].to(device, non_blocking=non_blocking)
        output_fields = rearrange(output_fields, "b t_out h w f -> b (t_out f) h w")[
            :, :, :, :256
        ]  # crop to 256 width

        input_fields = torch.nan_to_num(input_fields)
        output_fields = torch.nan_to_num(output_fields)
        # if cfg.trainer.channels_last:
        #     input_fields = input_fields.contiguous(memory_format=torch.channels_last)
        #     output_fields = output_fields.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            enabled=use_amp, dtype=amp_dtype, device_type="cuda" if use_cuda else "cpu"
        ):
            pred = model(input_fields)
            loss = torch.nn.functional.mse_loss(pred, output_fields)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return loss.item()

    def val_step(batch):
        """Validation step."""
        non_blocking = True if use_cuda else False
        # Shape: (batch_size, T_in, H, W, F)
        input_fields = batch["input_fields"].to(device, non_blocking=non_blocking)
        input_fields = rearrange(input_fields, "b t_in h w f -> b (t_in f) h w")[
            :, :, :, :256
        ]  # crop to 256 width
        # Shape: (batch_size, T_out, H, W, F)
        output_fields = batch["output_fields"].to(device, non_blocking=non_blocking)
        output_fields = rearrange(output_fields, "b t_out h w f -> b (t_out f) h w")[
            :, :, :, :256
        ]  # crop to 256 width

        input_fields = torch.nan_to_num(input_fields)
        output_fields = torch.nan_to_num(output_fields)

        with torch.autocast(
            enabled=use_amp, dtype=amp_dtype, device_type="cuda" if use_cuda else "cpu"
        ):
            pred = model(input_fields)
            loss = torch.nn.functional.mse_loss(pred, output_fields)
        return loss.item()

    if cfg.trainer.profile:
        tb_dir = outdir / "tb_logs"
        tb_dir.mkdir(exist_ok=True, parents=True)
        sched = schedule(wait=2, warmup=2, active=6)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if use_cuda else [])
        with profile(
            activities=acts,
            schedule=sched,
            with_modules=True,
            on_trace_ready=tensorboard_trace_handler(str(tb_dir)),
        ) as prof:
            it = iter(datamodule.train_dataloader())
            for i in range(cfg.trainer.num_epochs):
                batch = next(it)
                loss = train_step(batch)
                if i % 10 == 0:
                    log.info(f"{i}: {loss:.6f}")
                prof.step()
            torch.cuda.synchronize() if use_cuda else None
    else:
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "amp": cfg.trainer.amp,
                    "batch_size": cfg.dataloader.batch_size,
                    "hidden_channels": cfg.model.hidden_channels,
                }
            )
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=3,
                max_epochs=cfg.trainer.num_epochs,
                warmup_start_lr=cfg.trainer.lr * 0.1,
                eta_min=1e-6,
            )

            log.info("Starting training...")
            for epoch in range(cfg.trainer.num_epochs):
                model.train()
                train_epoch_loss = 0.0
                for batch in datamodule.train_dataloader():
                    loss = train_step(batch)
                    train_epoch_loss += loss
                train_epoch_loss /= len(datamodule.train_dataloader())
                log.info(
                    f"Epoch {epoch + 1}/{cfg.trainer.num_epochs}, Train Loss: {train_epoch_loss:.6f}"
                )
                scheduler.step()
                mlflow.log_metric("train_loss", train_epoch_loss, step=epoch)

                model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    for batch in datamodule.val_dataloader():
                        batch["input_fields"] = (
                            datamodule.train_dataset.norm.normalize_flattened(
                                batch["input_fields"], "variable"
                            )
                        )
                        batch["output_fields"] = (
                            datamodule.train_dataset.norm.normalize_flattened(
                                batch["output_fields"], "variable"
                            )
                        )
                        loss = val_step(batch)
                        val_epoch_loss += loss
                val_epoch_loss /= len(datamodule.val_dataloader())
                log.info(
                    f"Epoch {epoch + 1}/{cfg.trainer.num_epochs}, Val Loss: {val_epoch_loss:.6f}"
                )
                mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)


if __name__ == "__main__":
    main()

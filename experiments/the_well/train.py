"""Train a Fourier Neural Operator (FNO) model on the "The Well" dataset."""

import logging
from pathlib import Path
from time import perf_counter, time

import hydra
import torch
import torch.nn as nn
from einops import rearrange
from neuralop.models.fno import FNO2d
from omegaconf import DictConfig
from the_well.data import WellDataset
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

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
    dataset = WellDataset(
        well_base_path=cfg.data.base_path,
        well_dataset_name=cfg.data.name,
        well_split_name="train",
        n_steps_input=4,
        n_steps_output=1,
    )
    num_fields: int = dataset.metadata.n_fields
    log.info(f"Number of fields: {num_fields}")
    log.info(f"Train dataset length: {len(dataset)}")

    xs = []
    for i in range(0, 1000, 100):
        x = dataset[i]["input_fields"]
        xs.append(x)
    xs = torch.stack(xs)
    mu = xs.reshape(-1, num_fields).mean(dim=0).to(device)
    sigma = xs.reshape(-1, num_fields).std(dim=0).to(device)

    def _preprocess(x):
        return (x - mu) / sigma

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=True,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    # --- model ---
    model = FNO2d(
        n_modes_height=cfg.model.n_modes_height,
        n_modes_width=cfg.model.n_modes_width,
        hidden_channels=cfg.model.hidden_channels,
        in_channels=4 * dataset.metadata.n_fields,
        out_channels=1 * dataset.metadata.n_fields,
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
                mode="reduce-overhead",  # fewer autotune sweeps than max-autotune
                fullgraph=False,  # allow graph breaks (critical here)
                dynamic=True,
            )  # handle symbolic shapes better

            model = torch.compile(model, **compile_kwargs)
        except Exception as e:
            log.info(f"torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.lr, weight_decay=1e-4)

    # --- amp ---
    use_amp = cfg.trainer.amp and use_cuda
    amp_dtype = torch.float16  # bfloat16 is unsupported by torch.fft
    scaler = torch.GradScaler(enabled=use_amp)

    # --- training step ---
    def train_step(batch):
        non_blocking = True if use_cuda else False
        # Shape: (batch_size, T_in, H, W, F)
        input_fields = batch["input_fields"].to(device, non_blocking=non_blocking)
        input_fields = _preprocess(input_fields)
        input_fields = rearrange(input_fields, "b t_in h w f -> b (t_in f) h w")[
            :, :, :, :256
        ]  # crop to 256 width
        # Shape: (batch_size, T_out, H, W, F)
        output_fields = batch["output_fields"].to(device, non_blocking=non_blocking)
        output_fields = _preprocess(output_fields)
        output_fields = rearrange(output_fields, "b t_out h w f -> b (t_out f) h w")[
            :, :, :, :256
        ]  # crop to 256 width

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
            it = iter(dataloader)
            for i in range(cfg.trainer.steps):
                batch = next(it)
                loss = train_step(batch)
                if i % 10 == 0:
                    log.info(f"{i}: {loss:.6f}")
                prof.step()
            torch.cuda.synchronize() if use_cuda else None
    else:
        log.info("Starting training...")
        it = iter(dataloader)
        t0 = time()
        for i in range(cfg.trainer.steps):
            batch = next(it)
            loss = train_step(batch)
            if i % 10 == 0:
                log.info(f"{i}: {loss:.6f}")

        if use_cuda:
            torch.cuda.synchronize()

        t1 = time()
        log.info(f"Trained {cfg.trainer.steps} steps in {t1 - t0:.2f} seconds.")


if __name__ == "__main__":
    main()

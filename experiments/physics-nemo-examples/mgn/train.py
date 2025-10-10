"""Example of training MGN on 2DMultiscaleHyperelasticity dataset."""

import torch
import torch._inductor.config as cfg
from datasets import load_dataset
from physicsnemo.models.meshgraphnet import MeshGraphNet
from plaid import Dataset
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
    huggingface_description_to_problem_definition,
)
from plaid.types import FeatureIdentifier
from plaid_bridges.torch import PyGBridge
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian


def get_datasets(train_split: str, test_split: str) -> tuple[Dataset, Dataset]:
    """Load 2DMultiscaleHyperelasticity dataset from Hugging Face and convert to PLAID datasets."""
    hf_dataset = load_dataset(
        "PLAID-datasets/2DMultiscaleHyperelasticity", split="all_samples"
    )
    problem_definition = huggingface_description_to_problem_definition(
        hf_dataset.info.description
    )

    # Convert to PLAID dataset
    ids_train = problem_definition.get_split(train_split)
    ids_test = problem_definition.get_split(test_split)
    train_dataset, _ = huggingface_dataset_to_plaid(
        hf_dataset, ids=ids_train, processes_number=5
    )
    test_dataset, _ = huggingface_dataset_to_plaid(
        hf_dataset, ids=ids_test, processes_number=5
    )

    return train_dataset, test_dataset


def process_for_pyg(dataset: Dataset, test: bool) -> list[Data]:
    """Process PLAID dataset to PyG Data list suitable for MeshGraphNet."""
    feature_ids = [
        FeatureIdentifier(name="C11", type="scalar"),
        FeatureIdentifier(name="C22", type="scalar"),
        FeatureIdentifier(name="C12", type="scalar"),
    ]
    if not test:
        feature_ids.append(FeatureIdentifier(name="effective_energy", type="scalar"))
        feature_ids.append(FeatureIdentifier(name="u1", type="field"))
        feature_ids.append(FeatureIdentifier(name="u2", type="field"))

    bridge = PyGBridge()
    pyg_data_list = bridge.transform(dataset, features_ids=feature_ids)

    # We need edge attributes for MeshGraphNet, let's use Cartesian distances
    transform = Cartesian()
    pyg_data_list = [transform(data) for data in pyg_data_list]

    # convert data.x, edge_attr to float32
    for data in pyg_data_list:
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
    return pyg_data_list


if __name__ == "__main__":
    """Train MeshGraphNet on 2DMultiscaleHyperelasticity dataset with profiling.

    We use torch.profiler to profile the training loop and save the results to TensorBoard logs.

    To visualize the profiling results, run the following command in the terminal:
        tensorboard --logdir=./tb_logs

    Then open the provided URL in a web browser.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    train_dataset, test_dataset = get_datasets("DOE_train", "DOE_test")
    train_pyg = process_for_pyg(train_dataset, test=False)
    test_pyg = process_for_pyg(test_dataset, test=True)

    train_loader = DataLoader(
        train_pyg,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_pyg, batch_size=1, shuffle=False, pin_memory=True, num_workers=1
    )

    model = MeshGraphNet(
        input_dim_nodes=2,
        input_dim_edges=2,
        output_dim=2,
        processor_size=1,
        mlp_activation_fn="relu",
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        hidden_dim_processor=256,
        hidden_dim_node_encoder=256,
        num_layers_node_encoder=2,
        hidden_dim_edge_encoder=256,
        num_layers_edge_encoder=2,
        hidden_dim_node_decoder=256,
        num_layers_node_decoder=2,
        aggregation="sum",
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        checkpoint_offloading=False,
        recompute_activation=False,
        norm_type="LayerNorm",
    ).to(device)

    cfg.triton.cudagraph_skip_dynamic_graphs = True
    model: torch.nn.Module = torch.compile(  # pyright: ignore[reportAssignmentType]
        model, mode="reduce-overhead", fullgraph=True, dynamic=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-5, fused=True
    )
    loss_fn = torch.nn.MSELoss()
    scaler = torch.GradScaler(device.type)

    # Warmup before profiling
    WARMUP_STEPS = 5
    model.train()
    it = iter(train_loader)
    for _ in range(WARMUP_STEPS):
        batch = next(it)
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(batch.pos, batch.edge_attr, batch)
            loss = loss_fn(out, batch.x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.synchronize()

    # Profiling
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
    ) as prof:
        num_epochs = 1
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                with record_function("dataloader.to(device)"):
                    batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with record_function("forward"):
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        out = model(batch.pos, batch.edge_attr, batch)
                        loss = loss_fn(out, batch.x)

                with record_function("backward"):
                    scaler.scale(loss).backward()

                with record_function("optimizer.step"):
                    scaler.step(optimizer)

                scaler.update()
                prof.step()

        if use_cuda:
            torch.cuda.synchronize()

"""Minimal example of using MeshGraphNet and MeshGraphKAN on a PLAID dataset."""

from datasets import load_dataset
from physicsnemo.models.meshgraphnet import (
    MeshGraphKAN,
    MeshGraphNet,
)
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
    huggingface_description_to_problem_definition,
)
from plaid.types import FeatureIdentifier
from plaid_bridges.torch import PyGBridge
from rich.pretty import pprint
from torch_geometric.transforms import Cartesian

# Load dataset from Hugging Face
hf_dataset = load_dataset("PLAID-datasets/Rotor37", split="all_samples")
problem_definition = huggingface_description_to_problem_definition(
    hf_dataset.info.description
)

# Convert to PLAID dataset
ids_train = problem_definition.get_split("train_16")
dataset, _ = huggingface_dataset_to_plaid(hf_dataset, ids=ids_train, processes_number=5)

# pprint(dataset.get_all_features_identifiers())

feature_ids = [
    FeatureIdentifier(name="Compression_ratio", type="scalar"),
    FeatureIdentifier(name="Pressure", type="field"),  # -> stored in data.x
    # FeatureIdentifier(type="nodes") -> nodes are always included, this should raise an error but it doesn't
]
bridge = PyGBridge()
pyg_data_list = bridge.transform(
    dataset, features_ids=feature_ids
)  # -> the bridge converts to torch tensors in float64, that's bad

# We need edge attributes for MeshGraphNet, let's use Cartesian distances
transform = Cartesian()
pyg_data_list = [transform(data) for data in pyg_data_list]

# convert data.x, edge_attr to float32
for data in pyg_data_list:
    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    data.pos = data.pos.float()

# Define and run MeshGraphNet
model = MeshGraphNet(
    input_dim_nodes=3, input_dim_edges=3, output_dim=1, processor_size=1
)
out = model(
    pyg_data_list[0].pos,  # nodes_features
    pyg_data_list[0].edge_attr,  # edges_features
    pyg_data_list[0],  # graph (Data)
)
pprint(out)

# Define and run MeshGraphKAN
model = MeshGraphKAN(input_dim_nodes=3, input_dim_edges=3, output_dim=1)
out = model(
    pyg_data_list[0].pos,  # nodes_features
    pyg_data_list[0].edge_attr,  # edges_features
    pyg_data_list[0],  # graph (Data)
)
pprint(out)

# <!-- Those two seem to only work with DGLGraph, need to investigate -->

# # Define and run HybridMeshGraphNet
# model = HybridMeshGraphNet(
#     input_dim_nodes=3, input_dim_edges=3, output_dim=1, processor_size=1
# )
# out = model(pyg_data_list[0].pos, pyg_data_list[0].edge_attr, pyg_data_list[0].edge_attr, pyg_data_list[0])
# pprint(out)

# # Define and run BiStrideMeshGraphNet
# model = BiStrideMeshGraphNet(
#     input_dim_nodes=3, input_dim_edges=3, output_dim=1, processor_size=1
# )
# out = model(pyg_data_list[0].pos, pyg_data_list[0].edge_attr, pyg_data_list[0])
# pprint(out)

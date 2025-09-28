import torch
from datasets import load_dataset
from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet
from plaid import Sample
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
    huggingface_description_to_problem_definition,
)

# Load dataset from Hugging Face
hf_dataset = load_dataset("PLAID-datasets/Rotor37", split="all_samples")
problem_definition = huggingface_description_to_problem_definition(
    hf_dataset.info.description
)

# Convert to PLAID dataset
ids_train = problem_definition.get_split("train_16")
dataset, _ = huggingface_dataset_to_plaid(hf_dataset, ids=ids_train, processes_number=5)

samples = dataset.get_samples(as_list=True)
sample: Sample = samples[0]
nodes = sample.get_nodes()

# We should either project on a 3D grid or compute the SDF.
# ...

# Define and run FIGConvUNet
model = FIGConvUNet(
    in_channels=3, out_channels=5, hidden_channels=[3, 32], kernel_size=3, num_levels=1
).cuda()

x = torch.randn(16, 10, 3).cuda()
features, drag_pred = model(x)
print(features.shape, drag_pred.shape)

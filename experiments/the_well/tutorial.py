"""Tutorial for loading The Well dataset using the `the_well` package."""

import matplotlib.pyplot as plt
from the_well.data import WellDataset

base_path = "/mnt/c/Users/brian/Downloads/the_well/datasets"

name = "turbulent_radiative_layer_2D"
dataset = WellDataset(
    well_base_path=base_path,
    well_dataset_name=name,
    well_split_name="train",
)

input_fields = dataset[99]["input_fields"]
output_fields = dataset[99]["output_fields"]

print("Input fields shape:", input_fields.shape)
print("Output fields shape:", output_fields.shape)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]

data = input_fields[0].numpy()
for i in range(4):
    im = axes[i].imshow(data[:, :, i], cmap="seismic")
    axes[i].set_title(titles[i])
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("input_fields.png", dpi=300)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]

data = output_fields[0].numpy()
for i in range(4):
    im = axes[i].imshow(data[:, :, i], cmap="seismic")
    axes[i].set_title(titles[i])
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("output_fields.png", dpi=300)

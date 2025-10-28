"""Train a Fourier Neural Operator (FNO) model on the "The Well" dataset."""

import logging

import matplotlib.pyplot as plt
import torch
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
        n_steps_output=1,
    )
    print(f"train dataset length: {len(dataset)}")

    # sample = dataset.__getitem__(50)
    # for key, value in sample.items():
    #     if value.ndim > 2:
    #         print(key, value.shape)
    #     else:
    #         print(key, value)
    # for i in range(200):
    #     sample = dataset.__getitem__(i)
    #     input_fields = sample["input_fields"]
    #     output_fields = sample["output_fields"]
    #     plt.figure()
    #     plt.imshow(input_fields[0, :, :, 0].numpy())
    #     plt.title(f"Input field sample {i}")
    #     plt.savefig(f"input_field_sample_{i}.png")
    #     plt.close()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, batch in enumerate(dataloader):
        print("--- Batch", idx, "---")
        for key, value in batch.items():
            if value.ndim > 2:
                print(key, value.shape)
            else:
                print(key, value)


if __name__ == "__main__":
    main()

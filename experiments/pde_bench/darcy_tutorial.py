"""Simple tutorial to read and visualize 2D Darcy Flow data from PDEBench."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

filepath = Path("/mnt/c/Users/brian/Downloads")
filename = "2D_DarcyFlow_beta0.1_Train.hdf5"

nb = 200
with h5py.File(filepath / filename, "r") as h5_file:
    data = np.array(h5_file["tensor"], dtype=np.float32)[
        nb
    ]  # (batch, t, x, y, channel) --> (t, x, y, channel)
    nu = np.array(h5_file["nu"], dtype=np.float32)[
        nb
    ]  # (batch, t, x, y, channel) --> (t, x, y, channel)
    print(data.shape, nu.shape)


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(data.squeeze())
ax[1].imshow(nu.squeeze())
ax[0].set_title("Data u")
ax[1].set_title("diffusion coefficient nu")
plt.show()

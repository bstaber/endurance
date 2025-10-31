"""Script to read the Car CFD Dataset."""

from neuralop.data.datasets.car_cfd_dataset import CarCFDDataset

dataset = CarCFDDataset(
    root_dir="/mnt/c/Users/brian/cache/car_cfd_dataset/processed-car-pressure-data",
    n_train=-1,
    n_test=-1,
    download=False,
)

train_loader = dataset.train_loader(
    batch_size=32, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True
)
test_loader = dataset.test_loader(
    batch_size=32,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)

for batch in train_loader:
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
    """
    vertices: torch.Size([32, 3586, 3])
    vertex_normals: torch.Size([32, 3586, 3])
    triangle_normals: torch.Size([32, 7168, 3])
    centroids: torch.Size([32, 7168, 3])
    triangle_areas: torch.Size([32, 7168])
    distance: torch.Size([32, 32, 32, 32, 1])
    closest_points: torch.Size([32, 32, 32, 32, 3])
    normalized_triangle_areas: torch.Size([32, 7168])
    press: torch.Size([32, 1, 3586])
    query_points: torch.Size([32, 32, 32, 32, 3])
    """
    break

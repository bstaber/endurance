"""What the hell am I looking at?

DoMINO: Deep Differential Operators for Mesh-based Implicit Neural
------------------------------------------------------------------

DoMINO predicts fields at *points* (surface and/or volume), but first builds a
global, grid-based encoding of the geometry. The forward pass has two stages:

1) Global geometry encoding (once per forward):
   - Project the input surface point cloud onto a regular 3-D grid inside a
     bounding box (and optionally onto a larger “computational domain” grid).
   - For each grid voxel, gather nearby surface points (ball query “point
     convolution”) and aggregate them with a small MLP; optionally concatenate
     SDF (+ scaled/binary SDF and ∇SDF).
   - Run a small 3-D CNN/UNet over the whole grid to propagate/contextualize
     features. The result is a dense feature field on the grid(s).

2) Local prediction at query points (many times, no grid conv here):
   - For each query point i, crop a tiny l×l×l patch from the global grid
     around i (sampling/indexing only) to get a *local geometry code*.
   - Build a *stencil* for i (i plus K neighbors: KNN on the surface or
     random sphere/shell in the volume). Pass these through a basis-function
     MLP; concatenate with the local geometry code (+ optional encoded global
     parameters); predict center/neighbor values and inverse-distance blend
     back to the center.

Notes:
- Outputs are on points, not on the grid. The grid is a latent canvas to hold
  multi-scale geometric context.
- In “STL” geometry mode the SDF channels are not used (but some keys still
  need to exist in the current reference implementation).
- PCA/alignment of the surface before building the grid tightens the box and
  reduces wasted voxels.

Inputs: tensors and how they map to the paper
---------------------------------------------

Common (surface-only; STL mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Required keys (B = batch, Ns = surface points, K = neighbors, nx×ny×nz = grid):
- geometry_coordinates        : (B, Ns, 3)
    Surface point cloud used to paint geometry onto the surface grid (global stage).
- surface_mesh_centers        : (B, Ns, 3)
    Same centers used for local prediction (stencil center).
- surface_mesh_neighbors      : (B, Ns, K, 3)
    KNN coordinates per center (stencil neighbors).
- surface_normals             : (B, Ns, 3)
- surface_neighbors_normals   : (B, Ns, K, 3)
    If use_surface_normals=True. Included as extra per-point features.
- surface_areas               : (B, Ns)
- surface_neighbors_areas     : (B, Ns, K)
    If use_surface_area=True. Positive areas; often per-vertex area from faces.
- surface_min_max             : (B, 2, 3)
    Per-axis min/max of surface points; defines the surface bounding box.
- surf_grid                   : (B, nx, ny, nz, 3)
    Coordinates of the surface bounding-box grid (Fig. 2A). Used to write/read
    global geometry features. Build from surface_min_max.
- sdf_surf_grid               : (B, nx, ny, nz)
    SDF on the surface grid. In STL mode not used; pass zeros but keep the key.
- pos_surface_center_of_mass  : (B, Ns, 3)
    Vector from global COM to each surface point; a simple basis feature.
- global_params_values        : (B, G, 1)
- global_params_reference     : (B, G, 1)
    Placeholders if encode_parameters=False (e.g., zeros/ones). If True, these
    carry physical scalars (e.g., freestream velocity, density).

Surface + SDF mode (geometry_encoding_type="sdf" or "both")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add/compute real SDF values:
- sdf_surf_grid               : (B, nx, ny, nz)
    Signed distance on the surface grid. The model also forms scaled/binary SDF
    and ∇SDF internally.

Combined mode (surface + volume)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add a *computational domain* grid and volume point features:
- grid                        : (B, nx, ny, nz, 3)
    Coordinates of the larger domain grid (Fig. 2A; pink box).
- sdf_grid                    : (B, nx, ny, nz)
    SDF on the domain grid (zeros in STL-only pipelines).
- volume_min_max              : (B, 2, 3)
    Min/max defining the domain grid’s bounding box.
- volume_mesh_centers         : (B, Nv, 3)
    Query points in the volume for prediction.
- sdf_nodes                   : (B, Nv, 1)
    SDF sampled at volume points (used if use_sdf_in_basis_func=True).
- pos_volume_closest          : (B, Nv, 3)
    Vector from each volume point to its closest surface point.
- pos_volume_center_of_mass   : (B, Nv, 3)
    Vector from global COM to each volume point.

Concepts (paper ↔ tensors)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- “Surface bounding box / grid (Gs)”     ↔ surf_grid (+ sdf_surf_grid).
- “Computational domain / grid (Gc)”     ↔ grid (+ sdf_grid) in combined mode.
- “Point convolution kernels (ball-query)
   & painting geometry on grids”         ↔ GeometryRep uses geometry_coordinates
                                           + (surf_grid / grid) (+ SDF) to
                                           produce global grid features (Fig. 2B).
- “Local l×l×l subregion crop”           ↔ geo_encoding_local samples small
                                           patches around each query point (no
                                           extra inputs needed—done internally).
- “Stencil”                              ↔ surface_mesh_neighbors (surface) or
                                           sampled sphere/shell (volume). Neighbors
                                           are passed through a basis MLP and
                                           inverse-distance aggregated.

Typical shapes
~~~~~~~~~~~~~~
- geometry_coordinates, surface_mesh_centers         : (B, Ns, 3)
- surface_mesh_neighbors                             : (B, Ns, K, 3)
- surface_normals / neighbors_normals                : (B, Ns, 3) / (B, Ns, K, 3)
- surface_areas / neighbors_areas                    : (B, Ns) / (B, Ns, K)
- surf_grid / grid                                   : (B, nx, ny, nz, 3)
- sdf_surf_grid / sdf_grid                           : (B, nx, ny, nz)
- surface_min_max / volume_min_max                   : (B, 2, 3)
- pos_surface_center_of_mass / pos_volume_*          : (B, N*, 3)
- global_params_values / reference                   : (B, G, 1)

Practical tips
~~~~~~~~~~~~~~
- STL mode: provide zeros for sdf_* keys but keep them present.
- Align the mesh with PCA before building grids to tighten the box.
- Match flags to what you include (use_surface_normals / use_surface_area /
  encode_parameters / geometry_encoding_type).
"""

# /// script
# dependencies = [
#   "pyplaid<=1.0.0",
# ]
# ///

import numpy as np
import pyvista as pv
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from physicsnemo.models.domino import DoMINO
from plaid import Sample
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
    huggingface_description_to_problem_definition,
)
from scipy.spatial import cKDTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = OmegaConf.load("config.yaml")
config = OmegaConf.to_container(config.model, resolve=True)


def pca_rotate_nodes(
    nodes: np.ndarray,
):
    """PCA-align a point cloud to its principal axes (right-handed)."""
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("`nodes` must have shape (N, 3)")

    c = nodes.mean(axis=0)
    X = nodes - c

    C = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    R = vecs[:, order]

    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    nodes_rot = X @ R

    return nodes_rot


def make_bbox_grid_from_minmax(
    surf_min_max: torch.Tensor, nx: int, ny: int, nz: int, device=None
):
    """Build a regular grid of shape (B, nx, ny, nz, 3) inside the bounding box defined by surf_min_max."""
    B = surf_min_max.shape[0]
    dev = device if device is not None else surf_min_max.device
    mins = surf_min_max[:, 0]  # (B, 3)
    maxs = surf_min_max[:, 1]  # (B, 3)

    # build per-batch grids
    grids = []
    for b in range(B):
        xs = torch.linspace(mins[b, 0], maxs[b, 0], nx, device=dev)
        ys = torch.linspace(mins[b, 1], maxs[b, 1], ny, device=dev)
        zs = torch.linspace(mins[b, 2], maxs[b, 2], nz, device=dev)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")  # (nx,ny,nz)
        G = torch.stack([X, Y, Z], dim=-1)  # (nx,ny,nz,3)
        grids.append(G)
    surf_grid = torch.stack(grids, dim=0)  # (B,nx,ny,nz,3)
    return surf_grid


def pos_from_center_of_mass(centers: torch.Tensor) -> torch.Tensor:
    """Center-of-mass position basis feature."""
    com = centers.mean(dim=1, keepdim=True)  # (B,1,3)
    return centers - com


def build_domino_surface_from_pyvista(
    nodes: np.ndarray,
    quads: np.ndarray,
    *,
    representation: str = "vertex",  # "vertex" -> use points; "face" -> use cell centers
    k_neighbors: int = 7,
    include_normals: bool = True,
    include_areas: bool = True,
    device: str = "cuda:0",
    dtype_torch=torch.float32,
):
    """Build a DoMINO surface from a PyVista mesh."""
    q = quads.astype(np.int64, copy=False)
    if q.min() == 1:
        q = q - 1

    faces = np.hstack(
        [np.full((q.shape[0], 1), 4, dtype=np.int64), q]
    ).ravel()  # [4, i0, i1, i2, i3] ...
    mesh = pv.PolyData(nodes, faces)
    mesh = mesh.clean()

    # Normals: both point & cell (auto_orient gives consistent directions)
    mesh = mesh.compute_normals(
        point_normals=True, cell_normals=True, auto_orient_normals=True
    )
    # Cell areas:
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    # Cell centers:
    cell_centers = mesh.cell_centers().points  # (M,3)

    # ---- Choose representation (what DoMINO will see as "centers") ----
    if representation == "vertex":
        centers_np = np.asarray(mesh.points)  # (N,3)
        if include_normals:
            normals_np = np.asarray(mesh.point_data["Normals"])  # (N,3)
        if include_areas:
            # convert cell areas to per-vertex area = sum(face_area/4) for incident quads
            areas_np = np.zeros(centers_np.shape[0], dtype=float)
            quad_idx = mesh.faces.reshape(-1, 5)[:, 1:]  # (M,4)
            cell_area = np.asarray(mesh.cell_data["Area"])  # (M,)
            for i in range(quad_idx.shape[0]):
                areas_np[quad_idx[i]] += cell_area[i] * 0.25
    elif representation == "face":
        centers_np = cell_centers  # (M,3)
        if include_normals:
            normals_np = np.asarray(mesh.cell_data["Normals"])  # (M,3)
        if include_areas:
            areas_np = np.asarray(mesh.cell_data["Area"])  # (M,)
    else:
        raise ValueError("representation must be 'vertex' or 'face'")

    # ---- KNN neighbors on chosen centers ----
    tree = cKDTree(centers_np)
    _, idx = tree.query(centers_np, k=k_neighbors + 1)  # include self
    idx = idx[:, 1:]  # drop self
    neighbors_np = centers_np[idx]  # (Ns,K,3)

    # ---- Torchify + DoMINO keys ----
    dev = torch.device(device)
    centers = torch.as_tensor(centers_np, dtype=dtype_torch, device=dev)[
        None, ...
    ]  # (1,Ns,3)
    neighbors = torch.as_tensor(neighbors_np, dtype=dtype_torch, device=dev)[
        None, ...
    ]  # (1,Ns,K,3)

    input_dict = {
        "surface_mesh_centers": centers,
        "surface_mesh_neighbors": neighbors,
        "geometry_coordinates": centers,  # DoMINO.forward expects this
    }

    if include_normals:
        surf_normals = torch.as_tensor(normals_np, dtype=dtype_torch, device=dev)[
            None, ...
        ]  # (1,Ns,3)
        neigh_normals = torch.as_tensor(normals_np[idx], dtype=dtype_torch, device=dev)[
            None, ...
        ]  # (1,Ns,K,3)
        input_dict.update(
            {
                "surface_normals": surf_normals,
                "surface_neighbors_normals": neigh_normals,
            }
        )

    if include_areas:
        areas_np = np.maximum(areas_np, 1e-12)  # strictly positive
        surf_areas = torch.as_tensor(areas_np, dtype=dtype_torch, device=dev)[
            None, ...
        ]  # (1,Ns)
        neigh_areas = torch.as_tensor(areas_np[idx], dtype=dtype_torch, device=dev)[
            None, ...
        ]  # (1,Ns,K)
        input_dict.update(
            {
                "surface_areas": surf_areas,
                "surface_neighbors_areas": neigh_areas,
            }
        )

    mins = centers.min(dim=1).values
    maxs = centers.max(dim=1).values
    input_dict["surface_min_max"] = torch.stack((mins, maxs), dim=1)  # (1,2,3)

    return input_dict, {"mesh": mesh, "idx": idx}


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
nodes = pca_rotate_nodes(nodes)
elements = sample.meshes.get_elements()["QUAD_4"]

# 0) Build DoMINO input_dict from surface mesh
input_dict, _extras = build_domino_surface_from_pyvista(
    nodes,
    elements,
    representation="vertex",
    k_neighbors=7,
    include_normals=True,
    include_areas=True,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype_torch=torch.float32,
)

device = input_dict["geometry_coordinates"].device
nx, ny, nz = config["interp_res"]

# 1) surf_grid: required even for STL (used by GeometryRep’s ball-query)
# Coordinates of the surface bounding-box grid (Gs) used to paint geometry via point kernels.
input_dict["surf_grid"] = make_bbox_grid_from_minmax(
    input_dict["surface_min_max"], nx, ny, nz, device=device
)

# 2) sdf_surf_grid: not used when geometry_encoding_type="stl", but must exist; zeros are fine
input_dict["sdf_surf_grid"] = torch.zeros(
    (1, nx, ny, nz), dtype=torch.float32, device=device
)

# 3) pos_surface_center_of_mass: required by forward()
input_dict["pos_surface_center_of_mass"] = pos_from_center_of_mass(
    input_dict["geometry_coordinates"]
)

# 4) global params placeholders: forward() reads them even if encode_parameters=False
G = 2  # matches DoMINO default global_features
input_dict["global_params_values"] = torch.zeros(
    (1, G, 1), dtype=torch.float32, device=device
)
input_dict["global_params_reference"] = torch.ones(
    (1, G, 1), dtype=torch.float32, device=device
)

for key, value in input_dict.items():
    print(f"{key}: {value.shape} {value.dtype} {value.device}")

config = OmegaConf.load("config.yaml")
model = DoMINO(
    input_features=3,
    output_features_vol=None,
    output_features_surf=1,
    model_parameters=config.model,
).to(device)

output = model(input_dict)
print("Output shape")
for out in output:
    if out is None:
        print(out)
    else:
        print(out.shape)

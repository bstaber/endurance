import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) Make a synthetic dataset (heteroscedastic + a touch of multi-modality)
g = torch.Generator().manual_seed(0)
N_train, N_test = 3000, 600
x_train = torch.rand(N_train, 1, generator=g) * 6 - 3.0
noise = 0.2 + 0.3 * torch.sigmoid(2.0 * x_train)
y_train = torch.sin(2.0 * x_train) + noise * torch.randn_like(x_train)
# add a small second mode with low prob
mask = torch.rand(N_train, 1, generator=g) < 0.10
y_train = torch.where(mask, y_train + 1.2 * torch.sin(6 * x_train), y_train)

# Test grid to visualize predictions
x_test = torch.linspace(-3, 3, N_test).unsqueeze(1)

# 2) Build quantile bins for y (Riemann buckets)
K = 80  # number of bins (50–200 is typical)
with torch.no_grad():
    edges = torch.quantile(y_train.squeeze(1), torch.linspace(0, 1, K + 1))
edges[0] -= 1e-6  # ensure inclusivity on the left


def bin_index(y, edges):
    # returns indices in [0, K-1]
    return torch.bucketize(y.squeeze(-1), edges[1:-1])


def bin_centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])


centers = bin_centers(edges)  # [K]
widths = edges[1:] - edges[:-1]  # [K]


# 3) Model: MLP that outputs logits over K bins
class MLP_Riemann(nn.Module):
    def __init__(self, in_dim=1, hidden=128, K=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        return self.net(x)  # logits


model = MLP_Riemann(K=K)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4) Train with cross-entropy on the true bin
B = 256
steps = 1000
for s in range(steps):
    idx = torch.randint(0, N_train, (B,), generator=g)
    xb = x_train[idx]
    yb = y_train[idx]
    jb = bin_index(yb, edges)  # target bin id
    logits = model(xb)  # [B,K]
    loss = F.cross_entropy(logits, jb)  # CE over bins
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (s + 1) % 10 == 0:
        print(f"step {s + 1}/{steps} | loss {loss.item():.4f}")


# 5) Inference: bin masses, PDF, mean/var
@torch.no_grad()
def predictive_mass(x):
    return F.softmax(model(x), dim=-1)  # [B,K]


@torch.no_grad()
def pdf_from_mass(mass):
    # piece-wise constant PDF value per bin height
    return mass / widths.unsqueeze(0)  # [B,K]


@torch.no_grad()
def mean_from_mass(mass):
    return (mass * centers.unsqueeze(0)).sum(dim=1)


@torch.no_grad()
def var_from_mass(mass):
    mu = mean_from_mass(mass).unsqueeze(1)
    return (mass * (centers.unsqueeze(0) - mu) ** 2).sum(dim=1)


mass = predictive_mass(x_test)  # [N_test, K]
pdf = pdf_from_mass(mass)  # [N_test, K]
mu = mean_from_mass(mass)  # [N_test]
var = var_from_mass(mass)  # [N_test]
std = var.sqrt()

# 6) Visualizations
# (a) Heatmap of predicted PDF over (x,y)
Y = centers.numpy()
X = x_test.squeeze(1).numpy()
Z = pdf.numpy().T  # shape [K, N_test] so rows correspond to y-bins
plt.figure(figsize=(8, 4))
plt.imshow(
    Z,
    aspect="auto",
    origin="lower",
    extent=[X.min(), X.max(), Y.min().item(), Y.max().item()],
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted density q(y|x) (piece-wise constant)")
plt.colorbar()
plt.tight_layout()

# (b) Mean ± 2 std overlaid on training scatter (downsample for viz)
plt.figure(figsize=(8, 4))
idx_plot = torch.randperm(N_train, generator=g)[:1000]
plt.scatter(
    x_train[idx_plot].squeeze(1).numpy(),
    y_train[idx_plot].squeeze(1).numpy(),
    s=6,
    alpha=0.4,
)
plt.plot(X, mu.numpy())
plt.plot(X, (mu - 2 * std).numpy())
plt.plot(X, (mu + 2 * std).numpy())
plt.xlabel("x")
plt.ylabel("y")
plt.title("Mean prediction with ~95% band and training samples")
plt.tight_layout()

plt.show()

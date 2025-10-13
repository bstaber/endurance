"""LoRA applied to Physics-Informed Neural Networks (PINNs) for Burgers' equation."""

import numpy as np
import scipy
import torch
import torch.nn as nn
from lora.models import BaseMLP, LoRAMLP
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset


def loss_burgers(
    model: nn.Module,
    xs_pde: torch.Tensor,
    nu: float = 0.01 / torch.pi,
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    n_ic: int = 128,
    n_bc: int = 128,
):
    """Compute the PINN loss for the Burgers' equation."""
    # xs_pde: (B_pde, 2) with requires_grad=True
    xs_pde = xs_pde.requires_grad_(True)
    u = model(xs_pde)  # (B_pde, 1)

    # grads wrt inputs
    g = grad(u, xs_pde, torch.ones_like(u), create_graph=True)[0]
    u_x = g[:, 0:1]
    u_t = g[:, 1:2]
    u_xx = grad(u_x, xs_pde, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]

    r = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(r**2)

    # sample IC/BC fresh each call (common in PINNs)
    device = xs_pde.device
    x_ic = x_min + (x_max - x_min) * torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(torch.cat([x_ic, t_ic], dim=1))
    target_ic = -torch.sin(torch.pi * x_ic)
    loss_ic = torch.mean((u_ic - target_ic) ** 2)

    t_bc = t_min + (t_max - t_min) * torch.rand(n_bc, 1, device=device)
    x0 = torch.full_like(t_bc, x_min)
    x1 = torch.full_like(t_bc, x_max)
    u_left = model(torch.cat([x0, t_bc], dim=1))
    u_right = model(torch.cat([x1, t_bc], dim=1))
    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

    return loss_pde + loss_ic + loss_bc


if __name__ == "__main__":
    filename = "./data/burgers_shock.mat"
    data = scipy.io.loadmat(filename)

    nu = 0.01 / torch.pi
    x = torch.tensor(data["x"], dtype=torch.float32)
    t = torch.tensor(data["t"], dtype=torch.float32)
    u_sol = torch.tensor(data["usol"], dtype=torch.float32)

    # Create meshgrid (training points)
    X_grid, T_grid = torch.meshgrid(x.flatten(), t.flatten(), indexing="ij")
    X_star = torch.stack([X_grid.flatten(), T_grid.flatten()], axis=-1)  # (25600, 2)

    # Subsample the training data
    n_pde, n_obs = 1000, 1000
    idx_pde = np.random.choice(len(X_star), size=(n_pde,), replace=False)
    idx_obs = np.random.choice(len(X_star), size=(n_obs,), replace=False)
    xs_pde = X_star[idx_pde]
    xs_obs = X_star[idx_obs]
    u_obs = u_sol.flatten()[idx_obs].unsqueeze(-1)

    # Base MLP
    xs_pde.requires_grad = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseMLP(input_dim=2, hidden_dim=10, output_dim=1, num_layers=10).to(device)

    # Training base model
    optim = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=0.0)
    obs_loader = DataLoader(TensorDataset(xs_obs, u_obs), batch_size=64, shuffle=True)

    num_epochs = 1000
    loss_data = nn.MSELoss()
    train_loss = []
    for epoch in range(num_epochs):
        model.train()
        loss_train_epoch = 0.0
        for xs_batch_obs, u_obs_batch in obs_loader:
            xs_batch_obs, u_obs_batch = xs_batch_obs.to(device), u_obs_batch.to(device)
            optim.zero_grad()
            pred = model(xs_batch_obs)
            loss = loss_data(pred, u_obs_batch)
            loss.backward()
            optim.step()
            loss_train_epoch += loss.item()
        loss_train_epoch /= len(obs_loader)
        train_loss.append(loss_train_epoch)

    # Fine-tune with LoRA
    pde_loader = DataLoader(TensorDataset(xs_pde), batch_size=128, shuffle=True)

    model_lora = LoRAMLP(
        input_dim=2,
        hidden_dim=10,
        output_dim=1,
        num_layers=10,
        base_state_dict=model.state_dict(),
        r=4,
        alpha=8.0,
    ).to(device)

    lora_optim = torch.optim.AdamW(
        [p for p in model_lora.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.0,
    )

    loss_train_lora = []
    model_lora.train()
    for _ in range(1000):
        loss_epoch = 0.0
        for xs_pde_batch in pde_loader:
            xs_pde_batch = xs_pde_batch[0].to(device)
            lora_optim.zero_grad()
            loss = loss_burgers(model_lora, xs_pde_batch)
            loss.backward()
            loss_epoch += loss.item()
            lora_optim.step()
        loss_train_lora.append(loss_epoch / len(pde_loader))

    # Plot losses
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].semilogy(train_loss, color="k", ls="-")
    axs[1].semilogy(loss_train_lora, color="b", ls="-")
    plt.show()

    # Plot predictions
    model.eval()
    with torch.no_grad():
        pred_base_model = model(X_star).reshape(X_grid.shape).cpu().numpy()
        pred_lora_model = model_lora(X_star).reshape(X_grid.shape).cpu().numpy()
    u_sol = u_sol.cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    c2 = axs[0, 0].pcolormesh(
        T_grid, X_grid, pred_base_model, shading="auto", cmap="jet"
    )
    fig.colorbar(c2, ax=axs[0, 0], label="u(x, t)")
    axs[0, 0].set_title("Base MLP solution")
    c2 = axs[0, 1].pcolormesh(
        T_grid, X_grid, pred_lora_model, shading="auto", cmap="jet"
    )
    fig.colorbar(c2, ax=axs[0, 1], label="u(x, t)")
    axs[0, 1].set_title("LoRA fine-tuned solution")
    c2 = axs[1, 0].pcolormesh(
        T_grid, X_grid, np.abs(pred_base_model - u_sol), shading="auto", cmap="jet"
    )
    fig.colorbar(c2, ax=axs[1, 0], label="Error")
    axs[1, 0].set_title("Abs error")
    c2 = axs[1, 1].pcolormesh(
        T_grid, X_grid, np.abs(pred_lora_model - u_sol), shading="auto", cmap="jet"
    )
    fig.colorbar(c2, ax=axs[1, 1], label="Error")
    axs[1, 1].set_title("Abs error")
    plt.show()

"""LoRA MLP model implementation for experiments.

We implement a simple MLP model and its LoRA variant for experiments.

The code includes:
- BaseMLP: A standard MLP model.
- LoRALinear: A linear layer with LoRA adaptation.
- LoRAMLP: An MLP model using LoRA layers.
- Example usage with dummy regression data.

In the example, we:
1. Train a base MLP on a regression task and plot the predictions (sanity check).
2. Fully fine-tune the base MLP on a new task for comparison.
3. Fine-tune the LoRA MLP on the new task and compare performance.
4. Fully retrain a new base MLP on the new task for comparison.
5. Then, for the three models, we plot the training losses and predictions for comparison.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lora.models import BaseMLP, LoRAMLP

# Example usage
input_dim = 1
hidden_dim = 20
output_dim = input_dim
num_layers = 3

# Dummy regression data
x = torch.rand(1000, input_dim)
y = torch.cos(x * 3.14 * 2) + torch.randn_like(x) * 0.1
x_test = torch.linspace(0, 1, 100).unsqueeze(1).repeat(1, input_dim)

# Train base MLP
model_first_task = BaseMLP(input_dim, hidden_dim, output_dim, num_layers)

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
model_first_task.train()
optimizer = torch.optim.AdamW(model_first_task.parameters(), lr=5e-3)
for _ in range(1000):
    optimizer.zero_grad()
    loss = F.mse_loss(model_first_task(x), y)
    loss.backward()
    optimizer.step()

model_first_task.eval()
with torch.no_grad():
    preds = model_first_task(x_test)

plt.figure()
plt.scatter(x.numpy(), y.numpy(), label="data", alpha=0.3)
plt.plot(x_test.numpy(), preds.numpy(), color="red", label="model")
plt.legend()
plt.show()

# Dummy new task data
x = torch.rand(1000, input_dim)
y = torch.sin(x * 3.14 * 4) + torch.randn_like(x) * 0.1
x_test = torch.linspace(0, 1, 100).unsqueeze(1).repeat(1, input_dim)
dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the base model on the new task for comparison
model = BaseMLP(input_dim, hidden_dim, output_dim, num_layers)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

loss_train_base = []
model.train()
for _ in range(50):
    loss_epoch = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        loss_epoch += loss.item()
        optimizer.step()
    loss_train_base.append(loss_epoch / len(dataloader))

# Fully finetune the first model
model_full_finetuned = BaseMLP(input_dim, hidden_dim, output_dim, num_layers)
model_full_finetuned.load_state_dict(model_first_task.state_dict())
optimizer = torch.optim.AdamW(model_full_finetuned.parameters(), lr=5e-3)

loss_train_full_finetuned = []
model_full_finetuned.train()
for _ in range(50):
    loss_epoch = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        loss = F.mse_loss(model_full_finetuned(xb), yb)
        loss.backward()
        loss_epoch += loss.item()
        optimizer.step()
    loss_train_full_finetuned.append(loss_epoch / len(dataloader))

# Fine-tune the LoRA MLP on another task
# Initialize LoRA MLP with the pre-trained base model's state dict
model_lora = LoRAMLP(
    input_dim,
    hidden_dim,
    output_dim,
    num_layers,
    model_first_task.state_dict(),
    r=4,
    alpha=8.0,
)
opt = torch.optim.AdamW(
    [p for p in model_lora.parameters() if p.requires_grad],
    lr=5e-3,
    weight_decay=1e-2,
)

loss_train_lora = []
model_lora.train()
for _ in range(50):
    loss_epoch = 0.0
    for xb, yb in dataloader:
        opt.zero_grad()
        loss = F.mse_loss(model_lora(xb), yb)
        loss.backward()
        loss_epoch += loss.item()
        opt.step()
    loss_train_lora.append(loss_epoch / len(dataloader))

model_lora.eval()
with torch.no_grad():
    preds_lora = model_lora(x_test)
    preds_full_finetune = model_full_finetuned(x_test)
    preds_base_retrained = model(x_test)

plt.figure()
plt.plot(loss_train_lora, label="LoRA model on 2nd task")
plt.plot(loss_train_base, label="Fully trained base model")
plt.plot(loss_train_full_finetuned, label="Fully finetuned 1st base model")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

plt.figure()
plt.scatter(x.numpy(), y.numpy(), color="k", label="data", alpha=0.3)
plt.plot(x_test.numpy(), preds_lora.numpy(), color="red", label="Finetuned LoRA model")
plt.plot(
    x_test.numpy(),
    preds_full_finetune.numpy(),
    color="blue",
    label="Fully finetuned 1st base model",
)
plt.plot(
    x_test.numpy(),
    preds_base_retrained.numpy(),
    color="green",
    label="Fully trained base model",
)
plt.legend()
plt.show()

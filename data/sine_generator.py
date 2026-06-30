import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# utils.py liegt im Repo-Root (nicht unter synthesis/). Aus dem Repo-Root
# ausfuehren, oder den Repo-Root in PYTHONPATH haben.
from utils import generate_data


class SineTarget(nn.Module):
    def forward(self, x):                       # x: (N, 1)
        return torch.sin(x) * 5.0 + 0.5 * x     # -> (N, 1)


# ===================
# Data Generation
# ===================
generation_model = SineTarget()
x, y = generate_data(generation_model, xmin=-10, xmax=10, n_samples=1_000, eps=1e-4)
x_test, y_test = generate_data(generation_model, xmin=-15, xmax=15, n_samples=1_000, eps=1e-4)
print(f"Train: x={tuple(x.shape)} y={tuple(y.shape)} | "
      f"Test: x={tuple(x_test.shape)} y={tuple(y_test.shape)}")

# ---- Datensatz speichern ----
# WICHTIG: Die Keys muessen exakt x, y, x_test, y_test heissen, sonst meldet
# load_dataset() "missing keys". Zusatz-Keys wie meta_data werden ignoriert.
# Endung .pt, damit `--dataset sine` die Datei als data/sine_dataset.pt findet.
os.makedirs("data", exist_ok=True)
torch.save(
    {
        "x": x,
        "y": y,
        "x_test": x_test,
        "y_test": y_test,
        "meta_data": {"generation_model": "SineTarget"},
    },
    "data/sine_dataset.pt",
)

# ---- Modell separat speichern (optional) ----
# os.makedirs("models", exist_ok=True)
# torch.save(generation_model.state_dict(), "models/ode_v1.pth")

# ---- Plot ----
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1).numpy(), y_test.detach().view(-1).numpy())
plt.savefig("plots/sine.png")
plt.savefig("plots/sine.pdf")
plt.close()
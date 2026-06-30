import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Repo-Root finden (das Verzeichnis, das utils.py enthaelt) und auf den
# Importpfad legen -- so funktioniert `from utils import ...` unabhaengig davon,
# von wo das Skript gestartet wird (z.B. `python data/sine_generator.py`).
_here = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in (_here.parent, *_here.parents) if (p / "utils.py").is_file()),
    _here.parent,
)
sys.path.insert(0, str(REPO_ROOT))

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
# Keys MUESSEN x, y, x_test, y_test heissen (load_dataset-Vertrag). Extra-Keys
# wie meta_data werden ignoriert. Endung .pt, damit `--dataset sine` greift.
# Ausgaben werden relativ zum Repo-Root abgelegt, nicht zum Arbeitsverzeichnis.
data_dir = REPO_ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "x": x,
        "y": y,
        "x_test": x_test,
        "y_test": y_test,
        "meta_data": {"generation_model": "SineTarget"},
    },
    data_dir / "sine_dataset.pt",
)
print(f"Saved dataset -> {data_dir / 'sine_dataset.pt'}")

# ---- Modell separat speichern (optional) ----
# models_dir = REPO_ROOT / "models"
# models_dir.mkdir(parents=True, exist_ok=True)
# torch.save(generation_model.state_dict(), models_dir / "ode_v1.pth")

# ---- Plot ----
plots_dir = REPO_ROOT / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(12, 8))
plt.plot(x_test.view(-1).numpy(), y_test.detach().view(-1).numpy())
plt.savefig(plots_dir / "ode_v1.png")
plt.savefig(plots_dir / "ode_v1.pdf")
plt.close()
print(f"Saved plots  -> {plots_dir}")
"""Regime-2 dataset generator: gated compositional targets.

Regime 2 = targets that are natural for a compositional grammar (linear pieces
combined through gates, like the trapezoid) but NOT sparse in the flat
polynomial/trigonometric libraries that SINDy-style sparse regression uses.
On such targets a flat-library fit can match the train domain yet break under
extrapolation -- exactly what the recovery criterion (recovery.py) detects.

Targets (all 1D -> 1D, so they run on the EXISTING pipeline targets with
input/output Literal(1); no damg_targets change needed):

  gate_switch : sigmoid gate blends two affine regimes
                  g(x) * (a1*x + b1) + (1 - g(x)) * (a2*x + b2)
  double_gate : two gates carve out a middle regime (trapezoid family, but
                with sloped shoulders instead of a flat plateau)

Usage:
  python data/regime2_generator.py --target gate_switch
  python data/regime2_generator.py --target double_gate --eps 1e-2
Then run the pipeline with e.g. `--dataset gate_switch`.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

_here = Path(__file__).resolve()
REPO_ROOT = next(
    (p for p in (_here.parent, *_here.parents) if (p / "utils.py").is_file()),
    _here.parent,
)
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
#  Gated compositional targets (deterministic, fixed parameters)
# ============================================================
class GateSwitch(nn.Module):
    """One sigmoid gate blending two affine regimes at x = 0."""

    def __init__(self, sharpness=2.0, a1=1.5, b1=2.0, a2=-0.5, b2=-1.0):
        super().__init__()
        self.k, self.a1, self.b1, self.a2, self.b2 = sharpness, a1, b1, a2, b2

    def forward(self, x):                                  # x: (N, 1)
        g = torch.sigmoid(self.k * x)
        return g * (self.a1 * x + self.b1) + (1 - g) * (self.a2 * x + self.b2)


class DoubleGate(nn.Module):
    """Two gates at +/-c select a middle regime; shoulders keep their own slopes."""

    def __init__(self, sharpness=3.0, c=4.0,
                 a_left=-1.0, b_left=-2.0, a_mid=0.5, b_mid=3.0,
                 a_right=1.2, b_right=-4.0):
        super().__init__()
        self.k, self.c = sharpness, c
        self.left = (a_left, b_left)
        self.mid = (a_mid, b_mid)
        self.right = (a_right, b_right)

    def forward(self, x):                                  # x: (N, 1)
        g_lo = torch.sigmoid(self.k * (x + self.c))        # off left of -c
        g_hi = torch.sigmoid(self.k * (x - self.c))        # on  right of +c
        f_left = self.left[0] * x + self.left[1]
        f_mid = self.mid[0] * x + self.mid[1]
        f_right = self.right[0] * x + self.right[1]
        return (1 - g_lo) * f_left + g_lo * (1 - g_hi) * f_mid + g_hi * f_right


TARGETS = {"gate_switch": GateSwitch, "double_gate": DoubleGate}

TRAIN_RANGE = (-10.0, 10.0)
TEST_RANGE = (-15.0, 15.0)   # wider: extrapolation, mirroring the repo setup


def generate(model, n_samples, lo, hi, eps, seed):
    """Uniform grid inputs, truth + Gaussian noise. Shapes (N,1) / (N,)."""
    x = torch.linspace(lo, hi, n_samples).view(-1, 1)
    y = model(x).detach().view(-1)
    if eps > 0:
        g = torch.Generator().manual_seed(seed)
        y = y + torch.randn(y.shape, generator=g) * (eps ** 0.5)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), default="gate_switch")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="Variance of additive Gaussian noise")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = TARGETS[args.target]()
    x, y = generate(model, args.n_samples, *TRAIN_RANGE, args.eps, args.seed)
    x_test, y_test = generate(model, args.n_samples, *TEST_RANGE, args.eps,
                              args.seed + 1)
    print(f"Train: x={tuple(x.shape)} y={tuple(y.shape)} | "
          f"Test: x={tuple(x_test.shape)} y={tuple(y_test.shape)}")

    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{args.target}_dataset.pt"
    torch.save(
        {
            "x": x,
            "y": y,
            "x_test": x_test,
            "y_test": y_test,
            "meta_data": {
                "generation_model": type(model).__name__,
                "regime": "2 (gated compositional, not library-sparse)",
                "train_range": list(TRAIN_RANGE),
                "test_range": list(TEST_RANGE),
                "eps": args.eps,
                "seed": args.seed,
            },
        },
        out_path,
    )
    print(f"Saved dataset -> {out_path}   (use: --dataset {args.target})")

    # ---- Plot ----
    plots_dir = REPO_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    xs = torch.linspace(*TEST_RANGE, 1000).view(-1, 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs.view(-1), model(xs).detach().view(-1), lw=2, label="truth")
    ax.scatter(x.view(-1)[::20], y[::20], s=8, alpha=0.5, label="train samples")
    for b in TRAIN_RANGE:
        ax.axvline(b, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title(f"Regime-2 target: {args.target}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{args.target}.png")
    plt.close(fig)
    print(f"Saved plot   -> {plots_dir / f'{args.target}.png'}")


if __name__ == "__main__":
    main()
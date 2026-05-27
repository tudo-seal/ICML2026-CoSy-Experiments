"""
Kernel Alignment Analysis for Program Search Spaces
===================================================

This script loads experimental CSV results and produces
publication-quality figures and tables for the ICML paper.

Goals of the analysis
---------------------

1. Measure kernel–objective alignment
   → does structural similarity predict objective similarity?

2. Measure surrogate learnability
   → does higher alignment lead to better GP surrogates?

3. Evaluate search space synthesis
   → do synthesized subspaces improve kernel alignment?

The resulting figures support the experimental claims of the paper.

Expected structure of the input directory
-----------------------------------------

results_1.csv
results_2.csv
...
results_n.csv

Each CSV contains kernel evaluation results.

Outputs
-------

plots_output/

    fig1_kernel_alignment_comparison.pdf
    fig2_alignment_vs_surrogate.pdf
    fig3_alignment_per_target.pdf
    fig4_global_vs_refined_alignment.pdf
    fig5_alignment_distribution.pdf
    fig6_surrogate_r2.pdf
    fig7_variance_vs_alignment.pdf

    kernel_summary.csv
    kernel_table.tex
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    csv_dir: str
    output_dir: str = "plots_output"

    kernels: List[str] = (
        "tree_kernel_1",
        "wl_kernel_1",
        #"noisy_hybrid_kernel",
        "damg_kernel_1",
    )

    kernel_names: dict = None

    def __post_init__(self):
        if self.kernel_names is None:
            self.kernel_names = {
                "tree_kernel_1": "Tree",
                "wl_kernel_1": "WL",
                #"noisy_hybrid_kernel": "Hybrid",
                "damg_kernel_1": "DAMG",
            }


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_results(config: Config) -> pd.DataFrame:
    """
    Load all results_*.csv files.

    Returns
    -------
    DataFrame containing concatenated experimental results.
    """

    pattern = os.path.join(config.csv_dir, "results_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError(f"No CSV files found in {config.csv_dir}")

    logging.info("Loading %d CSV files", len(files))

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        df["run"] = os.path.basename(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    logging.info("Loaded %d rows", len(data))

    return data


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(data: pd.DataFrame, config: Config) -> pd.DataFrame:

    data = data[data["kernel"].isin(config.kernels)].copy()
    data["kernel_pretty"] = data["kernel"].map(config.kernel_names)

    return data


# ============================================================
# PLOT UTILITIES
# ============================================================

def save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logging.info("Saved %s", path)


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# FIGURE 1
# Kernel–Objective Alignment
# ============================================================

def plot_kernel_alignment(data: pd.DataFrame, config: Config):

    """
    Interpretation for the paper
    ----------------------------

    This plot measures Spearman correlation between:

        kernel similarity k(x_i,x_j)

    and

        objective similarity -|f(x_i)-f(x_j)|

    Higher values indicate that kernel similarity
    provides useful information about the objective landscape.

    Expected result:

        Tree < WL < Hybrid < DAMG
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    # Calculate mean and SEM for each kernel
    grouped = data.groupby("kernel_pretty")["kernel_objective_spearman_correlation"]
    means = grouped.mean()
    sems = grouped.sem()

    x_pos = np.arange(len(means))
    ax.bar(x_pos, means.values, yerr=sems.values, capsize=5, alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Kernel–Objective Alignment (Spearman ρ)")
    ax.set_xlabel("Kernel")
    ax.set_title("Kernel–Objective Alignment")

    save(fig, os.path.join(config.output_dir,
                           "fig1_kernel_alignment_comparison.pdf"))


# ============================================================
# FIGURE 2
# Alignment vs Surrogate Learnability
# ============================================================

def plot_alignment_vs_surrogate(data: pd.DataFrame, config: Config):

    """
    Paper interpretation
    --------------------

    If kernel–objective alignment is meaningful,
    we expect a positive relationship with surrogate performance.

    x-axis : alignment
    y-axis : GP R²
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    # Create scatter plot with different colors per kernel
    kernels = data["kernel_pretty"].unique()
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / len(kernels)) for i in range(len(kernels))]

    for i, kernel in enumerate(kernels):
        subset = data[data["kernel_pretty"] == kernel]
        ax.scatter(
            subset["kernel_objective_spearman_correlation"],
            subset["surrogate_r2_mean"],
            label=kernel,
            s=80,
            alpha=0.6,
            color=colors[i]
        )

    # Add regression line
    x = data["kernel_objective_spearman_correlation"].values
    y = data["surrogate_r2_mean"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "k-", linewidth=2, label="Linear fit")

    ax.set_xlabel("Kernel–Objective Alignment")
    ax.set_ylabel("Surrogate Performance (R²)")
    ax.set_title("Alignment vs Surrogate Learnability")
    ax.legend()

    save(fig, os.path.join(config.output_dir,
                           "fig2_alignment_vs_surrogate.pdf"))


# ============================================================
# FIGURE 3
# Alignment per Target
# ============================================================

def plot_alignment_per_target(data: pd.DataFrame, config: Config):

    """
    Shows that results are consistent across tasks.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    # Get unique targets and kernels
    targets = data["target"].unique()
    kernels = data["kernel_pretty"].unique()
    x = np.arange(len(targets))
    width = 0.2
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / len(kernels)) for i in range(len(kernels))]

    # Plot bars for each kernel
    for i, kernel in enumerate(kernels):
        subset = data[data["kernel_pretty"] == kernel]
        means = [subset[subset["target"] == t]["kernel_objective_spearman_correlation"].mean() for t in targets]
        ax.bar(x + i * width, means, width, label=kernel, color=colors[i], alpha=0.8)

    ax.set_xlabel("Target")
    ax.set_ylabel("Kernel–Objective Alignment")
    ax.set_title("Alignment across Targets")
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets)
    ax.legend()

    save(fig, os.path.join(config.output_dir,
                           "fig3_alignment_per_target.pdf"))


# ============================================================
# FIGURE 4
# Search Space Synthesis Effect
# ============================================================

def plot_global_vs_refined(data: pd.DataFrame, config: Config):

    """
    This plot demonstrates the key claim of the paper:

        search space synthesis improves kernel alignment.
    """

    damg = data[data["kernel"] == "damg_kernel_1"].copy()

    damg["space_type"] = damg["target"].apply(
        lambda x: "refined" if "refined" in x else "global"
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    # Calculate mean and SEM for each space type
    grouped = damg.groupby("space_type")["kernel_objective_spearman_correlation"]
    means = grouped.mean()
    sems = grouped.sem()

    x_pos = np.arange(len(means))
    ax.bar(x_pos, means.values, yerr=sems.values, capsize=5, alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Kernel–Objective Alignment")
    ax.set_xlabel("Search Space")
    ax.set_title("Effect of Search Space Synthesis")

    save(fig, os.path.join(config.output_dir,
                           "fig4_global_vs_refined_alignment.pdf"))


# ============================================================
# FIGURE 5
# Alignment Distribution
# ============================================================

def plot_alignment_distribution(data: pd.DataFrame, config: Config):

    fig, ax = plt.subplots(figsize=(6, 4))

    # Prepare data for violin plot
    kernels = data["kernel_pretty"].unique()
    positions = np.arange(len(kernels))
    data_to_plot = [data[data["kernel_pretty"] == k]["kernel_objective_spearman_correlation"].values for k in kernels]

    parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Kernel–Objective Alignment")
    ax.set_xlabel("Kernel")

    save(fig, os.path.join(config.output_dir,
                           "fig5_alignment_distribution.pdf"))


# ============================================================
# FIGURE 6
# Surrogate R²
# ============================================================

def plot_surrogate_r2(data: pd.DataFrame, config: Config):

    fig, ax = plt.subplots(figsize=(6, 4))

    # Calculate mean and SEM for each kernel
    grouped = data.groupby("kernel_pretty")["surrogate_r2_mean"]
    means = grouped.mean()
    sems = grouped.sem()

    x_pos = np.arange(len(means))
    ax.bar(x_pos, means.values, yerr=sems.values, capsize=5, alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Surrogate Performance (R²)")
    ax.set_xlabel("Kernel")

    save(fig, os.path.join(config.output_dir,
                           "fig6_surrogate_r2.pdf"))


# ============================================================
# FIGURE 7
# Search Space Variance vs Alignment
# ============================================================

def plot_variance_vs_alignment(data: pd.DataFrame, config: Config):

    """
    Tests hypothesis:

        higher structural variance
        → lower kernel alignment
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    # Create scatter plot with different colors per kernel
    kernels = data["kernel_pretty"].unique()
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i / len(kernels)) for i in range(len(kernels))]

    for i, kernel in enumerate(kernels):
        subset = data[data["kernel_pretty"] == kernel]
        ax.scatter(
            np.log10(subset["kernel_condition_number"]),
            subset["kernel_objective_spearman_correlation"],
            label=kernel,
            s=80,
            alpha=0.6,
            color=colors[i]
        )

    ax.set_xlabel("Search Space Variance (log condition number)")
    ax.set_ylabel("Kernel–Objective Alignment")
    ax.legend()

    save(fig, os.path.join(config.output_dir,
                           "fig7_variance_vs_alignment.pdf"))


# ============================================================
# TABLE GENERATION
# ============================================================

def generate_tables(data: pd.DataFrame, config: Config):

    summary = data.groupby("kernel_pretty").agg({
        "kernel_objective_spearman_correlation": ["mean", "std"],
        "surrogate_r2_mean": ["mean", "std"],
        "surrogate_normalized_mse_mean": ["mean", "std"],
    })

    summary.columns = [
        "alignment_mean", "alignment_std",
        "r2_mean", "r2_std",
        "nmse_mean", "nmse_std"
    ]

    summary = summary.reset_index()

    summary.to_csv(os.path.join(config.output_dir,
                                "kernel_summary.csv"),
                   index=False)

    def fmt(m, s):
        return f"{m:.3f} $\\pm$ {s:.3f}"

    rows = []

    for _, r in summary.iterrows():
        rows.append(
            f"{r['kernel_pretty']} & "
            f"{fmt(r['alignment_mean'], r['alignment_std'])} & "
            f"{fmt(r['r2_mean'], r['r2_std'])} & "
            f"{fmt(r['nmse_mean'], r['nmse_std'])} \\\\"
        )

    latex = r"""
\begin{table}[t]
\centering
\caption{Kernel comparison results.}
\begin{tabular}{lccc}
\toprule
Kernel & Alignment ($\rho$) & Surrogate $R^2$ & NMSE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    with open(os.path.join(config.output_dir,
                           "kernel_table.tex"), "w") as f:
        f.write(latex)

    logging.info("Generated summary tables")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    # Determine repository root (relative path handling)
    # Current file: .../cosy-examples/bayesian_optimization/examples/ODEs/plot_new_kernel_experiments.py
    # Need to go up 4 levels to reach cosy-examples/
    current_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    csv_dir = os.path.join(repo_root, "bayesian_optimization", "examples", "ODEs", "csv")

    config = Config(csv_dir=csv_dir)

    ensure_output_dir(config.output_dir)

    data = load_results(config)
    data = preprocess(data, config)

    plot_kernel_alignment(data, config)
    plot_alignment_vs_surrogate(data, config)
    plot_alignment_per_target(data, config)
    plot_global_vs_refined(data, config)

    plot_alignment_distribution(data, config)
    plot_surrogate_r2(data, config)
    plot_variance_vs_alignment(data, config)

    generate_tables(data, config)

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()

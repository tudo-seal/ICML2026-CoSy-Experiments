"""
ICML-ready plot generation for Bayesian optimization experiments.

The script discovers CSV files in the local `csv/` directory and only loads
files that match the expected naming scheme:

    bo_trace_target_len_{n}[_{global|refinement}]_run_{i}.csv
    random_trace_target_len_{n}[_{global|refinement}]_run_{i}.csv
    bo_ranking_target_len_{n}[_{global|refinement}]_run_{i}.csv

The optional `_{global|refinement}` segment is supported because some legacy
files omit it and should be interpreted as global runs.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = SCRIPT_DIR / "csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots_output"

FILENAME_PATTERN = re.compile(
    r"^(?P<kind>bo_trace|random_trace|bo_ranking)"
    r"_target_len_(?P<target_len>\d+)"
    r"(?:_(?P<space>global|refinement))?"
    r"_run_(?P<run>\d+)\.csv$"
)

KIND_TO_LABEL = {
    "bo_trace": "BO",
    "random_trace": "Random",
    "bo_ranking": "BO",
}

METHOD_ORDER = ["BO", "Random"]
SPACE_ORDER = ["global", "refinement"]


@dataclass(frozen=True)
class Config:
    csv_dir: Path = DEFAULT_CSV_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR


# ---------------------------------------------------------------------------
# Logging and style
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (5, 4),
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

def parse_filename(file_path: Path) -> dict[str, object] | None:
    """Extract metadata from an experiment CSV filename."""

    match = FILENAME_PATTERN.match(file_path.name)
    if match is None:
        return None

    space = match.group("space") or "global"

    return {
        "kind": match.group("kind"),
        "target_len": int(match.group("target_len")),
        "space": space,
        "run": int(match.group("run")),
    }


def discover_experiment_files(csv_dir: Path) -> list[Path]:
    """Return all CSV files in `csv_dir` that match the naming scheme."""

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")

    files: list[Path] = []
    for file_path in sorted(csv_dir.glob("*.csv")):
        if parse_filename(file_path) is not None:
            files.append(file_path)

    return files


def load_experiment_data(config: Config) -> pd.DataFrame:
    """Load and annotate all matching experiment CSV files."""

    files = discover_experiment_files(config.csv_dir)
    if not files:
        raise RuntimeError(
            f"No experiment CSV files matching the expected naming scheme were found in {config.csv_dir}."
        )

    logging.info("Loading %d experiment CSV files", len(files))

    datasets: list[pd.DataFrame] = []
    for file_path in files:
        meta = parse_filename(file_path)
        if meta is None:
            continue

        df = pd.read_csv(file_path)
        for key, value in meta.items():
            df[key] = value
        df["method_label"] = df["kind"].map(KIND_TO_LABEL)
        df["source_file"] = file_path.name
        datasets.append(df)

    data = pd.concat(datasets, ignore_index=True)

    # Normalize types used for plotting and sorting.
    data["evaluation"] = pd.to_numeric(data["evaluation"], errors="coerce")
    data["target_len"] = pd.to_numeric(data["target_len"], errors="coerce").astype("Int64")
    data["run"] = pd.to_numeric(data["run"], errors="coerce").astype("Int64")
    data["space"] = pd.Categorical(data["space"], categories=SPACE_ORDER, ordered=True)
    data["method_label"] = pd.Categorical(data["method_label"], categories=METHOD_ORDER, ordered=True)

    logging.info("Loaded %d rows", len(data))
    logging.info(
        "Discovered files: %s",
        ", ".join(file.name for file in files[:6]) + (" ..." if len(files) > 6 else ""),
    )

    return data


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def final_rows(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    """Return the last row per group after sorting by evaluation."""

    sort_cols = list(group_cols) + ["evaluation"]
    ordered = df.sort_values(sort_cols)
    return ordered.groupby(list(group_cols), as_index=False).tail(1)


def aggregate_curve(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    hue_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """Aggregate a curve as mean and standard error.

    If `hue_col` is provided, the result is a pair of pivot tables
    (means, standard errors). Otherwise a single summary DataFrame is returned.
    """

    columns = [x_col, value_col] + ([hue_col] if hue_col is not None else [])
    working = df[columns].copy()
    working = working.dropna(subset=[x_col, value_col])
    working[x_col] = pd.to_numeric(working[x_col], errors="coerce")
    working = working.dropna(subset=[x_col])

    if hue_col is None:
        summary = (
            working.groupby(x_col)[value_col]
            .agg(mean="mean", std="std", sem="sem", count="count")
            .sort_index()
        )
        summary[["std", "sem"]] = summary[["std", "sem"]].fillna(0.0)
        return summary

    summary = (
        working.groupby([x_col, hue_col])[value_col]
        .agg(mean="mean", std="std", sem="sem", count="count")
        .reset_index()
    )
    summary[["std", "sem"]] = summary[["std", "sem"]].fillna(0.0)

    means = summary.pivot(index=x_col, columns=hue_col, values="mean").sort_index()
    sems = summary.pivot(index=x_col, columns=hue_col, values="sem").sort_index().fillna(0.0)
    return means, sems


def set_paper_style(ax: plt.Axes, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply a consistent paper-friendly axis style."""

    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure and close it immediately."""

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_mean_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    label: str,
    color: str,
    line_style: str = "-",
) -> None:
    """Draw a line with an optional shaded uncertainty band."""

    mask = ~np.isnan(y)
    if yerr is not None:
        mask &= ~np.isnan(yerr)

    if not np.any(mask):
        return

    x = x[mask]
    y = y[mask]
    if yerr is not None:
        yerr = np.nan_to_num(yerr[mask], nan=0.0)

    ax.plot(x, y, linestyle=line_style, linewidth=2.2, label=label, color=color)
    if yerr is not None:
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.18, linewidth=0)


# ---------------------------------------------------------------------------
# Figure 1: BO vs Random progress
# ---------------------------------------------------------------------------

def plot_progress(df: pd.DataFrame, config: Config) -> None:
    trace = df[df["kind"].isin(["bo_trace", "random_trace"])].copy()

    means, sems = aggregate_curve(trace, "evaluation", "best_objective_value", hue_col="method_label")
    if not isinstance(means, pd.DataFrame):
        raise TypeError("Expected a DataFrame for method-based curve aggregation.")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = {"BO": "#1f77b4", "Random": "#ff7f0e"}

    x = means.index.to_numpy(dtype=float)
    for method in METHOD_ORDER:
        if method not in means.columns:
            continue
        plot_mean_curve(
            ax,
            x,
            means[method].to_numpy(dtype=float),
            sems[method].to_numpy(dtype=float),
            label=method,
            color=colors[method],
        )

    set_paper_style(
        ax,
        title="Optimization progress",
        xlabel="Function evaluations",
        ylabel="Best objective value",
    )
    ax.legend(frameon=False)

    save_figure(fig, config.output_dir / "bo_vs_random_progress.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Regret curve
# ---------------------------------------------------------------------------

def plot_regret(df: pd.DataFrame, config: Config) -> None:
    trace = df[df["kind"].isin(["bo_trace", "random_trace"])].copy()
    if trace.empty:
        logging.warning("Skipping regret plot because no trace data is available.")
        return

    global_best = trace["best_objective_value"].min()
    trace["simple_regret"] = trace["best_objective_value"] - global_best

    means, sems = aggregate_curve(trace, "evaluation", "simple_regret", hue_col="method_label")
    if not isinstance(means, pd.DataFrame):
        raise TypeError("Expected a DataFrame for method-based curve aggregation.")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = {"BO": "#1f77b4", "Random": "#ff7f0e"}

    x = means.index.to_numpy(dtype=float)
    for method in METHOD_ORDER:
        if method not in means.columns:
            continue
        plot_mean_curve(
            ax,
            x,
            means[method].to_numpy(dtype=float),
            sems[method].to_numpy(dtype=float),
            label=method,
            color=colors[method],
        )

    set_paper_style(
        ax,
        title="Simple regret",
        xlabel="Function evaluations",
        ylabel="Best objective value - global best",
    )
    ax.legend(frameon=False)

    save_figure(fig, config.output_dir / "bo_vs_random_regret.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Final performance distribution
# ---------------------------------------------------------------------------

def plot_final_distribution(df: pd.DataFrame, config: Config) -> None:
    trace = df[df["kind"].isin(["bo_trace", "random_trace"])].copy()
    if trace.empty:
        logging.warning("Skipping final distribution plot because no trace data is available.")
        return

    final = final_rows(trace, ["kind", "run"]).copy()
    final["simple_regret"] = final["best_objective_value"] - trace["best_objective_value"].min()

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    colors = {"BO": "#1f77b4", "Random": "#ff7f0e"}

    box_data = []
    labels = []
    for method in METHOD_ORDER:
        values = final.loc[final["method_label"] == method, "best_objective_value"].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        box_data.append(values)
        labels.append(method)

    if not box_data:
        logging.warning("Skipping final distribution plot because no valid values were found.")
        return

    box = ax.boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[label])

    set_paper_style(
        ax,
        title="Final optimization performance",
        xlabel="Method",
        ylabel="Final best objective value",
    )

    save_figure(fig, config.output_dir / "bo_final_distribution.pdf")


# ---------------------------------------------------------------------------
# Figure 4: GP ranking quality
# ---------------------------------------------------------------------------

def plot_ranking(df: pd.DataFrame, config: Config) -> None:
    ranking = df[df["kind"] == "bo_ranking"].copy()
    ranking = ranking.dropna(subset=["spearman_rho"])

    if ranking.empty:
        logging.warning("Skipping ranking plot because no ranking data is available.")
        return

    curve = aggregate_curve(ranking, "evaluation", "spearman_rho")
    if not isinstance(curve, pd.DataFrame):
        raise TypeError("Expected a DataFrame for single-series curve aggregation.")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = curve.index.to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    yerr = curve["sem"].to_numpy(dtype=float)

    plot_mean_curve(
        ax,
        x,
        y,
        yerr,
        label="Spearman ρ",
        color="#2ca02c",
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    set_paper_style(
        ax,
        title="GP ranking quality",
        xlabel="Function evaluations",
        ylabel="Spearman correlation",
    )
    ax.legend(frameon=False)

    save_figure(fig, config.output_dir / "gp_ranking_quality.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Acquisition quality
# ---------------------------------------------------------------------------

def plot_acquisition(df: pd.DataFrame, config: Config) -> None:
    trace = df[(df["kind"] == "bo_trace") & df["ei_ratio"].notna()].copy()

    if trace.empty:
        logging.warning("Skipping acquisition plot because no EI ratio values are available.")
        return

    curve = aggregate_curve(trace, "evaluation", "ei_ratio")
    if not isinstance(curve, pd.DataFrame):
        raise TypeError("Expected a DataFrame for single-series curve aggregation.")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    x = curve.index.to_numpy(dtype=float)
    y = curve["mean"].to_numpy(dtype=float)
    yerr = curve["sem"].to_numpy(dtype=float)

    plot_mean_curve(
        ax,
        x,
        y,
        yerr,
        label="EI ratio",
        color="#d62728",
    )
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="Parity")

    set_paper_style(
        ax,
        title="Acquisition optimization quality",
        xlabel="Function evaluations",
        ylabel="EI ratio",
    )
    ax.legend(frameon=False)

    save_figure(fig, config.output_dir / "acquisition_quality.pdf")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def generate_latex_tables(df: pd.DataFrame, config: Config) -> None:
    trace = df[df["kind"].isin(["bo_trace", "random_trace"])].copy()
    if trace.empty:
        logging.warning("Skipping LaTeX tables because no trace data is available.")
        return

    final = final_rows(trace, ["kind", "run"]).copy()
    global_best = trace["best_objective_value"].min()
    final["simple_regret"] = final["best_objective_value"] - global_best

    trace_summary = (
        final.groupby("method_label", observed=True)
        .agg(
            final_best_mean=("best_objective_value", "mean"),
            final_best_std=("best_objective_value", "std"),
            simple_regret_mean=("simple_regret", "mean"),
            simple_regret_std=("simple_regret", "std"),
            runs=("run", "count"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
        .rename(columns={"method_label": "method"})
    )
    trace_summary[["final_best_std", "simple_regret_std"]] = trace_summary[["final_best_std", "simple_regret_std"]].fillna(0.0)

    table = pd.DataFrame(
        {
            "Method": trace_summary["method"],
            "Final best objective": [
                format_mean_std(m, s)
                for m, s in zip(trace_summary["final_best_mean"], trace_summary["final_best_std"], strict=False)
            ],
            "Simple regret": [
                format_mean_std(m, s)
                for m, s in zip(trace_summary["simple_regret_mean"], trace_summary["simple_regret_std"], strict=False)
            ],
            "Runs": trace_summary["runs"].astype(int),
        }
    )

    table_latex = table.to_latex(index=False, escape=False)
    with open(config.output_dir / "table_bo_vs_random.tex", "w", encoding="utf-8") as f:
        f.write(table_latex)

    ranking = df[df["kind"] == "bo_ranking"].copy().dropna(subset=["spearman_rho", "kendall_tau"])
    if ranking.empty:
        logging.warning("Skipping ranking table because no ranking data is available.")
        return

    ranking_final = final_rows(ranking, ["run"]).copy()
    ranking_summary = {
        "spearman_mean": float(ranking_final["spearman_rho"].mean()),
        "spearman_std": float(ranking_final["spearman_rho"].std(ddof=0)),
        "kendall_mean": float(ranking_final["kendall_tau"].mean()),
        "kendall_std": float(ranking_final["kendall_tau"].std(ddof=0)),
    }

    ranking_table = pd.DataFrame(
        {
            "Metric": ["Spearman ρ", "Kendall τ"],
            "Final value": [
                format_mean_std(ranking_summary["spearman_mean"], ranking_summary["spearman_std"]),
                format_mean_std(ranking_summary["kendall_mean"], ranking_summary["kendall_std"]),
            ],
        }
    )

    table_latex = ranking_table.to_latex(index=False, escape=False)
    with open(config.output_dir / "table_gp_ranking.tex", "w", encoding="utf-8") as f:
        f.write(table_latex)

    logging.info("Generated summary tables")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate publication-ready BO plots from experiment CSV files.")
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="Directory containing experiment CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots and tables.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = Config(csv_dir=args.csv_dir, output_dir=args.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(config)

    plot_progress(data, config)
    plot_regret(data, config)
    plot_final_distribution(data, config)
    plot_ranking(data, config)
    plot_acquisition(data, config)

    generate_latex_tables(data, config)

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()

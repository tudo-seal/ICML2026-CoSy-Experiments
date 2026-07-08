"""Quick viewer for bo_cli results.

Scans results/<target>_<kernel>_<seed>__<timestamp>/ folders, reads the trace
CSVs and prints one summary row per run: final best objective, evaluations,
recovery verdict (if the recovery hook wrote its columns) and the best
candidate term. Optionally plots best-so-far progress curves.

Usage:
  python show_results.py                          # table for ./results
  python show_results.py --results-root results   # explicit root
  python show_results.py --best 5                 # also print top-5 candidate terms
  python show_results.py --plot                   # save progress.png per target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_runs(results_root: Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(results_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("aggregated"):
            continue
        for fname, method in (("bo_trace.csv", "BO"), ("random_trace.csv", "Random")):
            path = folder / fname
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:                     # noqa: BLE001
                print(f"[warn] unreadable {path}: {exc}")
                continue
            if df.empty:
                continue
            last = df.sort_values("evaluation").iloc[-1]
            rows.append({
                "folder": folder.name,
                "method": method,
                "target": last.get("target_name", "?"),
                "kernel": last.get("kernel_name", "?"),
                "seed": last.get("seed", "?"),
                "evals": int(last["evaluation"]),
                "best_objective": float(last["best_objective_value"]),
                "recovered": last.get("recovered", float("nan")),
                "nmse_extrap": last.get("recovery_nmse_extrap", float("nan")),
                "complexity": last.get("best_complexity", float("nan")),
                "best_candidate": str(last.get("best_candidate", ""))[:90],
                "_trace": df,
            })
    if not rows:
        raise SystemExit(f"No run folders with trace CSVs found under {results_root}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--best", type=int, default=0,
                    help="Print the N best runs' full candidate terms")
    ap.add_argument("--plot", action="store_true",
                    help="Save best-so-far progress curves per target")
    args = ap.parse_args()

    runs = load_runs(Path(args.results_root))
    table = (runs.drop(columns=["_trace"])
                 .sort_values(["target", "kernel", "method", "seed"]))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 92)
    print(f"\n{len(table)} runs under {args.results_root}:\n")
    print(table.drop(columns=["folder"]).to_string(index=False,
          float_format=lambda v: f"{v:.4g}"))

    agg = (table.groupby(["target", "kernel", "method"])
                .agg(runs=("seed", "count"),
                     best_mean=("best_objective", "mean"),
                     best_min=("best_objective", "min"),
                     recovery_rate=("recovered", "mean"))
                .reset_index())
    print("\nAggregated over seeds:\n")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.}"))

    if args.best:
        print(f"\nTop {args.best} runs by best objective:\n")
        top = table.nsmallest(args.best, "best_objective")
        for _, r in top.iterrows():
            print(f"  [{r['method']}] {r['target']} / {r['kernel']} / seed {r['seed']} "
                  f"-> {r['best_objective']:.}\n      {r['best_candidate']}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = Path(args.results_root) / "plots_quicklook"
        out_dir.mkdir(exist_ok=True)
        for target, group in runs.groupby("target"):
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for _, r in group.iterrows():
                tr = r["_trace"].sort_values("evaluation")
                ax.plot(tr["evaluation"], tr["best_objective_value"], alpha=0.7,
                        label=f"{r['method']} {r['kernel']} s{r['seed']}")
            ax.set_xlabel("evaluation")
            ax.set_ylabel("best objective value")
            ax.set_title(str(target))
            ax.set_yscale("log")
            ax.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / f"progress_{target}.png")
            plt.close(fig)
        print(f"\nProgress plots -> {out_dir}/")


if __name__ == "__main__":
    main()
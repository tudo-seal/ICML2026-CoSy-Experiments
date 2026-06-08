# Companion code for "Search Space Synthesis for Parametric Functions" (ICML 2026)

This repository contains the companion code for the ICML 2026 paper
"Search Space Synthesis for Parametric Functions" (poster reference: https://icml.cc/virtual/2026/poster/62530).

Summary
-------
This project implements the experimental pipeline used in the paper: it
provides search-space synthesis utilities, evolutionary operators, and a
Bayesian Optimization (BO) runner used to compare search strategies on
synthetic ODE modelling tasks. The repository also includes the datasets and
CSV artifacts produced for the paper under `data/` and `csv/` (where present).

Key experiment entry points
---------------------------
- `bo_cli.py` — command-line interface to run the experiments, aggregate
  results and generate plots. This is the primary entry point for reproducing
  the BO experiments from the paper.
- `kernel_experiments.py` — standalone kernel-analysis script. It measures
  kernel-objective alignment and surrogate learnability for the DAMG/ODE
  search spaces used in the paper. Run it directly, e.g.:
  ```bash
  python3 kernel_experiments.py --mode both --targets target_len_5 --n-samples 10
  ```
  Use `--mode domain`, `--mode surrogate`, or `--mode both` to control the
  analysis pass.
- `best_candidate_found.py` — self-contained script that reproduces the best
  candidate experiment from Appendix C. It constructs the one-off search space,
  identifies the unique best candidate, retrains the corresponding model, and
  writes the illustrative plots.

Notes on the BO implementation
------------------------------
The BO implementation in this repository uses an Ask/Tell-style interface
compatible with the `cosy-examples` Ask/Tell refactor. The CLI and runner were
recently refactored: the BO loop now constructs an optimizer exposing
`initialize(x0, y0)`, `suggest()` and `observe(x, y)` and instruments runtime
and ranking diagnostics around those calls. Any earlier experimental variant
called "RefinedBO" has been removed from the CLI surface — the core code in
`bo_runner.py` is Ask/Tell-compatible and intended to interoperate with the
updated `cosy-examples` implementations.

Quick start
-----------
Run the minimal smoke test (fast, low-budget) to verify your environment and
that the CLI works on your machine:

```bash
python3 bo_cli.py run --config configs/minimal_test.json
```

This creates timestamped per-run folders under the `results/` path specified in
the config and writes per-iteration CSVs (`*_trace.csv`, `*_ranking.csv`) and
a `manifest.json` describing run parameters.

Reproducing paper experiments
-----------------------------
The repository includes configuration files in `configs/` that were used to
generate the experiment suites reported in the paper. To reproduce a prepared
experiment, call `bo_cli.py run --config <path-to-config.json>` from the
project root. Use `aggregate` to combine CSVs from multiple runs and `plot` to
generate figures (plotting functions may require matplotlib/numpy).

For the kernel-analysis experiments described in the paper, invoke
`kernel_experiments.py` directly as shown above. For the Appendix C
reproduction, run `best_candidate_found.py` directly from the repository root:

```bash
python3 best_candidate_found.py
```

This script constructs the compact search space used for the appendix example,
locates the unique best candidate, and recreates the accompanying plots.

Repository contents
-------------------
- `bo_cli.py` — primary CLI for running and aggregating experiments
- `bo_runner.py` — BO runner and instrumentation (Ask/Tell loop)
- `bo_plotting.py` — helper plot functions (used by `plot` subcommand)
- `kernel_experiments.py` — kernel diagnostics and surrogate learnability
- `best_candidate_found.py` — Appendix C reproduction script
- `configs/` — JSON experiment definitions (including `minimal_test.json`)
- `results/` — runtime output (created by `bo_cli.py`)
- `data/`, `csv/` — (when present) datasets and CSV artifacts produced for
  experiments reported in the paper

Paper reference
---------------
ICML 2026 poster (temporary link): https://icml.cc/virtual/2026/poster/62530

License and citation
--------------------
This repository is the companion code to the ICML 2026 paper. If you use
these artifacts in your research, please cite the paper. Update the poster
link above when the permanent DOI / proceedings entry is available.

If you want, I can also add a small `kernel_experiments.py` example config
table or a short section describing the exact Appendix C target construction in
`best_candidate_found.py`.

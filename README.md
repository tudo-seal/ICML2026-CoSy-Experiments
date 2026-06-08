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
- `bo_cli.py` — command-line interface to run the experiments. For reproducing
  the paper, use its `run` command only; the paper figures were generated
  separately with `plot_paper_figures.py`.
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

Python 3.11 and virtualenv (recommended)
--------------------------------------
We recommend running the code inside an isolated Python 3.11 virtual environment. The following beginner-friendly instructions work on macOS (Homebrew) and Linux and ensure that Python 3.11 is used for the project.

1. Check whether Python 3.11 is already available:

```bash
python3.11 --version
# or (if you use pyenv)
python --version
```

2. If Python 3.11 is missing: install via Homebrew on macOS (recommended):

```bash
brew install python@3.11
```

Alternatively, manage project Python versions with `pyenv`:

```bash
brew install pyenv
pyenv install 3.11.12   # e.g. 3.11.12 — choose the latest 3.11.x
pyenv local 3.11.12     # set the version for the project directory
```

3. Create a virtual environment in the project root:

```bash
python3.11 -m venv .venv
```

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```

5. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Verify that the venv Python is being used:

```bash
which python   # should point to .venv/bin/python
python --version  # should show Python 3.11.x
```

7. Deactivate the venv when finished:

```bash
deactivate
```

These steps are intentionally simple to help less experienced Python users and data scientists set up a reproducible environment. If you need assistance installing Homebrew, configuring `pyenv`, or resolving version conflicts, please open an issue or contact the maintainer.

Reproducing paper experiments
-----------------------------
The repository includes configuration files in `configs/` that were used to
generate the experiment suites reported in the paper. To reproduce a prepared
experiment, call `bo_cli.py run --config <path-to-config.json>` from the
project root. For the paper itself, this is the only CLI command you should
use; the paper figures were generated with `plot_paper_figures.py`, not with
the `plot` subcommand of `bo_cli.py`.

For the kernel-analysis experiments described in the paper, invoke
`kernel_experiments.py` directly as shown above. For the Appendix C
reproduction, run `best_candidate_found.py` directly from the repository root:

```bash
python3 best_candidate_found.py
```

This script constructs the compact search space used for the appendix example,
locates the unique best candidate, and recreates the accompanying plots.

Which configs are required to reproduce the paper?
-----------------------------------------------
Briefly: the JSON files in `configs/` that are intended for reproducing the
experiments reported in the paper are those whose filenames end with
`_global` and the files with the suffixes `_refined_1`, `_refined_2`, and
`_refined_3`. Other JSON files in `configs/` are example or alternative
experiment setups and are not required to reproduce the standard experiments
presented in the paper.

Repository contents
-------------------
- `bo_cli.py` — primary CLI for running and aggregating experiments
- `bo_runner.py` — BO runner and instrumentation (Ask/Tell loop)
- `bo_plotting.py` — helper plot functions for local analysis
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

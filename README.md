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

Python 3.11 und virtualenv (empfohlen)
------------------------------------
Wir empfehlen, den Code in einer isolierten Python-Umgebung unter Python 3.11 auszuführen. Nachfolgend eine einfache, für Einsteiger geeignete Anleitung, die auf macOS (oder Linux) funktioniert und sicherstellt, dass wirklich Python 3.11 verwendet wird.

1. Prüfen, ob Python 3.11 bereits verfügbar ist:

```bash
python3.11 --version
# oder (falls Sie pyenv verwenden)
python --version
```

2. Falls Python 3.11 fehlt: auf macOS mit Homebrew installieren (empfohlen):

```bash
brew install python@3.11
```

Alternativ können Sie `pyenv` verwenden, um Python 3.11 zu installieren und zu verwalten:

```bash
brew install pyenv
pyenv install 3.11.*/  # z.B. 3.11.12 — wählen Sie die neueste 3.11.x
pyenv local 3.11.x     # setzt die Version für das Projekt-Verzeichnis
```

3. Virtual Environment im Projektordner anlegen (aus dem Projekt-Root):

```bash
python3.11 -m venv .venv
```

4. Virtualenv aktivieren:

```bash
source .venv/bin/activate
```

5. pip aktualisieren und Abhängigkeiten installieren:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Sicherstellen, dass die venv-Python-Version stimmt:

```bash
which python   # sollte auf .venv/bin/python zeigen
python --version  # sollte Python 3.11.x anzeigen
```

7. Deaktivieren der venv, wenn Sie fertig sind:

```bash
deactivate
```

Diese Schritte sind bewusst einfach gehalten, damit auch weniger erfahrene Python-Nutzer und Data Scientists die Umgebung reproduzierbar einrichten können. Wenn Sie Hilfe bei der Installation von Homebrew, pyenv oder bei Versionskonflikten brauchen, sagen Sie Bescheid.

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

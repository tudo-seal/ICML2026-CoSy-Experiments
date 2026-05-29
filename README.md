# BO Experimental Evaluation — CLI Guide

This directory contains the experimental pipeline that backs the evaluation
of search-space synthesis for Bayesian Optimization. Three methods are compared
on the same synthesized ODE search space:

- **`random`** — uniform random search baseline
- **`bo`** — Bayesian Optimization with optimized graph or tree kernel
- **`refined_bo`** — multi-slice BO that refines the search space between slices
  (the method under test)

Each experiment is uniquely identified by `(target × kernel × seed)`. The CLI
writes per-experiment CSVs containing the full per-iteration trace
(objective, best-so-far, timings, structural distance to best / top-k) plus
ranking metrics (Kendall τ, Spearman ρ vs. a precomputed candidate pool).

All commands below assume that they are executed from the **project root**
(`cosy-examples/`) inside the project's virtual environment (`venv/`).

---

## Prerequisites

1. Activate the project virtual environment:
   ```bash
   source venv/bin/activate
   ```
   (or call interpreter explicitly via `venv/bin/python …`)
2. Install the project dependencies once (see the top-level
   `bayesian_optimization/README.md` for details).
3. The pipeline trains a small PyTorch network per candidate; the dataset is
   created automatically on first invocation under
   `bayesian_optimization/examples/ODEs/data/`.

---

## Quick start — run one of the prepared experiments

Four ready-to-use configs live under `configs/`. They cover the 2×2 matrix
`(target_len_3, target_len_4) × (noisy_combined_hierarchical_kernel,
noisy_hierarchical_damg_kernel)` with `initial_sample_size=50`,
`eval_budget=100`, and `seed=42`:

```bash
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli run \
  --config bayesian_optimization/examples/ODEs/bo_experimental_evaluation/configs/target_len_3_combined.json
```

This produces a timestamped folder under `results/target_len_3_combined/` with
`random_trace.csv`, `bo_trace.csv`, `refined_trace.csv` plus the matching
`*_ranking.csv` files and a `manifest.json` recording all run parameters.

---

## CLI reference

The CLI is exposed as `bo_cli.py` and groups all functionality into
subcommands:

| Subcommand     | Purpose |
| -------------- | ------- |
| `run`          | Execute experiments for `targets × kernels × seeds` |
| `aggregate`    | Concatenate per-experiment trace/ranking CSVs into a single `aggregated__*` folder |
| `index`        | Build a central `manifest_index.json` over all `manifest.json` files |
| `plot`         | Generate plots from aggregated CSVs (function bodies are still stubbed — the command reports per-plot status without crashing) |
| `list-targets` | Print available target names |
| `list-kernels` | Print supported kernel names |

### `run`

```bash
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli run \
  --config <path-to-json>                # full experiment spec; CLI flags override
  [--targets t1,t2 --kernels k1,k2 --seeds 1,2,3]
  [--eval-budget 100] [--initial-sample-size 50] [--ranking-pool-size 200]
  [--refined-search-space-mode keep|reinitialize]
  [--distance-kernel wl|tree]
  [--results-root path/to/results]
  [--no-progress]
```

CLI arguments take precedence over JSON config values. If `targets`,
`kernels`, or `seeds` are not provided by either source, the command exits
with a descriptive error.

`run` enables tqdm progress bars by default (one bar per method × seed,
showing the current iteration count and ETA) and silences sklearn's
`ConvergenceWarning` so the GP messages from early BO iterations don't drown
out the actual output. Use `--no-progress` to opt out of the bars (e.g. when
redirecting output to a log file); the warning filter is always on inside
`run_command`.

### `aggregate`

```bash
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli aggregate \
  --results-root results/target_len_3_combined
```

Concatenates all `*_trace.csv` and `*_ranking.csv` under the given root into
`<root>/aggregated__<timestamp>/aggregated_{trace,ranking}.csv` and writes a
small per-method summary.

### `index`

```bash
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli index \
  --results-root results/target_len_3_combined
```

Builds `<root>/manifest_index.json` collecting every per-experiment
`manifest.json` under the root, grouped by `experiment_key`.

### `plot`

```bash
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli plot \
  --results-root results/target_len_3_combined \
  [--output-dir results/target_len_3_combined/bo_plots] \
  [--methods random,bo,refined_bo] [--targets target_len_3] [--kernels noisy_combined_hierarchical_kernel]
```

Loads the latest aggregated CSVs, applies the optional filters, and invokes
each plotting function. The plotting functions in `bo_plotting.py` currently
raise `NotImplementedError`; the command reports `"plot_*: not implemented
yet — skipping"` for each and exits cleanly. The CSV schema, however, is
guaranteed to contain every column the future plots will need.

### `list-targets` / `list-kernels`

Print the strings accepted in `--targets` / `--kernels` (or the corresponding
JSON keys).

---

## Config file format

The `--config` flag accepts a JSON object. Every CLI flag listed under `run`
maps to a JSON key with the same name (snake_case). CLI flags always
override JSON values.

| JSON key                       | Type            | Default                  | Notes |
| ------------------------------ | --------------- | ------------------------ | ----- |
| `targets`                      | list[str]       | —                        | Required (CLI or JSON) |
| `kernels`                      | list[str]       | —                        | Required (CLI or JSON) |
| `seeds`                        | list[int]       | —                        | Required (CLI or JSON) |
| `eval_budget`                  | int             | `50`                     | BO iterations after presamples |
| `initial_sample_size`          | int             | `10`                     | Presamples per experiment |
| `ranking_pool_size`            | int             | `200`                    | Reference pool for ranking + distance |
| `kernel_optimizer`             | str             | `"fmin_l_bfgs_b"`        | passed to the GP |
| `n_restarts_kernel_optimizer`  | int             | `10`                     | GP kernel restarts |
| `optimizer_population_size`    | int             | `100`                    | EA population for the acquisition function |
| `optimizer_mutation_rate`      | float           | `0.0`                    | EA mutation rate |
| `optimizer_recombination_rate` | float           | `0.99`                   | EA recombination rate |
| `max_depth`                    | int             | `10000`                  | derivation tree depth cap |
| `refined_search_space_mode`    | `"keep"`/`"reinitialize"` | `"keep"`        | RefinedBO behavior at slice boundaries |
| `distance_kernel_name`         | `"wl"`/`"tree"` | `"wl"`                   | Kernel used for Plot 5/6 distances |
| `results_root`                 | str             | `"results"`              | Output root (per-experiment subfolders inside) |
| `show_progress`                | bool            | `true`                   | Show tqdm progress bars during the run (CLI flag: `--no-progress` to disable) |

Accepted kernel names (case-sensitive): `"wl"`, `"tree"`, `"damg"`,
`"combined"` (short aliases) — or the full names
`"noisy_hierarchical_wl_kernel"`, `"noisy_hierarchical_damg_kernel"`,
`"noisy_combined_hierarchical_kernel"`. See `list-kernels`.

**Refinement schedule (RefinedBO).** The schedule
`(refinement_functions, n_iter_splits, ei_xis)` is callable-valued and
therefore not JSON-serializable. The CLI uses
`default_refinement_schedule(eval_budget)` (defined in `bo_runner.py`),
which mirrors the schedule used in `ode_experiment.py`: one
`algebra_based_refinement(refinement_1_algebra())` refinement applied at
roughly 2/3 of the budget, with `ei_xi` switching from `0.07` (explore) to
`0.01` (exploit). To use a custom schedule, build the three sequences in a
small Python wrapper and call `run_experiment(...)` directly — see the
docstring of `default_refinement_schedule` for an example.

---

## Example configs

The `configs/` directory ships four ready-to-use experiments. All four share
`initial_sample_size=50`, `eval_budget=100`, `seeds=[42]`,
`ranking_pool_size=200`, `refined_search_space_mode="keep"`,
`distance_kernel_name="wl"`.

| File                                  | Target          | Kernel                                 | Output folder                          |
| ------------------------------------- | --------------- | -------------------------------------- | -------------------------------------- |
| `configs/target_len_3_combined.json`  | `target_len_3`  | `noisy_combined_hierarchical_kernel`   | `results/target_len_3_combined/`       |
| `configs/target_len_3_damg.json`      | `target_len_3`  | `noisy_hierarchical_damg_kernel`       | `results/target_len_3_damg/`           |
| `configs/target_len_4_combined.json`  | `target_len_4`  | `noisy_combined_hierarchical_kernel`   | `results/target_len_4_combined/`       |
| `configs/target_len_4_damg.json`      | `target_len_4`  | `noisy_hierarchical_damg_kernel`       | `results/target_len_4_damg/`           |

Run all four sequentially from the project root:

```bash
for cfg in bayesian_optimization/examples/ODEs/bo_experimental_evaluation/configs/*.json; do
  venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli run --config "$cfg"
done
```

---

## End-to-end workflow

```bash
# 1. Run an experiment
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli run \
  --config bayesian_optimization/examples/ODEs/bo_experimental_evaluation/configs/target_len_3_combined.json

# 2. Aggregate per-experiment CSVs into a single trace/ranking pair
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli aggregate \
  --results-root results/target_len_3_combined

# 3. (Optional) Build the central manifest index
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli index \
  --results-root results/target_len_3_combined

# 4. Generate plots (functions still stubbed — reports skip messages, no crash)
venv/bin/python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli plot \
  --results-root results/target_len_3_combined
```

---

## Tips

- **Multiple seeds** for variance estimates: extend `"seeds": [42, 43, 44, 45, 46]`
  in the JSON (or pass `--seeds 42,43,44,45,46` on the CLI). Each seed produces
  its own experiment subfolder.
- **Parallelism across hosts**: give each host a different `results_root`
  (e.g. `results/host_a`, `results/host_b`) and aggregate afterwards by pointing
  `aggregate` at the parent folder.
- **`reinitialize` mode** for RefinedBO discards the BO history between slices
  and draws fresh presamples for the refined search space. This costs an extra
  `(slices − 1) × initial_sample_size` objective evaluations on top of the BO
  budget — make sure your wall-clock estimate accounts for that.
- **Progress bars vs. log files**: progress bars are on by default and render
  to stderr. When redirecting output to a log file or running detached
  (`nohup`, `screen`, CI), pass `--no-progress` (or `"show_progress": false`
  in the JSON) to avoid the carriage-return spam in the log.
- **Custom refinement schedule** (more slices, different algebras): build the
  schedule in a small Python wrapper that imports `run_experiment` directly
  and pass `refinement_functions=…, n_iter_splits=…, ei_xis=…`. The CLI itself
  cannot accept Python callables.

# =============================================================================
# BAYESIAN OPTIMIZATION EXPERIMENT CONFIGURATION
# =============================================================================
#
# This file defines all global constants and configuration helpers used
# across the distributed Bayesian Optimization experiment framework.
#
# The purpose of this module is to centralize all configuration parameters
# so that all experiment modules share identical settings.
#
# IMPORTANT:
# This file must NOT contain experiment logic.
# It should only define constants, enums, and helper functions.
#
#
# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================
#
# Evaluation budget for Bayesian Optimization iterations.
#
# This is the number of BO iterations AFTER the initial sample.
#
EVAL_BUDGET = 50
#
#
# Number of initial samples loaded from presample cache.
#
INITIAL_SAMPLE_SIZE = 10
#
#
# Number of candidate programs used for surrogate ranking evaluation.
#
# These samples are used only for ranking metrics and do NOT count
# toward the BO evaluation budget.
#
RANKING_POOL_SIZE = 200
#
#
# Number of programs used in top-k distance analysis.
#
TOP_K_PROGRAMS = 10
#
#
# Default random seed used when none is provided.
#
DEFAULT_RANDOM_SEED = 42
#
#
# =============================================================================
# RESULT DIRECTORY STRUCTURE
# =============================================================================
#
# Experiments produce results in the following directory structure:
#
# results/
#     bo_runs/
#         <experiment_id>/
#             optimization_trace.csv
#             ranking_metrics.csv
#             metadata.json
#
#
# Aggregated results and plots are written to:
#
# results/
#     bo_plots/
#
#
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
#
# Implement helper functions:
#
# get_default_results_dir()
#
# Returns Path("results/bo_runs")
#
#
# get_default_plot_dir()
#
# Returns Path("results/bo_plots")
#
#
# Ensure directories are created if they do not exist.
#
#
# =============================================================================
# DISTANCE KERNEL OPTIONS
# =============================================================================
#
# Supported structural kernels for distance analysis.
#
# "wl"
# "tree"
#
#
# Do NOT instantiate kernels in this module.
#
#
from pathlib import Path

# Supported structural kernels for distance analysis.
DISTANCE_KERNEL_OPTIONS = ("wl", "tree")


# =============================================================================
# REFINED BO DEFAULT SCHEDULE
# =============================================================================
#
# RefinedBayesianOptimization is the method-under-test in the experimental
# evaluation. The refinement schedule is functionally specified by three
# parallel sequences:
#   refinement_functions × n_iter_splits × ei_xis
# with sum(n_iter_splits) == eval_budget, len(n_iter_splits) == len(ei_xis),
# and len(n_iter_splits) == len(refinement_functions) + 1.
#
# These DEFAULT_* sentinels are intentionally None: when run_experiment is
# called without explicit schedule arguments, it pulls the budget-dependent
# default from `bo_runner.default_refinement_schedule(eval_budget)`, which
# mirrors the schedule used in `ode_experiment.py:217-239`:
#   - one `algebra_based_refinement(refinement_1_algebra())` refinement
#   - 2/3 vs 1/3 split of eval_budget (with modulo-3 tie-break)
#   - ei_xi 0.07 (explore) → 0.01 (exploit)
#
# To override the schedule for a custom experiment, build the three sequences
# manually in code and pass them through to run_experiment / run_command, e.g.:
#
#     from bayesian_optimization.bayesian_optimization import RefinedBayesianOptimization
#     from bayesian_optimization.examples.ODEs.ode_repo_algebras import (
#         refinement_1_algebra, refinement_2_algebra,
#     )
#     custom_schedule = (
#         [
#             RefinedBayesianOptimization.algebra_based_refinement(refinement_1_algebra()),
#             RefinedBayesianOptimization.algebra_based_refinement(refinement_2_algebra()),
#         ],
#         [20, 15, 15],
#         [0.07, 0.03, 0.01],
#     )
#     run_experiment(..., refinement_functions=custom_schedule[0],
#                    n_iter_splits=custom_schedule[1], ei_xis=custom_schedule[2])
#
# CLI-based runs always use the default schedule (callables can't be argparsed);
# custom schedules require a small Python wrapper script.
#

from bayesian_optimization.examples.damg_nas.damg_repo_algebras import refinement_1_algebra

DEFAULT_REFINEMENT_FUNCTIONS: tuple | None = None #(RefinedBayesianOptimization.algebra_based_refinement(refinement_1_algebra()))
DEFAULT_N_ITER_SPLITS: tuple[int, ...] | None = None #(35, 15)
DEFAULT_EI_XIS: tuple[float, ...] | None = None #(0.07, 0.01)

# Structural distance kernel for Plot 5/6 (default: WL).
DEFAULT_DISTANCE_KERNEL: str = "wl"


def _module_results_root() -> Path:
    """Return the default result root located next to this module."""
    return Path(__file__).resolve().parent / "results"


def get_default_results_dir() -> Path:
    """Return and create the default directory for per-run BO outputs."""
    path = _module_results_root() / "bo_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_plot_dir() -> Path:
    """Return and create the default directory for aggregated BO plots."""
    path = _module_results_root() / "bo_plots"
    path.mkdir(parents=True, exist_ok=True)
    return path

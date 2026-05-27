Debugging Bayesian Optimization: EI sanity checks and GP ranking diagnostics
===================================================================================

This document describes the small diagnostic extension added to the BO runner used
for quick local debugging of acquisition optimizer failures and GP ranking quality.

How to enable
--------------
When running the CLI use the `--debug` flag to enable debug plot/array saving.

Example:

```bash
python -m bayesian_optimization.examples.ODEs.bo_experimental_evaluation.bo_cli run \
  --targets target_len_3 \
  --kernels wl \
  --seeds 42 \
  --debug
```

Options
-------
- `--debug` : enable saving of debug plots and raw arrays for EI and GP diagnostics.
- `--debug-plots-dir <path>` : optional directory to write debug plots/arrays. By
  default they are written next to the per-experiment CSV outputs in a
  `debug_plots/` subfolder.
- `--sanity-n-sanity <int>` : number of random candidates sampled per iteration
  for the EI sanity check (controls overhead; default value is conservative).

What is computed
-----------------
Per BO iteration the runner computes and logs (into the trace CSV):

- EI sanity check:
  - `ei_optimizer` : EI value at the optimizer's suggested candidate
  - `ei_random_best` : best EI among the sampled random candidates
  - `ei_ratio` : `ei_optimizer / ei_random_best` (values << 1 indicate optimizer failure)
  - `ei_random_mean` : mean EI across random samples
  - `sigma_random_mean` : mean GP predictive standard deviation across random samples
  - `sigma_random_max` : max GP predictive std across random samples

- GP ranking diagnostics (computed using only already observed points):
  - `gp_rank_spearman` : Spearman rank correlation between GP predictions and true observed objectives
  - `gp_rank_kendall` : Kendall tau correlation
  - `gp_rank_pearson` : Pearson correlation

No additional evaluations of the expensive objective are performed by these diagnostics.

Plots generated
---------------
When `--debug` is used the following plots are generated per experiment (PNG files):

- `ei_optimizer_vs_random_best.png` : EI optimizer vs random best (per iteration)
- `ei_ratio.png` : EI ratio over iterations (with reference y=1)
- `sigma_diagnostics.png` : GP uncertainty diagnostics (mean/max sigma)
- `gp_ranking_correlations.png` : Spearman/Kendall/Pearson over iterations
- `gp_pred_vs_true_iter_<k>.png` : Predicted vs true scatter for saved iteration `k`
- `gp_residuals_iter_<k>.png` : Residual plot for saved iteration `k`

Interpretation hints
--------------------
- If `ei_ratio` is consistently << 1 it likely indicates the acquisition optimizer
  is not finding high EI regions (search/EA configuration or fitness evaluation
  errors).
- If GP ranking correlations are low early and do not improve, the surrogate
  is failing to model the objective shape (bad kernel choice, data issues,
  or model overfitting/underfitting).
- Residual plots can expose systematic bias (non-zero mean residual) or
  heteroscedasticity (residual variance depends on y).

Implementation notes
--------------------
- The diagnostics reuse the fitted `sklearn.gaussian_process.GaussianProcessRegressor`
  (`optimizer._model`) and the existing EI implementation; no BO algorithm
  behavior is changed.
- The EI calculation uses the same `xi` value as the BO when detectable
  (from suggestion diagnostics or RefinedBO slice configuration). When
  unavailable it falls back to xi=0.01.

Contact
-------
If you want the diagnostics to be more conservative (less IO) or to store
additional per-iteration state, tell me which artifacts you prefer and I can
adjust defaults.


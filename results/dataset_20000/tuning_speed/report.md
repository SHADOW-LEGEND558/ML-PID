# Frozen models versus numerical PID optimization

## Method

20 fixed validation plants were selected before timing: two zero-delay and 18 positive-delay plants, random seed 48 within each group. They are a small timing sample, not a new model-accuracy test set. The accepted 20k dataset, main models, settings and test results were not changed.

The numerical baseline is the current PIDOptimizer: up to four feasible starting controllers, SLSQP minimizing normalized regularized ITAE (lambda=0.01), bounds, <=10% overshoot, and independent validation of candidates. Numerical libraries use one thread and plants run sequentially to measure single-plant latency, not multiprocessing throughput. A separate one-start warm-up is excluded.

Models are loaded once and warmed up. Five one-plant predictions are timed for each plant and their median is used. Controller validation is timed separately using the shared default validator. Features are read from saved records; response measurement, feature extraction, model loading, training, dataset generation and system identification are excluded. The optimizer assumes known physical parameters, while inference assumes the five measured features are already available. RF retains six threads, XGBoost four, and PyTorch two. These are implementation/hardware-specific wall-clock timings, not theoretical limits.

## Optimizer latency

Median: 1.480 s; mean: 1.596 s; range: 1.225–2.130 s. All 20/20 optimization runs returned accepted controllers.

## Prediction-only speed

| Model | Median prediction (ms) | Median paired speedup vs optimizer | Accepted on timing sample |
|---|---:|---:|---:|
| linear | 0.4770 | 3,060× | 6/20 |
| ridge | 0.4960 | 3,058× | 6/20 |
| polynomial_ridge | 0.5745 | 2,594× | 7/20 |
| random_forest | 24.5823 | 61× | 18/20 |
| xgboost | 1.8817 | 800× | 19/20 |
| standard_mlp | 0.1385 | 10,875× | 18/20 |

Speedups are medians of per-plant ratios, not ratios of the two reported medians. Prediction-only timing does NOT provide the same acceptance checking as the complete optimizer. A very fast invalid prediction is not a successfully tuned controller. The 20-plant acceptance counts are timing-sample coverage; use the main 2,500-plant test report for accuracy claims.

## Prediction plus independent validation

| Model | Median verified speedup | Matched accepted cases | Median prediction + validation on accepted cases (ms) |
|---|---:|---:|---:|
| linear | 15.74× | 6/20 | 82.482 |
| ridge | 15.88× | 6/20 | 80.044 |
| polynomial_ridge | 17.90× | 7/20 | 78.922 |
| random_forest | 9.17× | 18/20 | 161.928 |
| xgboost | 12.45× | 19/20 | 129.343 |
| standard_mlp | 11.89× | 18/20 | 131.172 |

Verified speedups include only cases where both the optimizer and model-produced controller passed. Accepted subsets differ across models, so these conditional medians cannot rank overall reliability. Model validation includes the existing bounds, poles, refined overshoot, tail error and sampling/horizon-convergence checks. Failed predictions still require an alternative controller or further tuning; fallback cost is not included.

## Interpretation

The MLP is roughly 10,900× faster at proposing gains in this measurement, but the speedup is roughly 12× when the proposal also undergoes independent simulation validation. The simulation dominates the latter path. The comparison is against our current four-start optimizer, not every possible analytical or numerical tuning method.

The previous generation throughput (many plants in parallel) is not a single-plant optimizer latency. Likewise, amortized batch inference microseconds should not be used as one-request latency. Physical observation time and offline label generation/training remain separate costs.

All verification here uses the nominal ideal-PID/second-order-Padé model and known synthetic plant metadata. Real-world empirical-only inference would not automatically have a mathematical plant available for this model-based verification.

Reproduce from the project root with `python scripts/benchmark_tuning_speed.py`. The script times frozen models and reoptimizes only these benchmark plants; it does not train or modify main data/models. Raw measurements are in optimizer.csv and details.csv.

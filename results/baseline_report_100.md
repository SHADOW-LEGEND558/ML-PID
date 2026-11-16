# Baseline validation report — 100-plant experiment

This report describes the previous 100-plant dataset: 80 training plants and 20 held-out test plants. It does not describe training on the new 1,000-plant dataset. The old CSV has been removed at the user’s request; model artifacts, split hashes, and prediction records remain in results/.

All models receive only t_rise, t_settling, y_peak, y_ss, and tau_est. AUC and physical parameters are excluded from training. Physical parameters are retained for test-plant simulation.

| Model | Mean R² | Stable | Fully validated | Outside gain bounds | Median accepted-pair ITAE ratio | Paired count |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.349 | 35% | 5% | 55% | 150.02 | 1 |
| ridge | 0.411 | 50% | 5% | 30% | 176.67 | 1 |
| polynomial_ridge | 0.294 | 60% | 15% | 25% | 1.90 | 3 |
| random_forest | 0.459 | 75% | 40% | 0% | 1.69 | 8 |
| xgboost | 0.464 | 95% | 30% | 5% | 1.75 | 6 |

## Training and validation

The same deterministic plant-grouped holdout is used for all five models. Hyperparameters are selected using grouped cross-validation on training data only. Polynomial expansion and scaling are fitted within training folds; trees do not use scaling. Selection uses mean squared gain error, rather than closed-loop success.

Raw predictions are evaluated without clipping or fallback. Full acceptance requires finite bounded gains, stable closed-loop poles, refined peak overshoot at most 10% within numerical tolerance, tail error at most 0.001, and converged ITAE under time-grid and horizon refinement. Resource limits fail closed. Checks use ideal PID and second-order Padé delay; they are not exact-delay or hardware guarantees.

All 20 saved target controllers passed revalidation. ITAE ratios are predicted/target ITAE, computed only where both controllers pass. Paired counts expose this selection effect; the ratios do not describe failed controllers.

## Failure reasons

- linear: {'unstable': 13, 'overshoot': 6, 'accepted': 1}
- polynomial_ridge: {'overshoot': 9, 'unstable': 8, 'accepted': 3}
- ridge: {'unstable': 10, 'overshoot': 8, 'accepted': 1, 'invalid_gains': 1}
- random_forest: {'accepted': 8, 'overshoot': 7, 'unstable': 5}
- xgboost: {'overshoot': 12, 'accepted': 6, 'unstable': 1, 'invalid_gains': 1}

## Interpretation

XGBoost has the highest average gain R² and stability rate, but random forest has the highest full-validation rate. Overshoot is the main XGBoost failure: 12 of 20 test controllers fail the overshoot check. Neither tree model meets the project’s reliability objective yet.

Twenty test plants are too few for firm model rankings: one plant changes a rate by five percentage points. Linear baselines demonstrate that reasonable gain-error scores do not imply useful control. Larger-data results should be reported separately, with the new split and dataset hash. Repeated model selection against this old test set would make its scores optimistic.

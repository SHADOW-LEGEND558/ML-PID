# 20,000-plant experiment

## Frozen data and evaluation

Source: data\pid_dataset_20000.csv. SHA-256: 0bd540d44914e6dfcb76179688639203004f0a6ec3ac161ea7ca4c399dae2ba0. Generation uses regularized tuning with lambda=0.01.

Exact split: 15,000 training / 2,500 validation / 2,500 testing, seed 42. Delay-zero versus positive-delay stratification is preserved; identical plants are prohibited. Five empirical inputs only. Every model uses empirical-normalized targets; physical parameters enter the simulator only, not inference.

Linear regression, ridge alpha=100, and degree-two polynomial ridge alpha=1 retain their previously selected configurations. Forest compares depth=12/min-leaf=3 versus unlimited-depth/min-leaf=1, with 200 trees. XGBoost compares 150 versus 400 trees with depth=6 and learning rate=0.08. The standard MLP retains 5-128-64-32-3, positive normalized outputs, standardized log loss, dropout=0.05, AdamW weight decay=0.0001, and seed=42. Batch size is 256 for efficiency; maximum epochs=500, patience=50.

Input scaling and target statistics are fitted on the 15,000 training plants. MLP stopping uses validation supervised loss; tree setting selection ranks full-controller pass count then accepted J ratio on validation data. Settings and weights are saved before test scoring. There is no refit using validation/test labels and no post-test retuning. Models train on exactly 15,000 plants.

All 2,500 validation target controllers and 2,500 test target controllers are rechecked. Predicted gains are evaluated raw, without clipping/fallback. Full acceptance requires bounded finite gains, stable poles, refined overshoot <=10%, tail error <=0.001, and resolved ITAE across grids/horizons. Gain MSE/RMSE/R² describe label prediction; feasibility and ITAE/J describe control behavior.

## Final test results

| model | plants | success_pct | stable_pct | bounds_violation_pct | r2 | median_J_ratio | ratio_pairs | reference_passes |
|---|---|---|---|---|---|---|---|---|
| linear | 2500 | 27.760 | 70.600 | 14.360 | 0.478 | 2.806 | 694 | 2500 |
| ridge | 2500 | 28.240 | 70.680 | 14.000 | 0.480 | 2.793 | 706 | 2500 |
| polynomial_ridge | 2500 | 44.280 | 82.120 | 12.640 | 0.697 | 1.948 | 1107 | 2500 |
| random_forest | 2500 | 91.880 | 100.000 | 0.000 | 0.961 | 1.044 | 2297 | 2500 |
| xgboost | 2500 | 87.920 | 100.000 | 0.000 | 0.961 | 1.109 | 2198 | 2500 |
| standard_mlp | 2500 | 97.920 | 100.000 | 0.000 | 0.987 | 1.042 | 2448 | 2500 |

The test scores are descriptive final results for the frozen settings. J ratios include only validated prediction/reference pairs and subsets differ across models. Failed controllers remain in the full-success denominator; smaller conditional medians alone cannot rank overall reliability.

## Per-gain test errors

| model | gain | MAE | RMSE | R2 |
|---|---|---|---|---|
| linear | Kp | 0.528 | 0.737 | 0.738 |
| linear | Ki | 2.034 | 3.509 | 0.248 |
| linear | Kd | 0.189 | 0.406 | 0.448 |
| ridge | Kp | 0.526 | 0.738 | 0.737 |
| ridge | Ki | 2.024 | 3.500 | 0.252 |
| ridge | Kd | 0.188 | 0.405 | 0.451 |
| polynomial_ridge | Kp | 0.317 | 0.488 | 0.885 |
| polynomial_ridge | Ki | 1.240 | 2.249 | 0.691 |
| polynomial_ridge | Kd | 0.155 | 0.381 | 0.515 |
| random_forest | Kp | 0.060 | 0.179 | 0.984 |
| random_forest | Ki | 0.123 | 0.379 | 0.991 |
| random_forest | Kd | 0.055 | 0.167 | 0.907 |
| xgboost | Kp | 0.072 | 0.215 | 0.978 |
| xgboost | Ki | 0.152 | 0.406 | 0.990 |
| xgboost | Kd | 0.053 | 0.160 | 0.915 |
| standard_mlp | Kp | 0.052 | 0.146 | 0.990 |
| standard_mlp | Ki | 0.120 | 0.336 | 0.993 |
| standard_mlp | Kd | 0.027 | 0.079 | 0.979 |

## Raw ITAE compared with optimized targets

| model | paired_count | median_ITAE_ratio | p90_ITAE_ratio | median_ITAE_change_pct |
|---|---|---|---|---|
| linear | 694 | 3.709 | 18.590 | 270.852 |
| ridge | 706 | 3.690 | 17.057 | 268.979 |
| polynomial_ridge | 1107 | 2.361 | 13.777 | 136.094 |
| random_forest | 2297 | 1.046 | 1.481 | 4.628 |
| xgboost | 2198 | 1.118 | 1.716 | 11.827 |
| standard_mlp | 2448 | 1.043 | 1.495 | 4.282 |

ITAE ratio = predicted ITAE / optimized-target ITAE. Percentage change = 100*(ratio-1). A ratio of 1 or change of 0% means equal tracking cost. Only converged, validated pairs are compared; unstable/unresolved/failed controllers are not assigned zero ITAE. Coverage is explicit. Lower raw ITAE does not automatically mean better J because J also penalizes aggressive gains, and the reference labels are local numerical solutions.

## Overshoot severity on test plants

| model | at_most_10 | over_10_to_12 | over_12_to_15 | over_15_to_20 | over_20 | unstable | stable_unmeasured |
|---|---|---|---|---|---|---|---|
| linear | 699 | 43 | 53 | 119 | 844 | 735 | 7 |
| ridge | 710 | 43 | 61 | 110 | 835 | 733 | 8 |
| polynomial_ridge | 1114 | 82 | 103 | 149 | 603 | 447 | 2 |
| random_forest | 2297 | 78 | 64 | 31 | 30 | 0 | 0 |
| xgboost | 2198 | 102 | 91 | 79 | 30 | 0 | 0 |
| standard_mlp | 2449 | 38 | 10 | 3 | 0 | 0 | 0 |

Measured overshoot on rejected controllers is diagnostic. Unstable and unmeasured cases remain separate; below-limit overshoot alone is not full acceptance.

## Validation selections (before test)

| model | setting | success_pct | r2 | median_J_ratio |
|---|---|---|---|---|
| linear | fixed | 30.000 | 0.462 | 2.646 |
| ridge | alpha100 | 30.440 | 0.464 | 2.637 |
| polynomial_ridge | degree2_alpha1 | 43.560 | 0.682 | 1.919 |
| random_forest | depthNone_leaf1 | 92.000 | 0.963 | 1.048 |
| xgboost | trees400 | 88.280 | 0.964 | 1.118 |
| standard_mlp | standard | 98.240 | 0.983 | 1.048 |

## Learning curves

The same nested random training subsets (1,000/5,000/10,000/15,000 plants) are used for every model. Curves use the same 500 delay-stratified validation plants, not test plants. Full-size curve scores are filtered from existing validation results; smaller fits use fixed selected settings. MLP early stopping uses the full 2,500 validation set for all sizes. Curves are development diagnostics and can be optimistic because settings and stopping used validation data.

| model | training_size | success_pct | r2 | median_J_ratio |
|---|---|---|---|---|
| linear | 1000 | 28.600 | 0.281 | 2.756 |
| linear | 5000 | 28.200 | 0.301 | 2.730 |
| linear | 10000 | 29.200 | 0.306 | 2.806 |
| linear | 15000 | 28.800 | 0.311 | 2.815 |
| polynomial_ridge | 1000 | 38.200 | 0.561 | 2.079 |
| polynomial_ridge | 5000 | 39.000 | 0.596 | 2.054 |
| polynomial_ridge | 10000 | 39.000 | 0.590 | 2.009 |
| polynomial_ridge | 15000 | 40.000 | 0.592 | 2.055 |
| random_forest | 1000 | 69.600 | 0.905 | 1.138 |
| random_forest | 5000 | 84.200 | 0.946 | 1.061 |
| random_forest | 10000 | 87.800 | 0.961 | 1.049 |
| random_forest | 15000 | 91.200 | 0.969 | 1.043 |
| ridge | 1000 | 33.200 | 0.320 | 2.769 |
| ridge | 5000 | 30.200 | 0.312 | 2.683 |
| ridge | 10000 | 29.200 | 0.312 | 2.822 |
| ridge | 15000 | 29.000 | 0.315 | 2.784 |
| standard_mlp | 1000 | 80.000 | 0.959 | 1.110 |
| standard_mlp | 5000 | 96.000 | 0.977 | 1.073 |
| standard_mlp | 10000 | 96.600 | 0.981 | 1.056 |
| standard_mlp | 15000 | 98.000 | 0.984 | 1.050 |
| xgboost | 1000 | 73.400 | 0.837 | 1.174 |
| xgboost | 5000 | 83.400 | 0.955 | 1.153 |
| xgboost | 10000 | 87.400 | 0.958 | 1.109 |
| xgboost | 15000 | 89.000 | 0.967 | 1.118 |

## Failure counts

| model | reason | plants |
|---|---|---|
| linear | overshoot | 1053 |
| linear | unstable | 735 |
| linear | accepted | 694 |
| linear | duration_limit | 10 |
| linear | invalid_gains | 8 |
| ridge | overshoot | 1045 |
| ridge | unstable | 733 |
| ridge | accepted | 706 |
| ridge | duration_limit | 9 |
| ridge | invalid_gains | 6 |
| ridge | resolution_limit | 1 |
| polynomial_ridge | accepted | 1107 |
| polynomial_ridge | overshoot | 934 |
| polynomial_ridge | unstable | 447 |
| polynomial_ridge | duration_limit | 7 |
| polynomial_ridge | invalid_gains | 4 |
| polynomial_ridge | resolution_limit | 1 |
| random_forest | accepted | 2297 |
| random_forest | overshoot | 203 |
| xgboost | accepted | 2198 |
| xgboost | overshoot | 302 |
| standard_mlp | accepted | 2448 |
| standard_mlp | overshoot | 51 |
| standard_mlp | resolution_limit | 1 |

## Main findings

The standard MLP passes 2,448/2,500 test plants (97.92%), versus 2,297/2,500 for random forest (91.88%) and 2,198/2,500 for XGBoost (87.92%). All three are stable on the complete test set; full-pass failures still matter. The remaining regression models have much lower control success despite reasonable gain-error scores.

Among validated pairs only, median ITAE is 4.28% above the optimized target for the MLP, 4.63% above for random forest, and 11.83% above for XGBoost. This is conditional performance with the coverage shown above, not a guarantee that every accepted controller is this close to the target.

On the fixed 500-plant validation subset, MLP success rises 80.0% -> 96.0% -> 96.6% -> 98.0% at 1,000 -> 5,000 -> 10,000 -> 15,000 training plants. Forest rises 69.6% -> 84.2% -> 87.8% -> 91.2%. This supports the usefulness of the extra data for these nonlinear models within the studied bounds. Linear/ridge curves are largely flat. These are pipeline learning curves with fixed hyperparameters and an epoch cap/early stopping for MLP, not pure information-theoretic sample-size effects.

## Verification

All 29 regression tests passed. Split sizes, disjointness, and zero-delay stratification were checked. The dataset hash matched its sidecar; 128 training feature recomputations matched stored values. All 5,000 validation/test reference controllers passed revalidation. All six saved models reproduced their exported test predictions. Model-file and selection hashes remained unchanged after settings were frozen. Learning-curve evaluations use validation rows only. Both updated notebooks were executed, preserving earlier sections and adding the new-data sections at the end.


## Limits and artifacts

Simulation uses ideal PID and second-order Padé delay, not exact-delay/hardware validation or a robustness guarantee. This is one training seed and one split; initialization and distribution uncertainty remain. All models have the same input information, but differ in loss/output parameterization; improvements cannot be attributed solely to architecture. The dataset’s parameter distribution and accepted-label filtering define the evaluation population.

Saved model files, test/validation predictions, split manifest, candidate settings, MLP epoch histories, and learning-curve records are in this directory. Earlier 1,000-row results and notebook sections are preserved. New 20,000-row sections are appended at the end of notebooks/baseline_validation.ipynb and notebooks/mlp_validation.ipynb.

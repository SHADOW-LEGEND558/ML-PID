# Final 10,000-plant evaluation

## Protocol

Dataset: data/pid_dataset_final_10000.csv. SHA-256: e86f0039c0747afa186a322bdc89f77b99f03a1ba9aaa77a40c6243e64d97089.

Fresh independent cohort; exactly 7,000 training / 1,500 validation / 1,500 test plants. The prepared manifest is reused unchanged (seeds 42/43; zero/positive delay stratification). There are 719 zero-delay plants in training, 154 in validation and 154 in test. All six models share these rows.

All models use only t_rise, t_settling, y_peak, y_ss and tau_est. Empirical-normalized targets are q=[y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]. Preprocessing is fitted only on training plants. True K, wn, zeta and tau enter controller simulation only.

Linear, ridge alpha=100 and degree-two ridge alpha=1 retain the development configurations. Random forest compares 200 trees with (depth 12, leaf 3) versus (unlimited depth, leaf 1). XGBoost compares 150 versus 400 trees per target at depth 6, learning rate 0.08. Standard MLP is 5-128-64-32-3, SiLU, dropout 0.05, positive normalized outputs, standardized log loss, AdamW learning rate 0.001/weight decay 0.0001, batch 256, seed 42, max 500 epochs and patience 50; physics loss is disabled.

Select settings by validation full-controller pass count, then median accepted J ratio. MLP early stopping uses validation supervised loss. Saved weights, settings and protocol hashes are locked before test scoring. No refit on validation labels, prediction clipping, fallback controller or post-test retuning is used.

Reference and predicted controllers undergo the same independent checks: finite bounded gains, stable poles, refined overshoot <=10%, tail error <=0.001 and converged grids/horizons. J=wn^2*ITAE+0.01*((K*Kp)^2+(K*Ki/wn)^2+(K*wn*Kd)^2). Validation refers to simulated ideal PID with second-order Pade delay, not hardware safety.

## Held-out results

| model | passes | plants | success_pct | stable_pct | r2 | median_ITAE_ratio | median_J_ratio | ratio_pairs |
|---|---|---|---|---|---|---|---|---|
| linear | 437 | 1500 | 29.1333 | 71.3333 | 0.3919 | 3.4092 | 2.6691 | 437 |
| ridge | 442 | 1500 | 29.4667 | 71.7333 | 0.3970 | 3.3375 | 2.5608 | 442 |
| polynomial_ridge | 620 | 1500 | 41.3333 | 82.0667 | 0.6655 | 2.6048 | 2.1051 | 620 |
| random_forest | 1301 | 1500 | 86.7333 | 100.0000 | 0.9410 | 1.0677 | 1.0655 | 1301 |
| xgboost | 1272 | 1500 | 84.8000 | 100.0000 | 0.9575 | 1.1368 | 1.1223 | 1272 |
| standard_mlp | 1406 | 1500 | 93.7333 | 100.0000 | 0.9794 | 1.0577 | 1.0578 | 1406 |

## Main findings

MLP has the highest held-out acceptance: 1406/1500 (93.73%). Its median accepted-pair ITAE increase is 5.77%, and median J increase is 5.78%.

MLP outcomes: {'accepted': 1406, 'overshoot': 94}. All 1500/1500 are pole-stable; full acceptance remains a stricter test. Overshoot misses split into 61 between 10-12%, 28 between 12-15%, 3 between 15-20%, and 2 above 20%; maximum measured overshoot is 27.53%.


Ratios are conditional on both reference and prediction passing, with coverage shown. Failed controllers remain in the 1,500-plant acceptance denominator and are never assigned zero cost.

## Per-gain errors (all test plants)

| model | gain | MAE | RMSE | R2 |
|---|---|---|---|---|
| linear | Kp | 0.5613 | 0.8160 | 0.7154 |
| linear | Ki | 2.1680 | 3.8983 | 0.1259 |
| linear | Kd | 0.1921 | 0.4001 | 0.3343 |
| ridge | Kp | 0.5596 | 0.8178 | 0.7142 |
| ridge | Ki | 2.1551 | 3.8863 | 0.1312 |
| ridge | Kd | 0.1909 | 0.3967 | 0.3455 |
| polynomial_ridge | Kp | 0.3461 | 0.5415 | 0.8747 |
| polynomial_ridge | Ki | 1.3877 | 2.6671 | 0.5908 |
| polynomial_ridge | Kd | 0.1455 | 0.3358 | 0.5309 |
| random_forest | Kp | 0.0777 | 0.2327 | 0.9769 |
| random_forest | Ki | 0.1552 | 0.4839 | 0.9865 |
| random_forest | Kd | 0.0669 | 0.1838 | 0.8595 |
| xgboost | Kp | 0.0841 | 0.2326 | 0.9769 |
| xgboost | Ki | 0.1717 | 0.4318 | 0.9893 |
| xgboost | Kd | 0.0592 | 0.1500 | 0.9065 |
| standard_mlp | Kp | 0.0629 | 0.1934 | 0.9840 |
| standard_mlp | Ki | 0.1654 | 0.4461 | 0.9886 |
| standard_mlp | Kd | 0.0337 | 0.0909 | 0.9656 |

## ITAE tail analysis

| model | pairs | median_ITAE_ratio | median_change_pct | p90_ITAE_ratio | p95_ITAE_ratio | max_ITAE_ratio | ITAE_over_2x |
|---|---|---|---|---|---|---|---|
| linear | 437 | 3.4092 | 240.9186 | 12.7502 | 55.1730 | 2222.8545 | 306 |
| ridge | 442 | 3.3375 | 233.7538 | 12.9284 | 60.3374 | 1205.6368 | 309 |
| polynomial_ridge | 620 | 2.6048 | 160.4814 | 12.2619 | 30.3446 | 1725.4164 | 356 |
| random_forest | 1301 | 1.0677 | 6.7729 | 1.5380 | 1.9719 | 7.1555 | 63 |
| xgboost | 1272 | 1.1368 | 13.6830 | 1.8016 | 2.5185 | 8.7456 | 98 |
| standard_mlp | 1406 | 1.0577 | 5.7748 | 1.6000 | 2.1035 | 9.6121 | 78 |

Acceptance does not bound ITAE regret. Raw ITAE can improve while regularized J worsens; labels are best-found local numerical solutions, not global optima.

## Overshoot severity

| model | at_most_10 | over_10_to_12 | over_12_to_15 | over_15_to_20 | over_20 | stable_unmeasured | unstable |
|---|---|---|---|---|---|---|---|
| linear | 438 | 27 | 34 | 62 | 505 | 4 | 430 |
| ridge | 444 | 31 | 41 | 60 | 496 | 4 | 424 |
| polynomial_ridge | 624 | 69 | 68 | 104 | 361 | 5 | 269 |
| random_forest | 1301 | 58 | 57 | 46 | 38 | 0 | 0 |
| xgboost | 1273 | 65 | 76 | 61 | 25 | 0 | 0 |
| standard_mlp | 1406 | 61 | 28 | 3 | 2 | 0 | 0 |

Below-limit overshoot alone is not full acceptance. Unresolved numerical checks and unstable controllers remain separate.

## Validation-only selections

| model | setting | success_pct | median_J_ratio |
|---|---|---|---|
| linear | fixed | 28.8000 | 2.5059 |
| ridge | alpha100 | 29.2000 | 2.5096 |
| polynomial_ridge | degree2_alpha1 | 44.6000 | 2.1009 |
| random_forest | depthNone_leaf1 | 86.9333 | 1.0567 |
| xgboost | trees400 | 85.3333 | 1.1142 |
| standard_mlp | standard | 95.3333 | 1.0617 |

## Learning curves

Fixed selected settings; shared nested 1,000/3,000/5,000/7,000 training subsets and the same 500 stratified validation plants. MLP stopping uses the full validation set. These are development diagnostics, not repeated test evaluations or independent estimates of hyperparameter-selection performance.

| model | training_size | success_pct | median_J_ratio |
|---|---|---|---|
| linear | 1000 | 30.0000 | 2.4610 |
| linear | 3000 | 29.0000 | 2.6577 |
| linear | 5000 | 29.6000 | 2.6139 |
| linear | 7000 | 30.0000 | 2.7016 |
| polynomial_ridge | 1000 | 46.8000 | 2.0977 |
| polynomial_ridge | 3000 | 42.8000 | 2.1662 |
| polynomial_ridge | 5000 | 44.6000 | 2.1114 |
| polynomial_ridge | 7000 | 45.6000 | 2.1477 |
| random_forest | 1000 | 78.2000 | 1.0954 |
| random_forest | 3000 | 82.0000 | 1.0762 |
| random_forest | 5000 | 84.2000 | 1.0624 |
| random_forest | 7000 | 85.6000 | 1.0552 |
| ridge | 1000 | 33.4000 | 2.7008 |
| ridge | 3000 | 31.4000 | 2.6583 |
| ridge | 5000 | 30.4000 | 2.6654 |
| ridge | 7000 | 30.0000 | 2.6621 |
| standard_mlp | 1000 | 84.6000 | 1.1213 |
| standard_mlp | 3000 | 94.0000 | 1.0798 |
| standard_mlp | 5000 | 95.0000 | 1.0603 |
| standard_mlp | 7000 | 95.2000 | 1.0549 |
| xgboost | 1000 | 79.2000 | 1.1672 |
| xgboost | 3000 | 83.4000 | 1.1359 |
| xgboost | 5000 | 84.0000 | 1.1093 |
| xgboost | 7000 | 86.2000 | 1.1049 |

## Error analysis by plant group

Descriptive post-test breakdown; no settings are changed using these results. Group counts matter; physical parameters are audit metadata, not model inputs.

| model | group | plants | passes | success_pct |
|---|---|---|---|---|
| linear | zero delay | 154 | 105 | 68.1818 |
| linear | 0 < delay <= 0.1 | 122 | 63 | 51.6393 |
| linear | 0.1 < delay <= 0.5 | 552 | 90 | 16.3043 |
| linear | delay > 0.5 | 672 | 179 | 26.6369 |
| linear | zeta < 0.5 | 317 | 17 | 5.3628 |
| linear | 0.5 <= zeta < 1 | 373 | 52 | 13.9410 |
| linear | zeta >= 1 | 810 | 368 | 45.4321 |
| ridge | zero delay | 154 | 105 | 68.1818 |
| ridge | 0 < delay <= 0.1 | 122 | 63 | 51.6393 |
| ridge | 0.1 < delay <= 0.5 | 552 | 89 | 16.1232 |
| ridge | delay > 0.5 | 672 | 185 | 27.5298 |
| ridge | zeta < 0.5 | 317 | 16 | 5.0473 |
| ridge | 0.5 <= zeta < 1 | 373 | 52 | 13.9410 |
| ridge | zeta >= 1 | 810 | 374 | 46.1728 |
| polynomial_ridge | zero delay | 154 | 107 | 69.4805 |
| polynomial_ridge | 0 < delay <= 0.1 | 122 | 54 | 44.2623 |
| polynomial_ridge | 0.1 < delay <= 0.5 | 552 | 139 | 25.1812 |
| polynomial_ridge | delay > 0.5 | 672 | 320 | 47.6190 |
| polynomial_ridge | zeta < 0.5 | 317 | 54 | 17.0347 |
| polynomial_ridge | 0.5 <= zeta < 1 | 373 | 93 | 24.9330 |
| polynomial_ridge | zeta >= 1 | 810 | 473 | 58.3951 |
| random_forest | zero delay | 154 | 141 | 91.5584 |
| random_forest | 0 < delay <= 0.1 | 122 | 103 | 84.4262 |
| random_forest | 0.1 < delay <= 0.5 | 552 | 458 | 82.9710 |
| random_forest | delay > 0.5 | 672 | 599 | 89.1369 |
| random_forest | zeta < 0.5 | 317 | 252 | 79.4953 |
| random_forest | 0.5 <= zeta < 1 | 373 | 278 | 74.5308 |
| random_forest | zeta >= 1 | 810 | 771 | 95.1852 |
| xgboost | zero delay | 154 | 149 | 96.7532 |
| xgboost | 0 < delay <= 0.1 | 122 | 113 | 92.6230 |
| xgboost | 0.1 < delay <= 0.5 | 552 | 442 | 80.0725 |
| xgboost | delay > 0.5 | 672 | 568 | 84.5238 |
| xgboost | zeta < 0.5 | 317 | 262 | 82.6498 |
| xgboost | 0.5 <= zeta < 1 | 373 | 246 | 65.9517 |
| xgboost | zeta >= 1 | 810 | 764 | 94.3210 |
| standard_mlp | zero delay | 154 | 154 | 100.0000 |
| standard_mlp | 0 < delay <= 0.1 | 122 | 116 | 95.0820 |
| standard_mlp | 0.1 < delay <= 0.5 | 552 | 524 | 94.9275 |
| standard_mlp | delay > 0.5 | 672 | 612 | 91.0714 |
| standard_mlp | zeta < 0.5 | 317 | 290 | 91.4826 |
| standard_mlp | 0.5 <= zeta < 1 | 373 | 360 | 96.5147 |
| standard_mlp | zeta >= 1 | 810 | 756 | 93.3333 |

## Primary failure outcomes

| model | reason | plants |
|---|---|---|
| linear | overshoot | 620 |
| linear | accepted | 437 |
| linear | unstable | 430 |
| linear | invalid_gains | 8 |
| linear | duration_limit | 4 |
| linear | resolution_limit | 1 |
| ridge | overshoot | 620 |
| ridge | accepted | 442 |
| ridge | unstable | 424 |
| ridge | invalid_gains | 8 |
| ridge | duration_limit | 5 |
| ridge | resolution_limit | 1 |
| polynomial_ridge | accepted | 620 |
| polynomial_ridge | overshoot | 600 |
| polynomial_ridge | unstable | 269 |
| polynomial_ridge | duration_limit | 6 |
| polynomial_ridge | invalid_gains | 5 |
| random_forest | accepted | 1301 |
| random_forest | overshoot | 199 |
| xgboost | accepted | 1272 |
| xgboost | overshoot | 227 |
| xgboost | invalid_gains | 1 |
| standard_mlp | accepted | 1406 |
| standard_mlp | overshoot | 94 |

## Sampling uncertainty

95% Wilson intervals describe finite-test uncertainty within this accepted-label simulation population. They do not cover seed variability, model mismatch, selection bias, or physical robustness.

| model | acceptance_pct | Wilson95_low | Wilson95_high |
|---|---|---|---|
| linear | 29.1333 | 26.8895 | 31.4837 |
| ridge | 29.4667 | 27.2144 | 31.8239 |
| polynomial_ridge | 41.3333 | 38.8666 | 43.8444 |
| random_forest | 86.7333 | 84.9225 | 88.3565 |
| xgboost | 84.8000 | 82.8944 | 86.5278 |
| standard_mlp | 93.7333 | 92.3916 | 94.8516 |

## Development-study context

Different cohorts and test sets: these percentages are descriptive, not a controlled causal comparison of training sizes. The within-cohort learning curves are more informative about sample-size trends.

| model | 20K_development_test_pct | 10K_final_test_pct |
|---|---|---|
| linear | 27.7600 | 29.1333 |
| ridge | 28.2400 | 29.4667 |
| polynomial_ridge | 44.2800 | 41.3333 |
| random_forest | 91.8800 | 86.7333 |
| xgboost | 87.9200 | 84.8000 |
| standard_mlp | 97.9200 | 93.7333 |

## Verification

All 1500/1500 validation and 1500/1500 test numerical-reference controllers passed revalidation. All saved models reproduced their scored gains (max differences: {'linear': 0.0, 'ridge': 0.0, 'polynomial_ridge': 0.0, 'random_forest': 2.1316282072803006e-14, 'xgboost': 0.0, 'standard_mlp': 0.0}). Dataset, prepared split, selected settings and saved-model hashes remained unchanged after locking. Final test scoring occurred once per model.

## Limits

One split and training seed; a noise-free stable positive-gain SOPDT family in fixed ranges; ideal differentiation and second-order Pade delay; no actuator saturation, exact-delay robustness or hardware validation. Accepted-label filtering excludes teacher failures and can bias the main population. Results do not establish a universal or safety-certified tuner.

Artifacts: saved models, candidate/selected settings, immutable split and selection lock, per-plant test/validation predictions, reference checks, histories, learning curves and plots are stored alongside this report. The website presents the final 10K study; additional development curves, stress tests and response diagnostics retain their own scopes.

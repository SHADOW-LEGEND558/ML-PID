# Training-only baseline improvement experiments

Only the five existing models were used. No new dataset or model family was added. The 200 saved test plants were excluded from every fit, validation score, overshoot analysis, and learning curve. Protocol and row identities are in protocol.json.

## Method

The 800 development plants were evaluated with three grouped folds. Each candidate fits roughly 533 plants and predicts the other roughly 267; pooling the folds gives 800 out-of-fold predictions. Every candidate uses the same folds.

38 candidates combine the existing hyperparameter grids with raw or empirical-normalized targets. For normalization, q=[y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]; inverse scaling at prediction uses only the five input features. Polynomial expansion/scaling is fitted inside each fold. No gain clipping or fallback is applied.

Control selection ranks candidates by full-validation pass count, then median J ratio on accepted pairs, then raw gain MSE. MSE selection chooses the smallest raw gain MSE on these same predictions. All reference controllers were revalidated. The objective uses lambda=0.01 and the unchanged stability, gain-bound, overshoot and convergence criteria.

**These are development scores used to select settings, not unbiased final test estimates.** Selecting candidates and reporting their scores on the same folds introduces selection optimism. Paired J summaries exclude failed predictions, so success rate remains primary and coverage must be considered.

## 1–2. Control-based selection and normalized targets

| family | raw_mse_pass_pct | raw_control_pass_pct | normalized_mse_pass_pct | normalized_control_pass_pct |
|---|---|---|---|---|
| linear | 18.000 | 18.000 | 28.750 | 28.750 |
| ridge | 20.375 | 20.375 | 31.375 | 31.375 |
| polynomial_ridge | 30.500 | 30.500 | 44.000 | 44.000 |
| random_forest | 59.125 | 59.125 | 65.875 | 66.250 |
| xgboost | 56.125 | 66.625 | 64.750 | 69.625 |

The table holds the folds fixed. Comparing MSE versus control within a target representation isolates the selection criterion. Comparing raw versus normalized under the same criterion examines target scaling (with independently selected hyperparameters). Gains from one change need not add to gains from the other.

### Selected configuration per existing model

| family | normalized | success_pct | stable_pct | median_J_ratio | r2 | parameters |
|---|---|---|---|---|---|---|
| linear | True | 28.750 | 68.000 | 2.495 | 0.349 | {} |
| ridge | True | 31.375 | 70.500 | 2.313 | 0.449 | {"model__alpha": 100.0} |
| polynomial_ridge | True | 44.000 | 82.375 | 2.028 | 0.641 | {"model__alpha": 1.0} |
| random_forest | True | 66.250 | 97.625 | 1.203 | 0.822 | {"model__max_depth": 12, "model__min_samples_leaf": 3} |
| xgboost | True | 69.625 | 99.250 | 1.218 | 0.628 | {"model__estimator__learning_rate": 0.08, "model__estimator__max_depth": 6} |

## 3. Overshoot failure analysis

| family | passes | overshoot_misses | 10_to_12 | over_20 | unstable | unmeasured_stable |
|---|---|---|---|---|---|---|
| linear | 230 | 311 | 11 | 259 | 256 | 1 |
| ridge | 251 | 309 | 11 | 237 | 236 | 3 |
| polynomial_ridge | 352 | 301 | 21 | 195 | 141 | 5 |
| random_forest | 530 | 250 | 35 | 110 | 19 | 1 |
| xgboost | 557 | 237 | 37 | 99 | 6 | 0 |

Misses are measured stable responses above 10%; the near-miss band is >10–12%, and severe misses exceed 20%. Measurements from rejected controllers are finite-window diagnostics. Stable unmeasured and unstable cases remain separate.

### Highest-overshoot groups for the tree models (at least 30 plants)

| family | feature | bin | plants | pass_pct | overshoot_miss_pct | severe_miss_pct |
|---|---|---|---|---|---|---|
| xgboost | normalized_delay | (0.5, 1.0] | 100 | 41.000 | 57.000 | 32.000 |
| random_forest | zeta | (0.099, 0.575] | 221 | 36.199 | 55.204 | 32.127 |
| xgboost | wn | (0.499, 2.875] | 208 | 44.231 | 54.327 | 26.923 |
| xgboost | zeta | (0.099, 0.575] | 221 | 50.226 | 47.059 | 21.267 |
| random_forest | wn | (0.499, 2.875] | 208 | 47.115 | 46.154 | 25.000 |
| random_forest | normalized_delay | (0.5, 1.0] | 100 | 45.000 | 46.000 | 27.000 |
| xgboost | zeta | (0.575, 1.05] | 197 | 56.853 | 43.147 | 18.274 |
| random_forest | zeta | (0.575, 1.05] | 197 | 58.883 | 41.117 | 12.690 |

These are associations, not causal effects: bins overlap across different features and other parameters are not controlled. Do not sum counts across features. Full bin counts, coverage, and instability rates are in failure_clusters.csv.

## 4. Learning curves

Keep the selected hyperparameters and target representation fixed. In each fold, the 128-plant subset is nested within the 256-plant subset, both inside that fold’s full training set. The validation plants do not change. Each size has 800 pooled predictions. Full-fold sizes are 533 or 534 and are labeled 533. Curves are descriptive because settings were selected on the full-size folds.

| family | training_size | success_pct | median_J_ratio | r2 |
|---|---|---|---|---|
| linear | 128 | 28.500 | 2.818 | 0.381 |
| linear | 256 | 30.125 | 2.543 | 0.423 |
| linear | 533 | 28.750 | 2.495 | 0.349 |
| polynomial_ridge | 128 | 32.000 | 2.981 | 0.250 |
| polynomial_ridge | 256 | 33.375 | 2.409 | 0.355 |
| polynomial_ridge | 533 | 44.000 | 2.028 | 0.641 |
| random_forest | 128 | 51.875 | 1.535 | 0.730 |
| random_forest | 256 | 59.375 | 1.310 | 0.783 |
| random_forest | 533 | 66.250 | 1.203 | 0.822 |
| ridge | 128 | 27.875 | 2.813 | 0.307 |
| ridge | 256 | 33.875 | 2.456 | 0.416 |
| ridge | 533 | 31.375 | 2.313 | 0.449 |
| xgboost | 128 | 58.000 | 1.426 | 0.297 |
| xgboost | 256 | 64.625 | 1.274 | 0.716 |
| xgboost | 533 | 69.625 | 1.218 | 0.628 |

## Findings and recommendation

1. **Target normalization helps all five model families in these development folds.** Under control selection, random forest improves from 59.125% with raw targets to 66.250% with normalized targets (+7.125 percentage points). XGBoost improves from 66.625% to 69.625% (+3.000 points). The selected representation is normalized for every family.
2. **Control-based selection matters most for XGBoost in this grid.** For raw targets, MSE selection passes 56.125% versus 66.625% for control selection (+10.500 points). With normalized targets, the corresponding figures are 64.750% and 69.625% (+4.875 points). For normalized forest, the difference is only 0.375 points (3 plants out of 800); do not overinterpret that small setting difference. For the regression models, MSE and control selection choose the same settings within each representation in this search.
3. **The remaining overshoot errors are often substantial.** Selected normalized forest has 250 measured overshoot misses, of which 35 are in >10–12% and 110 exceed 20%. Selected normalized XGBoost has 237 misses, of which 37 are in >10–12% and 99 exceed 20%. These are separate 800-plant development results, not the historical 200-plant test figures.
4. **Failures cluster in particular regions.** Forest passes only 36.2% of the 221 low-damping plants (zeta 0.1–0.575). XGBoost passes 44.2% of the 208 slower plants (wn 0.5–2.875), and 41% of the 100 plants with normalized delay tau*wn between 0.5 and 1.0. These regions merit better coverage or feature/model investigation; their association with failures does not establish causality.
5. **More training data is supported for the trees, not guaranteed to solve the problem.** Forest success rises 51.875% -> 59.375% -> 66.250% at 128 -> 256 -> about 533 plants per training fold. XGBoost rises 58.000% -> 64.625% -> 69.625%. The final increases are +6.875 and +5.000 points. Linear and ridge curves are nonmonotonic, so simply adding data is less clearly beneficial for those models.

Recommended next step: retain empirical-normalized targets and control-oriented validation for RF/XGBoost, and collect more training data with deliberate auditing of low-damping and slower plants. Keep the declared deployment/test distribution separate from any targeted oversampling. Freeze the selected settings before any future holdout evaluation; a fresh final test set is preferable because the historical holdout has already informed earlier discussion. Current development success is still only about 66–70%, so these models are not reliable automatic controllers yet.

## Saved artifacts and limits

Each candidate’s out-of-fold predictions, parameter settings, selected configurations, and learning-curve records are retained. Selected models were refitted on all 800 development plants and saved with feature order and protocol. They have NOT been scored on the 200 test plants. No previous baseline models or results were overwritten.

Checks concern nominal ideal PID and second-order Padé delay, not exact-delay/hardware robustness. The dataset and penalty weight are fixed throughout; these experiments do not establish global optimality of the labels.

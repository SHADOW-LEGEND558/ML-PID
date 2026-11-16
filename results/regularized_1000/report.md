# Baseline report — 1,000 regularized plants

Date: 2026-11-16. Dataset: data/pid_dataset.csv. Hash: f2ef547e17b28d678768caa91385717e25aa4d6a6af5f5835486f71a153ac270.

## Protocol

Exactly five inputs: t_rise, t_settling, y_peak, y_ss, tau_est. Physical parameters are used only to simulate held-out plants. Targets were generated with lambda=0.01 and four numerical starts. The objective is J = wn² ITAE + 0.01 [(K Kp)² + (K Ki/wn)² + (K wn Kd)²].

800 plants train; 200 held out, including 19 zero-delay and 181 positive-delay plants. The same grouped split (seed 42) is used for every model. Identical plants cannot cross splits. Three-fold grouped cross-validation selects hyperparameters using gain MSE within the training set; closed-loop success and J are held-out evaluation metrics, not the CV selection criterion. Scaling and polynomial transformations are fitted within folds.

All 200 reference controllers passed revalidation. Raw model predictions are never clipped or replaced. A full pass requires bounded finite gains, stable poles, refined overshoot <=10% within tolerance, tail error <=0.001, and converged ITAE with finer time sampling and longer horizons. Failures and resource limits remain nonpasses.

## Main comparison

| Model | Mean test R² | Stable | Full pass | Bound violations | Median J ratio | Ratio pairs |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.612 | 67.0% | 24.0% | 20.0% | 4.356 | 48/200 |
| ridge | 0.597 | 69.5% | 23.5% | 17.0% | 3.089 | 47/200 |
| polynomial_ridge | 0.793 | 69.5% | 34.0% | 26.5% | 2.662 | 68/200 |
| random_forest | 0.820 | 99.0% | 60.0% | 0.0% | 1.225 | 120/200 |
| xgboost | 0.854 | 98.0% | 57.5% | 0.0% | 1.369 | 115/200 |

J ratio is predicted-controller J / reference-controller J; 1 is equal cost. Ratios include ONLY pairs where both controllers passed. These subsets differ by model, so ratio medians cannot by themselves rank overall performance. All failures remain in the denominator of full-pass rates. A ratio below 1 is possible because label generation uses local searches.

## Gain errors

| Model | Gain | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| linear | Kp | 0.5890 | 0.8370 | 0.715 |
| linear | Ki | 1.8271 | 3.2353 | 0.414 |
| linear | Kd | 0.1729 | 0.3279 | 0.706 |
| ridge | Kp | 0.5795 | 0.8642 | 0.696 |
| ridge | Ki | 1.7349 | 3.2629 | 0.404 |
| ridge | Kd | 0.1725 | 0.3362 | 0.691 |
| polynomial_ridge | Kp | 0.3629 | 0.5567 | 0.874 |
| polynomial_ridge | Ki | 1.4770 | 2.4046 | 0.676 |
| polynomial_ridge | Kd | 0.1359 | 0.2507 | 0.828 |
| random_forest | Kp | 0.2512 | 0.5890 | 0.859 |
| random_forest | Ki | 0.3513 | 0.8733 | 0.957 |
| random_forest | Kd | 0.1566 | 0.3610 | 0.644 |
| xgboost | Kp | 0.2257 | 0.5298 | 0.886 |
| xgboost | Ki | 0.3650 | 1.0291 | 0.941 |
| xgboost | Kd | 0.1419 | 0.3103 | 0.737 |

## Failure breakdown

| Model | Accepted | Overshoot | Unstable | Invalid gains | Duration limit |
|---|---:|---:|---:|---:|---:|
| linear | 48 | 84 | 66 | 2 | 0 |
| polynomial_ridge | 68 | 58 | 61 | 10 | 3 |
| ridge | 47 | 89 | 61 | 2 | 1 |
| random_forest | 120 | 78 | 2 | 0 | 0 |
| xgboost | 115 | 81 | 4 | 0 | 0 |

Bounds violations and failure reasons are different classifications: an unstable controller may also have invalid gains. Do not add their percentages.

## Delay groups

| Model | Zero-delay passes | Positive-delay passes |
|---|---:|---:|
| linear | 12/19 | 36/181 |
| polynomial_ridge | 15/19 | 53/181 |
| ridge | 12/19 | 35/181 |
| random_forest | 15/19 | 105/181 |
| xgboost | 16/19 | 99/181 |

## Selected hyperparameters

- linear: default linear regression. See the saved pipeline for all fixed settings.
- polynomial_ridge: {'model__alpha': 1.0}. See the saved pipeline for all fixed settings.
- ridge: {'model__alpha': 100.0}. See the saved pipeline for all fixed settings.
- random_forest: {'model__max_depth': None, 'model__min_samples_leaf': 1}. See the saved pipeline for all fixed settings.
- xgboost: {'model__estimator__learning_rate': 0.08, 'model__estimator__max_depth': 3}. See the saved pipeline for all fixed settings.

## Interpretation

Random forest has the best full-pass rate (120/200, 60%) and XGBoost has the best test gain R² (0.854). XGBoost passes 115/200, or 57.5%. A five-plant difference is insufficient to establish a decisive general winner from one split. Both models remain far short of reliable automatic tuning: overshoot accounts for 78 random-forest failures and 81 XGBoost failures.

Tree training R² is high (random forest 0.978, XGBoost 0.972) compared with held-out R² (0.820 and 0.854), suggesting a generalization gap. Polynomial ridge improves gain prediction over linear models but still often predicts unstable or out-of-bounds controllers. Hyperparameter search is intentionally modest and based on gain error; a next experiment could select hyperparameters by control performance inside training-only folds. Do not repeatedly tune against these 200 test plants.

Compared with historical 100-plant results, both dataset size and target objective changed; improvements cannot be attributed to larger sample size alone. These results concern nominal ideal PID with second-order Padé delay, not exact-delay, noise robustness, actuator limits, or hardware safety.

## Verification

22 regression tests passed. All five saved pipelines reproduce the exported predictions. Split identity, disjoint rows, metadata hashes, lambda, reference acceptance, and exclusion of failed controllers from cost ratios were checked. Models and fitted preprocessing, full CV tables, per-plant predictions, and metrics are retained in this directory. Batch inference timing and fit/CV timing are in metrics.csv; they are local measurements, not controlled latency benchmarks.

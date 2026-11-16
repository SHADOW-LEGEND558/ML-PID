# Frozen-model stress test on rejected generation plants

## Records recovered

All 142 unique rejected plants were recovered from data/generation_20000.log, matching 20,142 attempts minus 20,000 accepted rows. Parameters are retained at the precision printed by the generator. The five empirical features were recomputed using the unchanged observation rule. No rejected plant overlaps the accepted dataset (parameter comparison tolerance 1e-12).

Original plant-level rejection groups: {'resolution_limit': 102, 'overshoot': 36, 'overshoot+resolution_limit': 4}. These are plants, not summed per-start failure counts. Earlier shorthand 103/39 counts obtained by dividing start-level totals by four were inaccurate because four plants had mixed reasons.

## Protocol

All six final frozen models from results/dataset_20000 were loaded. No training, numerical optimization, parameter adjustment, clipping or controller fallback was performed. Model-file hashes were checked before and after the experiment. All models use only the five measured features at inference; physical parameters are used by simulation/validation only.

First, each raw prediction was assessed using the original 65,537-point maximum and all existing checks. Only predictions failing with resolution_limit were retried with up to 262,145 points. This secondary resource policy leaves the 10% overshoot threshold, gain bounds, pole criterion, tail tolerance, convergence tolerances, and 1,000 s horizon cap unchanged. Original successes are retained; other failure categories are not retried. Strict and secondary outcomes are reported separately.

## Results

| model | plants | strict_passes | stable | extended_attempts | passes_with_extended_budget |
|---|---|---|---|---|---|
| linear | 142 | 5 | 130 | 93 | 58 |
| ridge | 142 | 5 | 128 | 93 | 58 |
| polynomial_ridge | 142 | 16 | 141 | 95 | 79 |
| random_forest | 142 | 19 | 142 | 106 | 101 |
| xgboost | 142 | 28 | 142 | 106 | 111 |
| standard_mlp | 142 | 34 | 142 | 106 | 118 |

Standard MLP: 34/142 pass at the original budget; 118/142 pass with the secondary budget policy. All 142/142 predictions are pole-stable. Secondary outcomes: {'accepted': 118, 'resolution_limit': 22, 'overshoot': 2}.

The low strict pass rate is largely a resource-limited validation outcome, not evidence of instability. Conversely, a stable pole test or a plausible plot does not certify overshoot and numerical convergence for the unresolved plants. The 22 MLP cases still at resolution_limit remain unresolved; they are not counted as passes.

## Original rejection group breakdown

| model | original_reason | plants | strict_passes | passes_with_extended_budget |
|---|---|---|---|---|
| linear | resolution_limit | 102 | 4 | 57 |
| linear | overshoot | 36 | 1 | 1 |
| linear | overshoot+resolution_limit | 4 | 0 | 0 |
| ridge | resolution_limit | 102 | 4 | 57 |
| ridge | overshoot | 36 | 1 | 1 |
| ridge | overshoot+resolution_limit | 4 | 0 | 0 |
| polynomial_ridge | resolution_limit | 102 | 3 | 63 |
| polynomial_ridge | overshoot | 36 | 12 | 13 |
| polynomial_ridge | overshoot+resolution_limit | 4 | 1 | 3 |
| random_forest | resolution_limit | 102 | 2 | 78 |
| random_forest | overshoot | 36 | 17 | 19 |
| random_forest | overshoot+resolution_limit | 4 | 0 | 4 |
| xgboost | resolution_limit | 102 | 2 | 79 |
| xgboost | overshoot | 36 | 26 | 28 |
| xgboost | overshoot+resolution_limit | 4 | 0 | 4 |
| standard_mlp | resolution_limit | 102 | 2 | 80 |
| standard_mlp | overshoot | 36 | 32 | 34 |
| standard_mlp | overshoot+resolution_limit | 4 | 0 | 4 |

## What this establishes

Models can propose validated controllers for some plants the multi-start labeling pipeline rejected. This demonstrates that generator rejection does not establish PID control is impossible. The MLP also passes many cases whose original optimized candidates violated overshoot.

The recovered set is deliberately rejection-biased and is not an independent random test sample. These pass rates must not replace or be mixed with the 97.92% MLP result on the accepted-data test distribution. They reveal a label-selection/coverage limitation near difficult plants, especially tiny delays and stiff Padé dynamics.

No accepted optimized gains or objective values were saved for these plants. Therefore gain MAE/RMSE/R², target ITAE ratios, and claims of optimality cannot be computed. Absolute ITAE and J are saved for validated predictions, with coverage; comparing conditional averages across different passing subsets can be misleading.

Plots use illustrative finite display windows. Unresolved or divergent curves can look plausible initially, and outputs can extend beyond the display bounds. The checks concern ideal PID and second-order Padé delay, not exact-delay/hardware robustness.

Artifacts: recovered_plants.csv, predictions.csv, summary.csv, by_original_rejection.csv, separate strict/extended outcome tables, protocol.json, and the accompanying response figures. A standalone notebook is notebooks/rejected_plants_stress_test.ipynb. The accepted dataset, main models, and main benchmark scores are unchanged.

## Verification

All 30 regression tests passed. All 142 records were recovered, giving 852 model/plant predictions. The accepted dataset and six model-file hashes were unchanged. Secondary retries were restricted to original resolution-limit predictions. The executed notebook produced no errors, and all four example checks reproduced saved strict and secondary outcomes.

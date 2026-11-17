# Final response diagnostics and reproducibility checks

## Main conclusion

All 1,500 MLP controllers are pole-stable under the evaluated Padé model and all 1,500 have converged response diagnostics. The original strict result remains **1,406/1,500 (93.73%)**. The other 94 are overshoot constraint misses, not unstable or nonsettling controllers. Their median overshoot is 11.426%, median ITAE is 1.296 times the numerical reference, and median settling time is 1.283 times the reference. Whether these tradeoffs are acceptable depends on the application; ML does not itself justify relaxing a requirement.

## Completing metrics beyond the strict threshold

The new diagnostic evaluator never exits solely because overshoot exceeds 10%. It retains the original pole criterion, 1,000-second horizon cap, 65,537-point grid cap, five horizon iterations, 0.1% ITAE convergence tolerance, 0.001 percentage-point peak convergence tolerance, and 0.001 absolute tail tolerance. Coarse/fine grids and extended horizons must agree. Two-percent settling time is measured from the last sampled band violation, with the final crossing refined against the continuous LTI response. Thus it is numerically resolved time-domain evidence, not an analytic all-time settling proof.

Raw gains are never clipped. Gain bounds are reported separately, and some stable classical-baseline predictions are out of bounds. Threshold-sensitivity counts require stable poles, bounded gains, converged grids/horizons, and the stated overshoot threshold. They are a post-test analysis, not a change to the training objective or original validator. Instability and unresolved simulations remain in the 1,500-plant denominator.

| model | plants | stable | original_strict_passes | resolved | stable_unresolved | resolved_over_10 |
|---|---|---|---|---|---|---|
| linear | 1500 | 1070 | 437 | 1056 | 14 | 619 |
| ridge | 1500 | 1076 | 442 | 1060 | 16 | 618 |
| polynomial_ridge | 1500 | 1231 | 620 | 1217 | 14 | 594 |
| random_forest | 1500 | 1500 | 1301 | 1500 | 0 | 199 |
| xgboost | 1500 | 1500 | 1272 | 1500 | 0 | 227 |
| standard_mlp | 1500 | 1500 | 1406 | 1500 | 0 | 94 |
| numerical_reference | 1500 | 1500 | 1500 | 1500 | 0 | 0 |

## MLP overshoot severity and response quality

Ratios below use all numerically resolved MLP responses in each band, including strict overshoot misses. The same plant's original numerical controller supplies the reference. Ratios are summarized separately; the median of ratios is not a ratio of medians. Small groups (three and two plants in the highest bands) do not support precise population claims.

| band | plants | median_itae_ratio | p90_itae_ratio | median_settling_seconds | median_settling_ratio |
|---|---|---|---|---|---|
| <=10% | 1406 | 1.058 | 1.600 | 1.691 | 1.090 |
| 10-12% | 61 | 1.248 | 1.712 | 2.472 | 1.242 |
| 12-15% | 28 | 1.358 | 1.703 | 2.465 | 1.381 |
| 15-20% | 3 | 1.657 | 1.890 | 3.729 | 2.140 |
| >20% | 2 | 2.630 | 2.743 | 3.435 | 2.871 |

| threshold_pct | plants | resolved_bounded_within_threshold | pct_all_plants |
|---|---|---|---|
| 10 | 1500 | 1406 | 93.733 |
| 12 | 1500 | 1467 | 97.800 |
| 15 | 1500 | 1495 | 99.667 |
| 20 | 1500 | 1498 | 99.867 |

The 12%, 15%, and 20% rows are explicitly exploratory tolerance analyses. Original 10% strict flags and all frozen study artifacts remain unchanged. We should describe small misses as specification violations with quantified performance, rather than treating them as useless controllers. Conversely, these results do not erase true-delay or noisy-measurement weaknesses found in the separate stress study.

## Reproduce the scientific diagnostics

```text
python scripts/final_controller_diagnostics.py --jobs 4 --output temp/controller_diagnostics
```

This recomputes complete response metrics from the published test gains, with no original Python checkpoints required. It writes separate outputs under ignored `temp/` and does not change the published strict acceptance flags. The consolidated `model_testing/results_analysis.ipynb` displays these recorded scientific results alongside the other model comparisons.

The earlier environment/website/cleanup audit records were removed during repository cleanup; they are not required for this numerical experiment. The root README describes the current environment setup, model reconstruction and scientific test commands.

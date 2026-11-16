# Matched 200-plant regularization comparison

400 optimization runs: lambda 0.001 and 0.01 on the same 200 plants, with four starts per run. The calibration set includes all 96 zero-delay plants from the current dataset, the 20 smallest positive delays, and 84 additional positive-delay plants. Selection seed: 42. This is deliberately boundary-enriched, not representative of deployment frequency and not an independent final test set.

| Weight | Group | Validated | Upper-bound hits | Median ITAE | Median Ki |
|---|---|---:|---:|---:|---:|
| 0.001 | Positive delay | 100/104 | 0 | 0.216411 | 1.637 |
| 0.001 | Zero delay | 96/96 | 1 | 0.002920 | 16.434 |
| 0.01 | Positive delay | 100/104 | 0 | 0.216413 | 1.543 |
| 0.01 | Zero delay | 96/96 | 0 | 0.009083 | 8.840 |

Both weights validate exactly the same 196 plants. The remaining four plants fail because the required validation grid exceeds the sampling resource limit, not because instability was established. All four have very small positive delays (0.0037–0.0095 s). Failed attempts are preserved in trials.csv; there are no replacement plants in this comparison.

Among the 96 zero-delay pairs, the median within-plant ITAE ratio (0.01/0.001) is 3.155. The median normalized gain-norm ratio is 0.537: approximately 46% lower gain norm for the larger weight. Among the 100 positive-delay accepted pairs, median ITAE ratio is 1.00002, but its 90th percentile is 3.267; near-zero cases can have a substantial trade-off too.

Lambda 0.001 reaches Ki=100 on one zero-delay plant. Lambda 0.01 has no upper-bound hits in this calibration set. Recommendation: retain the provisional 0.01 if avoiding bound-driven labels and moderating gains is the priority; choose 0.001 if faster tracking is preferred and occasional bound saturation is acceptable. Neither weight guarantees no future bound hits, and neither penalty models physical actuator energy.

The two dataset_lambda_*.csv files contain the 196 matched accepted labels, with the same five empirical features and observation metadata. Failure records and all objective terms remain in trials.csv. The main 1,000-row dataset and existing baseline results have not been changed. Regenerate a final dataset after choosing the weight; tune ML on training/validation splits and preserve an independent final test set.

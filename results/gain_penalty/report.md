# Gain-penalty comparison

16 fixed plants: 8 parameter combinations, each at tau=0 and tau=0.3. Four starts per optimization. Four weights, 64 optimization runs. This is a small design experiment, not a final test set.

Objective: wn² ITAE + lambda [(K Kp)² + (K Ki / wn)² + (K wn Kd)²]. All successful candidates pass the existing independent stability, bounds, overshoot and numerical-convergence checks.

| Weight | Delay | Accepted | Upper-bound hits | Median raw ITAE | Median Ki |
|---|---|---:|---:|---:|---:|
| 0 | 0.3 s | 8/8 | 0 | 0.223741 | 2.092 |
| 0 | zero | 8/8 | 8 | 0.000694 | 100.000 |
| 0.001 | 0.3 s | 8/8 | 0 | 0.224311 | 2.092 |
| 0.001 | zero | 8/8 | 0 | 0.009615 | 11.509 |
| 0.01 | 0.3 s | 8/8 | 0 | 0.229869 | 1.924 |
| 0.01 | zero | 8/8 | 0 | 0.030214 | 6.460 |
| 0.1 | 0.3 s | 8/8 | 0 | 0.232240 | 1.349 |
| 0.1 | zero | 8/8 | 0 | 0.094386 | 3.604 |

Provisional default: lambda=0.01. All tested positive weights removed upper-bound hits on these plants. Increasing weight reduced gains and generally increased tracking ITAE. There is no guarantee of eliminating bound hits across all plants. This criterion regularizes gains rather than modeling physical actuator demand.

The existing 1,000-row dataset and all saved baseline results remain unregularized and require regeneration/retraining for this new criterion.

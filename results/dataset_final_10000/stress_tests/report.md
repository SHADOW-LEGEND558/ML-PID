# Frozen-model stress tests

## Main findings

- Measurement handling is the largest weakness: the original recording rule passed all 300 recordings at 0.01% noise but none at 0.05% or above. With the clean recording length supplied, MLP controller acceptance fell to 26.7% at 0.5% noise and 14.0% at 2% noise. This conditional result must not be presented as end-to-end acquisition success.
- Fixed-gain MLP acceptance under joint parameter uncertainty was 92.2%, 82.6%, and 68.8% at +/-5%, +/-10%, and +/-20%. No MLP pole instabilities occurred in these perturbations; failures were overshoot violations. The numerical reference achieved 99.0%, 87.2%, and 72.4%, respectively.
- The MLP passed all 25 zero-delay and all 25 high-damping boundary plants, but only 11/25 high-normalized-delay plants, versus RF 22/25 and XGBoost 21/25. Overall boundary acceptance was 180/241; 27 resolution-limit cases are not physical instability claims.
- True-delay time-domain passes fell from the nominal Padé counts: MLP 97 to 84, RF 85 to 68, XGBoost 78 to 68, numerical reference 100 to 89 (each out of 100). Thirteen previously accepted MLP controllers exceeded the overshoot limit with true delay. The worst of those changed from 8.94% to 33.23% overshoot. No finite-window pass is a stability certificate.
- All 100 controllers per family recovered from the specified input-load disturbance. MLP median peak output deviation was 6.05% of the unit setpoint, median recovery time 2.12 seconds, and median disturbance ITAE 0.0999. Numerical-reference values were 6.28%, 2.09 seconds, and 0.1020. This test does not establish performance for other disturbance magnitudes or locations.

The next priorities suggested by these tests are noise-aware observation/feature extraction and verification/tuning with true delay, followed by targeted high-normalized-delay coverage. Merely collecting more noise-free Padé-labeled rows will not directly address the first two weaknesses. No changes to models or the main study were made based on these results.



## Scope and reproducibility

Three frozen learned controllers (MLP, random forest, XGBoost) and numerical-reference controllers. No retraining, clipping, fallback, or post-stress tuning. All original artifact and dataset hashes are unchanged. Seed 17112026. The same 100 stratified final-test plants (10 zero delay) are used for the noise, uncertainty, true-delay, and disturbance tests. They are an additional post-test analysis, not a new independent accuracy estimate. Boundary tests use 241 fresh plants. Exact case manifests, gains, and row-level outcomes are saved alongside this report.

Nominal acceptance on the selected 100 plants:

| model | plants | passes | success_pct |
|---|---|---|---|
| numerical_reference | 100.000 | 100.000 | 100.000 |
| random_forest | 100.000 | 85.000 | 85.000 |
| standard_mlp | 100.000 | 97.000 | 97.000 |
| xgboost | 100.000 | 78.000 | 78.000 |

Repeated noisy recordings and parameter perturbations are clustered by plant. Bootstrap intervals in the CSVs/figure resample plants, not individual repeated cases. No claim of worst-case robustness follows from random perturbations.

## 1. Noisy open-loop measurements

IID Gaussian noise with standard deviation 0.01%, 0.05%, 0.1%, 0.5%, 1%, or 2% of the true final gain K is added to 0.01-second response samples, with three recordings per plant/level. Every model receives features extracted from the same noisy trace; features are not perturbed independently. No smoothing or revised feature definitions were introduced.

Two questions are deliberately separated. First, can the original acquisition rule find a settled recording within 640 seconds? Second, if the clean recording length is supplied, how well do the original features and models perform? The second is a conditional diagnostic, not end-to-end success when acquisition fails. The numerical reference is an oracle using true plant parameters; its gains are held fixed and do not depend on noisy features.

| noise_fraction | cases | recording_pass_pct | usable_fixed_window_features_pct |
|---|---|---|---|
| 0.000 | 300.000 | 100.000 | 100.000 |
| 0.001 | 300.000 | 0.000 | 100.000 |
| 0.001 | 300.000 | 0.000 | 100.000 |
| 0.005 | 300.000 | 0.000 | 100.000 |
| 0.010 | 300.000 | 0.000 | 100.000 |
| 0.020 | 300.000 | 0.000 | 100.000 |

MLP conditional diagnostic (300 recordings per level):

| noise_sigma_pct | cases | passes | success_pct | unstable_count | overshoot_max |
|---|---|---|---|---|---|
| 0.010 | 300.000 | 291.000 | 97.000 | 0.000 | 14.748 |
| 0.050 | 300.000 | 291.000 | 97.000 | 0.000 | 14.816 |
| 0.100 | 300.000 | 285.000 | 95.000 | 0.000 | 19.439 |
| 0.500 | 300.000 | 80.000 | 26.667 | 125.000 | 120.897 |
| 1.000 | 300.000 | 43.000 | 14.333 | 217.000 | 84.105 |
| 2.000 | 300.000 | 42.000 | 14.000 | 217.000 | 103.506 |

`noise_results.csv` includes unusable-feature cases in its denominator. Missing overshoot values are not assigned zero. `noise_acquisition.csv` contains the original recording-rule result independently of the conditional controller check.

Feature sensitivity (`noise_feature_drift.csv`):

| noise_fraction | t_rise_median_abs_change_pct | t_settling_median_abs_change_pct | false_zero_onset_pct | onset_median_abs_error_seconds |
|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.001 | 0.000 | 0.388 | 0.000 | 0.000 |
| 0.001 | 0.000 | 0.645 | 0.000 | 0.000 |
| 0.005 | 1.250 | 27.859 | 2.333 | 0.125 |
| 0.010 | 3.419 | 329.495 | 16.333 | 0.440 |
| 0.020 | 6.523 | 335.617 | 30.333 | 0.460 |

## 2. Plant uncertainty with fixed gains

For each plant, independently multiply K, natural frequency, damping, and positive delay by uniform factors in 1 +/- 5%, 10%, and 20%, with five perturbations per level. Nominal zero delay stays zero. Predict/choose gains once from the nominal plant; never retune on the changed plant. These perturbations can cross the original parameter bounds. All controllers see identical perturbed plants. The existing Padé evaluator and limits are unchanged.

| model | level | cases | passes | success_pct | unstable_count |
|---|---|---|---|---|---|
| random_forest | 0.050 | 500.000 | 402.000 | 80.400 | 0.000 |
| random_forest | 0.100 | 500.000 | 363.000 | 72.600 | 0.000 |
| random_forest | 0.200 | 500.000 | 314.000 | 62.800 | 2.000 |
| xgboost | 0.050 | 500.000 | 389.000 | 77.800 | 0.000 |
| xgboost | 0.100 | 500.000 | 352.000 | 70.400 | 0.000 |
| xgboost | 0.200 | 500.000 | 312.000 | 62.400 | 0.000 |
| standard_mlp | 0.050 | 500.000 | 461.000 | 92.200 | 0.000 |
| standard_mlp | 0.100 | 500.000 | 413.000 | 82.600 | 0.000 |
| standard_mlp | 0.200 | 500.000 | 344.000 | 68.800 | 0.000 |
| numerical_reference | 0.050 | 500.000 | 495.000 | 99.000 | 0.000 |
| numerical_reference | 0.100 | 500.000 | 436.000 | 87.200 | 0.000 |
| numerical_reference | 0.200 | 500.000 | 362.000 | 72.400 | 0.000 |

`uncertainty_itae_summary.csv` compares accepted perturbed responses against the same controller's accepted nominal response. Ratios are conditional, with paired counts provided; instability and other failures remain in the acceptance denominator.

## 3. Parameter boundaries

25 plants per targeted group: zero delay, tiny positive delay (1e-6 to 1e-4 seconds), low damping, high damping, low/high gain, low/high natural frequency, and high normalized delay (tau 0.9 to 1, wn 9 to 10). Also test all 16 exact corners of the physical parameter box. All stay within the original scope. New numerical references use the unchanged four-start optimizer; teacher failure is recorded, not replaced. Numerical-reference acceptance therefore includes the ability to obtain a validated label, unlike an oracle with an already supplied label.

| model | group | cases | passes | success_pct |
|---|---|---|---|---|
| random_forest | zero_delay | 25.000 | 23.000 | 92.000 |
| random_forest | tiny_delay | 25.000 | 1.000 | 4.000 |
| random_forest | low_damping | 25.000 | 13.000 | 52.000 |
| random_forest | high_damping | 25.000 | 23.000 | 92.000 |
| random_forest | low_gain | 25.000 | 22.000 | 88.000 |
| random_forest | high_gain | 25.000 | 21.000 | 84.000 |
| random_forest | slow_plant | 25.000 | 14.000 | 56.000 |
| random_forest | fast_plant | 25.000 | 20.000 | 80.000 |
| random_forest | large_normalized_delay | 25.000 | 22.000 | 88.000 |
| random_forest | corners | 16.000 | 10.000 | 62.500 |
| xgboost | zero_delay | 25.000 | 24.000 | 96.000 |
| xgboost | tiny_delay | 25.000 | 1.000 | 4.000 |
| xgboost | low_damping | 25.000 | 14.000 | 56.000 |
| xgboost | high_damping | 25.000 | 24.000 | 96.000 |
| xgboost | low_gain | 25.000 | 20.000 | 80.000 |
| xgboost | high_gain | 25.000 | 19.000 | 76.000 |
| xgboost | slow_plant | 25.000 | 18.000 | 72.000 |
| xgboost | fast_plant | 25.000 | 20.000 | 80.000 |
| xgboost | large_normalized_delay | 25.000 | 21.000 | 84.000 |
| xgboost | corners | 16.000 | 9.000 | 56.250 |
| standard_mlp | zero_delay | 25.000 | 25.000 | 100.000 |
| standard_mlp | tiny_delay | 25.000 | 1.000 | 4.000 |
| standard_mlp | low_damping | 25.000 | 19.000 | 76.000 |
| standard_mlp | high_damping | 25.000 | 25.000 | 100.000 |
| standard_mlp | low_gain | 25.000 | 21.000 | 84.000 |
| standard_mlp | high_gain | 25.000 | 23.000 | 92.000 |
| standard_mlp | slow_plant | 25.000 | 24.000 | 96.000 |
| standard_mlp | fast_plant | 25.000 | 20.000 | 80.000 |
| standard_mlp | large_normalized_delay | 25.000 | 11.000 | 44.000 |
| standard_mlp | corners | 16.000 | 11.000 | 68.750 |
| numerical_reference | zero_delay | 25.000 | 25.000 | 100.000 |
| numerical_reference | tiny_delay | 25.000 | 1.000 | 4.000 |
| numerical_reference | low_damping | 25.000 | 23.000 | 92.000 |
| numerical_reference | high_damping | 25.000 | 25.000 | 100.000 |
| numerical_reference | low_gain | 25.000 | 25.000 | 100.000 |
| numerical_reference | high_gain | 25.000 | 25.000 | 100.000 |
| numerical_reference | slow_plant | 25.000 | 24.000 | 96.000 |
| numerical_reference | fast_plant | 25.000 | 23.000 | 92.000 |
| numerical_reference | large_normalized_delay | 25.000 | 25.000 | 100.000 |
| numerical_reference | corners | 16.000 | 16.000 | 100.000 |

Tiny-delay resolution failures are computational limits, not evidence of physical instability or uncontrollability. Fresh teacher gains and failures are in `boundary_teachers.csv`.

## 4. True delay instead of second-order Padé

An independent method-of-steps simulator integrates the plant and controller integral state with actual delayed feedback. Ideal derivative-on-error is retained: the unit-reference impulse causes the correct velocity jump K*wn^2*Kd at t=tau. The output is exactly zero before the delay. Zero-delay equations are independently checked against the transfer-function solution. DOP853 integration uses two tolerances (1e-7 and 1e-9), two output grids, refined peaks, and horizon extension; acceptance requires ITAE convergence, <=10% overshoot, and tail error <=0.001. Limits: 1,000 seconds, 6,000 delay intervals, 65,537 output points. These are reported separately from the Padé validator's limits.

**A true-delay time-domain pass is not a pole-stability certificate.** A bounded, settled numerical response over converged finite windows is evidence, not a proof about all future times. `observed_divergence` denotes a numerical trajectory exceeding the specified 1e6 state threshold. A delay-segment or integration limit is unassessed, not unstable.

| model | plants | pade_passes | true_delay_time_domain_passes | pade_pass_true_delay_overshoot | median_itae_ratio |
|---|---|---|---|---|---|
| random_forest | 100.000 | 85.000 | 68.000 | 17.000 | 1.014 |
| xgboost | 100.000 | 78.000 | 68.000 | 10.000 | 1.013 |
| standard_mlp | 100.000 | 97.000 | 84.000 | 13.000 | 1.006 |
| numerical_reference | 100.000 | 100.000 | 89.000 | 11.000 | 1.013 |

ITAE ratios compare the same gains under true delay and Padé, conditional on both checks passing. Reasons and measured peaks are in `exact_delay_results.csv`; paired comparisons are in `exact_delay_paired.csv`.

## 5. Input-load disturbance rejection

Starting from the unit-setpoint equilibrium, add a persistent load step of amplitude 0.1/K before the plant (including its delay). Without feedback this corresponds to a steady output change of 0.1. The incremental response uses G/(1+CG), so derivative setpoint kick is not part of this experiment. Keep the same gains. We report maximum absolute output deviation, recovery to +/-0.01 of the setpoint, and disturbance ITAE. Recovery is counted only with stable Padé poles, converged grids/horizons, and tail <=0.001. A setpoint overshoot pass/fail rule is not applied to a disturbance experiment.

| model | cases | recovered | median_peak_deviation | p95_peak_deviation | median_recovery_seconds | median_disturbance_itae |
|---|---|---|---|---|---|---|
| numerical_reference | 100.000 | 100.000 | 0.063 | 0.124 | 2.091 | 0.102 |
| random_forest | 100.000 | 100.000 | 0.061 | 0.124 | 2.146 | 0.103 |
| standard_mlp | 100.000 | 100.000 | 0.060 | 0.124 | 2.119 | 0.100 |
| xgboost | 100.000 | 100.000 | 0.060 | 0.124 | 2.225 | 0.106 |

The table's quality metrics and the plotted distributions condition on recovered cases, with recovery counts shown explicitly. Row-level finite-window metrics for unverified cases are diagnostic only. This is an incremental load applied after equilibrium; it does not prove that every controller initially tracks the setpoint within the overshoot constraint.

## Numerical implementation checks

Five focused tests check: zero-delay agreement with the transfer function including derivative impulse; no anticipatory true-delay output and correct delayed impulse; feedback-free first-delay-interval agreement with the analytical plant; continuity toward zero delay; and disturbance-transfer normalization with and without delay. Run `python -m unittest discover -s tests -p test_stress_suite.py -v`.

At the time of the original stress experiment, the existing suite plus these additions passed 35 tests. A separate scalar, fixed-step RK4 solver with stored-history interpolation independently cross-checked the three largest MLP Padé discrepancies and two numerical-reference discrepancies. Step refinement reached maximum complete-curve error below 0.0001 output units and peak error below 0.01 percentage points against the method-of-steps solver. See `true_delay_crosscheck.csv` and `plots/true_delay_crosscheck.png`. These checks support the numerical comparison without converting time-domain evidence into a stability proof.

## Limits

Single frozen model per family; one 100-plant cohort; noise levels and uncertainty ranges are chosen stress scenarios, not empirically calibrated hardware conditions. No actuator limits, filtered derivative, measurement filtering, anti-windup, or nonlinear plant dynamics are included. Repeated trials do not create additional independent plants. Bounds, computational limits, feature extraction, and overshoot violations must be distinguished. These tests expose weaknesses; they do not certify safety.

## Reproduce

`python scripts/run_stress_suite.py --jobs 4 --plants 100 --output temp/stress_tests`

`python scripts/report_stress_suite.py --input temp/stress_tests`

`python scripts/check_true_delay_stress.py --input temp/stress_tests --output temp/stress_tests`

Rerun the report after the independent cross-check to include its audit. The cross-check requires `nominal_gains.csv` produced by the first report run.

Completed stage CSVs are reused only under an identical saved protocol. The main frozen dataset/models and website are not overwritten.

Numerical-method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

Disturbance-transfer reference: https://www.mathworks.com/help/slcontrol/ug/analyze-the-design-in-the-pid-tuner.html

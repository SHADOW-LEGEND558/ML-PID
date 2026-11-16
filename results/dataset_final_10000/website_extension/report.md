# Final website: additional development evidence

## Learning curves

All 42 points use the same 500 final-validation plants. The first 7,000 training
plants are the frozen final-training cohort (nested seed 45). Larger prefixes add
independent prior-development plants (seed 46); 20,000 training plants uses 13,000
additional development plants. Settings and early stopping rules remain fixed.
The final test partition is not used. Extended fits do not replace final models.

The axis counts TRAINING plants. The chosen 10,000-row final dataset uses 7,000
training plants. MLP acceptance rises 84.6% at 1K, 95.2% at 7K, and 99.6% at 20K.
The 4.4 percentage-point gain from 7K to 20K is real in this validation sample;
10K is a resource/scope tradeoff, not a proven plateau. Gains per added plant are
smaller on average than the initial 1K-to-3K gain. These single-seed development
curves can be optimistic because validation also informs settings and stopping.

```csv
model,1000,3000,5000,7000,10000,15000,20000
linear,30.0,29.0,29.6,30.0,30.0,30.0,30.0
polynomial_ridge,46.8,42.8,44.6,45.6,46.2,45.6,45.6
random_forest,78.2,82.0,84.2,85.6,88.2,90.0,91.6
ridge,33.4,31.4,30.4,30.0,29.8,30.0,30.0
standard_mlp,84.6,94.0,95.0,95.2,97.6,97.8,99.6
xgboost,79.2,83.39999999999999,84.0,86.2,89.60000000000001,88.8,88.8

```

## Rejected-plant stress set

83 rejected final-generation plants are separate from the accepted test cohort.
Original reasons: 57 resolution-only, 25 overshoot-only, 1 mixed. Frozen final
models predict gains without numerical optimization or retraining.

MLP: 21/83 strict passes; 64/83 passes after
resolution-only retries with a 262,145-point cap (versus 65,537 initially).
All gains, tolerances, bounds and the horizon cap are unchanged. Expanded passes
must not be counted as strict passes. Remaining outcomes: {'accepted': 64, 'resolution_limit': 15, 'overshoot': 4}.
There are no accepted reference gains for these plants; gain error and teacher-
relative cost cannot be reported. Rejection is not proof of uncontrollability.

```csv
model,plants,strict_passes,stable,extended_passes
linear,83,4,72,35
polynomial_ridge,83,11,82,43
random_forest,83,7,83,46
ridge,83,4,70,34
standard_mlp,83,21,83,64
xgboost,83,18,82,57

```

## Computational edge

20 fixed validation plants, warm CPU. Numerical optimization plus validation:
median 1.3907 s. MLP prediction: median
0.14415 ms; median matched speedup
9554.0x. Prediction plus verification:
115.913 ms, median matched speedup
11.56x on jointly accepted pairs (19/20).
Feature collection, loading and training are excluded. This sample is for timing,
not a new accuracy estimate. ML amortizes expensive offline labeling but does not
guarantee feasibility or remove the need for physical verification.

## Portable live model

The JavaScript explorer uses the frozen 7K-trained final MLP. Export parity is
checked on 64 validation examples. Physical verification and fresh numerical
comparisons run the same exported Python classes/evaluator in a Pyodide worker.
Feature-only mode predicts gains; a supplied plant is required for simulation.
Inputs stay local. All frozen-model/data hashes remain unchanged.

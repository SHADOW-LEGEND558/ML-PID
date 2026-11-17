# Standard and physics-informed MLP report

## Design and protocol

The MLP is in src/MLP.py. Architecture: 5 -> 128 -> 64 -> 32 -> 3, SiLU activations, dropout, AdamW, positive empirical-normalized outputs. Input standardization, normalized-target scale and log-loss standard deviation are fitted using each fold’s training rows only. Prediction reverses normalization from the five measured inputs; no physical parameters are supplied at inference. Gain upper bounds are checked by validation, not imposed by clipping.

Standard MLP loss is squared log error of normalized q, divided by training-fold log standard deviations per output. Delta=0.001. Physics MLP adds beta times a scaled Hurwitz violation penalty. Beta candidates are 0.1 and 1; beta=0 is the standard comparison. Two regularization pairs are searched: dropout 0.05/weight decay 0.0001, and dropout 0.1/weight decay 0.001. No extra gain penalty or differentiable time-domain simulation is added.

The characteristic polynomial uses the same second-order Padé approximation and tiny-delay cutoff as the shared validator. Exact zero delay uses a cubic; delayed cases use a quintic. Time is normalized by wn. Relative determinants and coefficient scaling make penalty terms dimensionless. The numerical soft margin is 0.001; it is not a physical pole-distance guarantee.

2/800 feasible labels activate the soft numerical margin; their minimum scaled margin is 0.000462374. The penalty is a training preference, not proof of stability.

Three existing development folds cover the same 800 plants as RF/XGBoost. The 200 holdout plants are excluded from all training, stopping, selection and scoring in this experiment. Early stopping uses validation total loss (patience 50, maximum 500 epochs); controller selection uses OOF full-pass count then median accepted J ratio. Checkpoints are selected by early-stopping loss, not by evaluating every epoch in the simulator.

Six seed-42 configurations are compared (three beta values x two regularization pairs). The best standard and best physics configurations are then repeated at seeds 43 and 44. Repeats assess initialization variation on fixed data, not uncertainty from new data splits. Final selected seed-42 models are refit on all 800 development plants for the median best fold epoch and saved as standard_mlp.pt and physics_mlp.pt.

**Fairness/limits:** physics MLP uses privileged physical metadata in its training and validation loss. Standard MLP and the tree regressors do not receive that loss information. Both MLPs use only empirical features at inference. Scores are development/selection scores, not unbiased final test estimates. No claim of hardware or exact-delay safety is made.

## Matched development-fold results (selected seed-42 configurations)

| model | success_pct | stable_pct | r2 | median_J_ratio | passes |
|---|---|---|---|---|---|
| random_forest | 66.250 | 97.625 | 0.822 | 1.203 | 530 |
| xgboost | 69.625 | 99.250 | 0.628 | 1.218 | 557 |
| standard_mlp | 81.625 | 100.000 | 0.945 | 1.142 | 653 |
| physics_mlp | 81.875 | 100.000 | 0.945 | 1.143 | 655 |

J ratios include only accepted prediction/reference pairs; the pass count is their coverage. Failed controllers stay in overall success denominators. Different accepted subsets mean ratio medians alone cannot rank overall quality.

## Selected MLP settings

| variant | beta | dropout | weight_decay | median_best_epoch | success_pct |
|---|---|---|---|---|---|
| standard | 0.000 | 0.050 | 0.000 | 484 | 81.625 |
| physics | 1.000 | 0.050 | 0.000 | 484 | 81.875 |

## Initialization sensitivity

| variant | seeds | mean_success_pct | std_success_pp | min_success_pct | max_success_pct |
|---|---|---|---|---|---|
| physics | 3 | 80.042 | 1.593 | 79.000 | 81.875 |
| standard | 3 | 80.458 | 1.078 | 79.500 | 81.625 |

## Sample-size learning curves

For each selected MLP configuration at seed 42, train on nested 128 and 256 subsets and compare with the full fold (533/534 plants). The same development validation folds and subset permutations as the tree curves are used. Input/target scaling is fitted separately on each subset. Early stopping remains enabled. Settings were selected at full size, so these curves are descriptive, not independently tuned or unbiased final scores.

| variant | training_size | success_pct | stable_pct | median_J_ratio | r2 |
|---|---|---|---|---|---|
| physics | 128 | 67.125 | 99.750 | 1.301 | 0.861 |
| physics | 256 | 74.375 | 100.000 | 1.230 | 0.876 |
| physics | 533 | 81.875 | 100.000 | 1.143 | 0.945 |
| standard | 128 | 67.375 | 99.750 | 1.301 | 0.860 |
| standard | 256 | 74.000 | 100.000 | 1.229 | 0.875 |
| standard | 533 | 81.625 | 100.000 | 1.142 | 0.945 |

## Overshoot severity

| model | at_most_10 | over_10_to_12 | over_12_to_15 | over_15_to_20 | over_20 | unstable | stable_unmeasured |
|---|---|---|---|---|---|---|---|
| random_forest | 530 | 35 | 53 | 52 | 110 | 19 | 1 |
| xgboost | 557 | 37 | 46 | 55 | 99 | 6 | 0 |
| standard_mlp | 653 | 38 | 45 | 47 | 17 | 0 | 0 |
| physics_mlp | 655 | 35 | 44 | 47 | 19 | 0 | 0 |

Rejected-controller overshoot values are finite-window diagnostics, not convergence-certified values. Full per-gain errors, raw predictions, failures, epoch histories, candidate scores and seed scores are saved alongside this report.

## Interpretation and recommendation

The standard MLP averages 80.46% full-validation success over three seeds; the physics variant averages 80.04%. Both are materially above the matched tree development results, but these are still selection/development estimates. The tiny seed-42 physics advantage is not consistent across seeds. Prefer the standard MLP as the simpler next candidate; the current stability penalty has not demonstrated additional benefit.

The MLP and tree models differ in training loss and output parameterization, not just architecture: MLP uses standardized log loss and positive outputs. This experiment does not isolate which component causes the improvement. The physics penalty addresses pole stability, not the 10% overshoot requirement. Residual controller failures still matter, and no model should be treated as a guaranteed tuner.

Training uses a 500-epoch cap; several folds choose the final epoch, so this is a bounded-budget comparison rather than evidence of complete optimization convergence. Further tuning must stay within development data. Before claiming final generalization, freeze settings and use an independent evaluation set; the existing 200-plant holdout was excluded here but has been inspected in earlier work.

## Verification

All 27 regression tests passed. Polynomial coefficients were compared against the shared simulator. Scaled Hurwitz signs were checked against pole stability on sampled zero/delayed plants, and gradients were checked for finiteness. Checkpoint round trips, safe weights-only loading, and positive predictions were tested. All OOF tables were audited to exclude every holdout plant. The executed notebook renders its figures and reproduces all 12 illustrated saved validation outcomes.

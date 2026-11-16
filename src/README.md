# Empirical PID baselines

Run from the repository root after installing `requirements.txt`:

```powershell
python src/Regression.py
python src/XGBoost.py
python -m unittest discover -s tests -v
```

Both commands accept `--data`, `--seed`, `--test-size`, `--cv`, and `--output`.
Default outputs are `results/regression` and `results/trees`.

Models receive exactly five features, in this order: `t_rise`, `t_settling`,
`y_peak`, `y_ss`, `tau_est`. Physical parameters and observation duration are
metadata for simulation and auditing, never training inputs. AUC is removed.
MLP is not part of either benchmark.

`benchmark.py` owns loading, plant-grouped splitting, cross-validation, evaluation,
and export. Identical plant parameter tuples stay together. Both families use the
same deterministic split for the same CSV, seed, and test fraction. All scaling
and polynomial expansion are fitted inside training folds. Hyperparameters are
selected by training-only grouped CV using mean squared gain error; the test set
is evaluated afterward. Trees do not need scaling. XGBoost is required explicitly,
with no replacement algorithm if its installation is missing.

Outputs include per-gain MAE, RMSE, and R², uniformly averaged R², CV scores,
per-plant raw predictions and failure reasons, a split manifest with dataset hash,
and fitted `.joblib` pipelines with feature order and selected parameters. Load
only trusted model files. To predict, load the bundle with joblib, select its
`features` columns, and call `bundle['pipeline'].predict(frame[bundle['features']])`.

Control evaluation uses raw predictions without clipping or fallback. Reports
include stable percentage, gain-bound violations, and full validated success.
The measured overshoot percentage is diagnostic: finite-window measurements on
rejected controllers are not validation certificates. Missing measurements count
as nonpasses, and coverage is reported. ITAE ratios compare against saved target
controllers only when both controllers pass revalidation; the paired count is
reported to expose selection effects. A ratio below 1 can occur because targets
come from local searches. All checks use ideal PID and second-order Padé delay,
not exact-delay or hardware guarantees.

Open-loop recording starts at 10 s and doubles up to 640 s, with 0.01 s samples.
The final 20% must vary by <=0.2% of its mean, and its two halves must agree within
0.1%. Extraction fails if it cannot settle within that limit. This rule uses the
observed output, rather than plant poles, to choose the window. Final value remains
an estimate. Onset includes early rise, not just dead time. The optimization grid
is separate from this observation grid. CSVs include `observation_duration` for
auditing. The existing 100-row dataset has been migrated to this feature rule;
its saved PID target gains were retained.

New generation draws exact zero delay with probability 0.1 by default; change it
with `python src/data_gen.py --zero-delay-fraction 0.1`. Remaining delays are uniform
between 0 and 1. Rejection and replacement can change the accepted proportion.
Existing nonzero delays were retained during migration: generate a new dataset
or a separate stress set to evaluate the exact zero-delay boundary.

The current small dataset is a smoke test, not evidence of reliable deployment.
Keep the final test set fixed as you iterate. Larger data and dedicated boundary
tests are needed before making performance claims.

## Regularized tuning update

The generator now minimizes `wn^2*ITAE + lambda*((K*Kp)^2 + (K*Ki/wn)^2 + (K*wn*Kd)^2)`.
The provisional default is `lambda=0.01`; use `--gain-penalty` to change it or 0
for unregularized tuning. This is a dimensionless gain penalty, not actuator energy.
Bounds and the 10% overshoot constraint remain. Accepted starts are compared using
the combined objective, while raw ITAE is retained for interpretation. Stability,
peak refinement, grid and horizon convergence checks are unchanged.

`python scripts/compare_gain_penalty.py` reproduces the 16-plant, four-weight
comparison in `results/gain_penalty/`. This small design comparison does not
establish a universally optimal weight. New dataset generation writes a `.meta.json`
sidecar containing the objective, weight, seed, and dataset hash.

The current `data/pid_dataset.csv` now contains 1,000 newly generated regularized
labels with lambda=0.01 (seed 20261116), including 86 exact zero-delay plants.
All 1,000 accepted rows passed generator validation; 7 failed attempts were replaced.
No accepted gains hit an upper bound within absolute tolerance 1e-5. The prior
unregularized dataset is archived as `data/pid_dataset_unregularized_1000.csv`.
The `.meta.json` sidecar identifies the current objective and dataset hash.
Existing model results remain historical unregularized results; retrain to evaluate
the new labels.

## Generation performance

The 1,000-row regularized run took about 6 minutes on an i5-13450HX using 12
processes and one numerical thread per process. No optimizer or validation settings
were loosened. Set thread variables before launching Python:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python src/data_gen.py --samples 1000 --jobs 12 --starts 4 --gain-penalty 0.01 --seed 20261116 --output data/pid_dataset.csv
```

This command overwrites the specified output on success; use a new output filename
to retain an existing dataset. Worker count is hardware-dependent; more workers do
not automatically improve throughput. There are no source-level speed optimizations
in this run, and GPU acceleration is not implemented for this SciPy optimizer.

## Current 1,000-row baseline results

The five models have now been retrained on the regularized dataset. Current results
are in `results/regularized_1000/report.md`, with combined metrics in `metrics.csv`
and family-specific artifacts in `regression/` and `trees/`. Earlier result folders
remain historical. The current run uses 800 training and 200 held-out plants, with
three-fold training-only grouped CV scored by gain MSE. All five pipelines use the
same holdout and only the five empirical inputs.

The benchmark requires dataset metadata with a matching SHA-256 hash and reads
lambda from that metadata. It now reports both validated raw ITAE ratios and the
regularized objective J ratios, only for accepted prediction/reference pairs,
with explicit coverage counts. Failed controllers stay in full-pass denominators.

```powershell
python src/Regression.py --output results/regularized_1000/regression
python src/XGBoost.py --output results/regularized_1000/trees
```

These commands overwrite those result folders' artifacts. Preserve the held-out
set for final reporting and use training-only validation for further tuning.

## Training-only control experiments

Run `python scripts/run_control_experiments.py` from the repository root, followed
by `python scripts/report_control_experiments.py`. This reruns 38 raw/normalized
configurations with three development folds and 10 smaller-subset learning-curve
configurations; it is substantially more expensive than training the baselines.
Outputs are retained in `results/control_experiments`, separately from the earlier
baseline results. The existing 200 held-out rows are explicitly excluded.

`control_experiments.EmpiricalTargetRegressor` learns empirically normalized targets
and reverses the transformation using only the five features at inference time.
The model wrapper is saved in each selected bundle; no preprocessing step should
be bypassed when loading it. Models are selected on full controller validation
success, then median accepted J ratio, then raw gain MSE. These development scores
are selection-biased and are not new final-test performance estimates.

All 48 saved out-of-fold tables were checked to contain exactly the 800 development
plants and none of the 200 excluded test plants. All selected bundles predict
successfully after loading. The complete 24-test regression suite passed.

## Standard and physics-informed MLP

`MLP.py` implements a positive-output 5-128-64-32-3 PyTorch MLP. It standardizes
inputs using training-fold statistics and learns empirical-normalized q targets
with standardized squared log loss. Physical gains are decoded before evaluation;
no predicted gains are clipped. Gradient clipping affects neural optimization
only. `beta=0` is the standard MLP; positive beta adds scaled Hurwitz violations
using the shared second-order Padé model and exact-zero-delay cubic. The penalty
is soft and does not enforce overshoot or certify stability.

Physical plant metadata is used only by the physics training/validation loss and
by the shared controller evaluator, never by inference. This is additional
training information relative to standard supervised baselines. No extra gain
regularizer is added because target generation already regularizes the gains.

```powershell
python scripts/train_mlp.py
python scripts/mlp_learning_curves.py
python scripts/report_mlp.py
```

Training uses the existing three development folds, six configurations at seed 42,
and two additional seeds for each selected variant. It uses CPU execution, early
stopping (patience 50, max 500 epochs), AdamW, and deterministic seeds. Selection
uses controller validation pass count then accepted J ratio. Test rows are
explicitly excluded. Final selected seed-42 variants are refitted on all 800
development plants for their median best stopping epoch and saved under
`results/mlp_experiments`. These commands overwrite that experiment's artifacts.

```python
from MLP import MLPModel
model = MLPModel.load('results/mlp_experiments/standard_mlp.pt')
gains = model.predict(frame[['t_rise', 't_settling', 'y_peak', 'y_ss', 'tau_est']])
```

The checkpoint contains fitted scaling, architecture and weights. Use the provided
loader rather than bypassing preprocessing. `notebooks/mlp_validation.ipynb` reads
saved outputs without retraining and compares MLPs with matched RF/XGBoost folds.
Seed repetitions reflect initialization variation, not new-data uncertainty. The
200 previously inspected holdout plants remain excluded from these experiments.

MLP sample-size curves reuse the selected seed-42 configurations on nested 128/256 subsets with the same validation folds and subset permutations as the tree experiments. These are development curves; the held-out 200 plants remain excluded.

## Preferred larger-data experiment (20,000 plants)

The new experiment uses `data/pid_dataset_20000.csv`, not the older default
`data/pid_dataset.csv`. Exact split: 15,000 training / 2,500 validation / 2,500
final test plants, stratified by zero-delay status. New results are separate in
`results/dataset_20000`. Earlier 1,000-row data, models and notebook sections remain.

```powershell
python scripts/train_20000.py
python scripts/report_20000.py
python scripts/append_20000_notebooks.py
```

The first command is a lengthy CPU training/simulation run and overwrites this
experiment's result directory. It freezes settings before scoring the test set;
do not retune based on those test scores. Learning curves use fixed nested random
training subsets and the same 500 validation plants, never test plants. Model
selection and MLP stopping use validation only. All models use normalized targets;
the larger-batch standard MLP has no physics penalty.

Open `notebooks/baseline_validation.ipynb` and `notebooks/mlp_validation.ipynb` and
scroll to **New experiment: 20,000 plants** at the end. These sections include raw
ITAE scatter/ratios/percentage changes, J ratios, gain errors, failure/overshoot
plots, learning curves, test responses, and repeat validation of example gains.
New model files and their preprocessing are in `results/dataset_20000`; the full
report is `results/dataset_20000/report.md`. Older scripts still target the older
experiments unless explicitly configured otherwise.

## Separate rejected-plant stress experiment

`python scripts/stress_rejected_plants.py` recovers all 142 plant records from
`data/generation_20000.log`, recomputes their empirical features, and predicts
with the six frozen 20k models. It does not train or optimize gains. Original
validation uses 65,537 points; only resolution-limit predictions are retried
with 262,145 points, with all other criteria unchanged. Both outcomes are saved
separately under `results/rejected_plants`.

Run `python scripts/report_rejected_plants.py` for the report/figures. The executed
notebook is `notebooks/rejected_plants_stress_test.ipynb`. There are no accepted
optimized gain labels for these plants: gain errors and reference ITAE ratios
cannot be computed. This is a rejection-biased stress set, not a replacement for
the main test results. Plant-level original groups are 102 resolution-only,
36 overshoot-only, and 4 mixed; start-level reason counts must not be mistaken
for plant counts. All 30 tests passed, model/dataset hashes stayed unchanged,
and all four notebook examples reproduced their strict/secondary outcomes.

# ML-PID · empirical PID auto-tuning

**Authors:** Aawish K Sudhish · Sai Srinivas Acholi  
**ES 335 undergraduate project**

Can five measurements of an open-loop response predict useful PID gains faster than repeated numerical tuning?

[Website](https://shadow-legend558.github.io/ML-PID/) · [Model notebooks](model_testing/) · [Published results](results/dataset_final_10000/report.md) · [Stress tests](results/dataset_final_10000/stress_tests/report.md)

## Setup

Use 64-bit **Python 3.12**, the version used for verification. From the repository root:

```text
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows, or `source .venv/bin/activate` on macOS/Linux, then:

```text
python -m pip install -r requirements-notebooks.txt
python -m unittest discover -s tests -v
```

Core versions are pinned in `requirements.txt`; notebook dependencies are in `requirements-notebooks.txt`. No GPU, API key or model download is needed for scientific reproduction.

## Study and reproduce the models

Open a notebook in [model_testing/](model_testing/), select the project environment and Run All:

| Notebook | Purpose |
|---|---|
| [linear_regression.ipynb](model_testing/linear_regression.ipynb) | Linear baseline |
| [ridge_regression.ipynb](model_testing/ridge_regression.ipynb) | Ridge baseline |
| [polynomial_ridge.ipynb](model_testing/polynomial_ridge.ipynb) | Nonlinear polynomial baseline |
| [random_forest.ipynb](model_testing/random_forest.ipynb) | Random Forest baseline |
| [xgboost.ipynb](model_testing/xgboost.ipynb) | XGBoost baseline |
| [mlp.ipynb](model_testing/mlp.ipynb) | Preferred MLP |
| [results_analysis.ipynb](model_testing/results_analysis.ipynb) | Compare all six published models and inspect supplemental evidence |

The training notebooks use the **same fixed 7,000 training / 1,500 validation / 1,500 test** rows from the final 10K CSV. They fit the published selected configurations, display validation/test gain errors, simulate all 1,500 predicted test controllers, and plot responses, overshoot and ITAE. Models and new result tables stay in memory; no checkpoints, CSVs or external figures are saved. Save the notebook to retain its cell outputs.

The analysis notebook reads published predictions and experiment records without retraining. This separates the reported frozen study from a new reconstruction. Settings were selected before the original test; do not retune them using the reconstructed test results. Floating-point values and timings can vary across platforms. Training plus full closed-loop evaluation takes minutes per model.

## Reproduce the scientific tests

Scientific runners reconstruct the selected models from the CSV **in memory**, so an original Python checkpoint archive is not required. Their new experiment files go under ignored `temp/`, keeping published evidence separate. Run from the repository root.

Five robustness tests—measurement noise, plant uncertainty, parameter boundaries, true delay and load disturbance:

```text
python scripts/run_stress_suite.py --jobs 4 --output temp/stress_tests
python scripts/report_stress_suite.py --input temp/stress_tests
python scripts/check_true_delay_stress.py --input temp/stress_tests --output temp/stress_tests
python scripts/report_stress_suite.py --input temp/stress_tests
```

This repeats the fixed 100-plant cohort and 241 boundary plants. It is substantially more expensive than unit tests and can take many minutes. Models are reconstructed once before the tests; there is no stress-case adaptation. The method-of-steps delay solver has a separate fixed-step RK4 cross-check. Completed stress stages are reused only under a matching protocol, including fitted-model fingerprints, implementation hashes and dependency versions; use a new output path when changing the experiment.

Other reported experiments:

```text
python scripts/final_controller_diagnostics.py --jobs 4 --output temp/controller_diagnostics
python scripts/extend_final_study.py --jobs 4 --output temp/additional_experiments
python scripts/benchmark_final_speed.py --output temp/timing
```

These reproduce complete response metrics for stable overshoot misses, common-validation learning curves through 20K training plants, the final 83 rejected plants, and model-versus-optimizer timing. Timing excludes model reconstruction and feature recording. The diagnostic analysis uses saved test gains; extended experiments reconstruct models where prediction is needed. Longer learning curves can take several minutes.

Optional historical method checks supporting the penalty/loss choices:

```text
python scripts/compare_gain_penalty.py --samples 200 --weights 0.001 0.01 --jobs 4 --output temp/gain_penalty
python scripts/compare_physics_loss.py --jobs 4 --output temp/physics_comparison
```

The penalty comparison uses matched plants. The physics comparison repeats the selected standard/physics-loss variants over three seeds and three recorded folds on the earlier 1K development cohort, excluding its held-out rows. These are separate development experiments, not the final 10K test result.

## Full data generation and model selection

To check the supplied dataset and reproduce the original validation-based model selection in a new working directory:

```text
python scripts/prepare_final_dataset.py --data data/pid_dataset_final_10000.csv --output temp/final_study --jobs 4
python scripts/train_final_10000.py --output temp/final_study --jobs 4
python scripts/report_final_10000.py --output temp/final_study
```

This CLI path writes intermediate histories, checkpoints and experiment records only in the chosen output directory. It refuses to overwrite a locked/scored study. The individual notebooks remain the simpler route and save no external models/results.

To generate new labels first:

```text
python src/data_gen.py --samples 10000 --jobs 4 --starts 4 --gain-penalty 0.01 --zero-delay-fraction 0.1 --seed 8675309 --output temp/generated_10000.csv
```

Substitute that CSV in the preparation command. Label generation can take hours. New optimization labels and timing are not guaranteed to be bit-identical on every machine.

## Read the website locally

```text
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/. The static site includes its saved evidence and original portable browser models; scientific Python runners are not its backend. Optional live simulation downloads pinned Pyodide/NumPy/SciPy on first use. The site needs no build, exporter, npm installation or model training. Publication is owner-controlled.

## Scientific scope and interpretation

- One row is one independently simulated plant, five empirical measurements and three best-found numerical PID gains. The final dataset has **10,000 accepted plants from 10,083 attempts**.
- Plant bounds: K 0.5–2, natural frequency 0.5–10 rad/s, damping 0.1–2, delay 0–1 s. Exact zero delay is sampled with probability 10%.
- Five required inputs: `t_rise`, `t_settling`, `y_peak`, `y_ss`, `tau_est`. Physical parameters are simulation/audit metadata, not privileged model inputs. Targets are normalized using measured final output and rise time; learned scaling uses training rows only.
- Labels minimize `J = wn^2*ITAE + 0.01*((K*Kp)^2 + (K*Ki/wn)^2 + (K*wn*Kd)^2)` subject to at most 10% overshoot and gain bounds. Four-start SLSQP gives local best-found solutions, not certified global optima. The penalty is not actuator energy.
- Open-loop recording uses analytical true delay. Optimization/main closed-loop evaluation uses ideal PID, a unit setpoint and second-order Padé delay. Padé enables rational LTI simulation and pole checks; it is not an exact-delay guarantee.
- Published MLP strict acceptance is **1,406/1,500 (93.73%)**. All 1,500 MLP responses are stable/resolved under Padé. Exploratory coverage is 97.8% at 12% overshoot, without changing the original strict flags. Small specification misses can retain useful tracking performance.
- Gain error alone is insufficient: controller tests check bounds, poles, overshoot, tail error and numerical convergence. ITAE/J ratios condition on accepted pairs, with coverage shown.
- The same 100-plant cohort gives 97 MLP Padé passes versus 84 true-delay time-domain passes. Noise, high normalized delay and finite numerical budgets expose additional weaknesses. Finite-window convergence is not an all-time stability proof. There is no hardware/actuator validation or missing-feature fault tolerance.

## Repository map

| Folder | Purpose |
|---|---|
| `src/` | Reusable scientific models, simulation, evaluation, normalization, plotting and fitting helpers |
| `model_testing/` | Six training notebooks and one consolidated results-analysis notebook |
| `scripts/` | Scientific experiment and reporting commands |
| `tests/` | Numerical, data, split, model and notebook-contract tests |
| `data/` | Final data/metadata plus development data required by recorded experiments |
| `results/dataset_final_10000/` | Published protocols, predictions, scores and supplemental scientific evidence |
| `docs/` | Buildless website and portable browser assets |
| `simulator/` | Optional earlier teaching simulator |
| `temp/` | Ignored scratch files, caches and newly reproduced experiment outputs |

Old website checks/exporters, one-time migration scripts and superseded model wrappers have been removed. Archived development notebooks, original local checkpoint copies and obsolete website/cleanup audit records have also been removed. Published scientific inputs and results remain available; current runners reconstruct the models when needed.

## AI assistance disclosure

We used OpenAI ChatGPT, Codex and Google Gemini to discuss control-system concepts and experimental choices, assist with writing and debugging code, develop notebooks and website visualizations, and improve documentation.

AI assistance contributed to the data-generation pipeline, model training and evaluation workflows, and stress-test implementation. The resulting work was checked through automated tests, numerical simulations, comparisons with recorded results, and independent true-delay simulation checks.

The authors are responsible for the project's assumptions, methodology, reported results and conclusions. The dataset is generated through numerical plant simulation and PID optimization; it is not generated by a trained machine-learning model.

## References

[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html), [SciPy integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/), [PyTorch](https://docs.pytorch.org/docs/stable/nn.html).

# Model training and results analysis

This folder contains one notebook per model: linear regression, ridge, polynomial ridge, Random Forest, XGBoost and MLP. Each trains from the final 10K CSV using the existing 7,000/1,500/1,500 plant split, displays validation/test gain errors, simulates every predicted test controller, and displays response plots and ITAE/overshoot analysis. Models and newly computed tables remain in memory; only notebook cell outputs are retained when saving the notebook. No checkpoint archive is needed. Settings are fixed to the published study; retraining can differ slightly across environments.

| Model | Notebook |
|---|---|
| Linear regression | [linear_regression.ipynb](linear_regression.ipynb) |
| Ridge | [ridge_regression.ipynb](ridge_regression.ipynb) |
| Polynomial ridge | [polynomial_ridge.ipynb](polynomial_ridge.ipynb) |
| Random Forest | [random_forest.ipynb](random_forest.ipynb) |
| XGBoost | [xgboost.ipynb](xgboost.ipynb) |
| MLP — preferred model | [mlp.ipynb](mlp.ipynb) |

[results_analysis.ipynb](results_analysis.ipynb) is the single comparison notebook. It brings together all six models' gain errors, controller acceptance, ITAE/overshoot analysis, extended learning curves, MLP training history, shared response examples and saved stress-test evidence. It reads published results and does not retrain or load saved checkpoints.

Install `requirements-notebooks.txt`, select the project virtual environment as kernel, and Run All. Paths are resolved from the repository, without a personal absolute path. The analysis notebook reads saved results; the six model notebooks retrain independently. Neither route overwrites published test scores. Executed outputs are included.

The separate `simulator/` notebook is an earlier teaching aid and may need `ipywidgets` in addition to the core environment.

Scientific runners under `scripts/` repeat the supplemental tests using models reconstructed in memory. Exact commands are in the root README. Shared scientific code is under `src/`; there is no dependency on website exporters or original saved Python checkpoints.

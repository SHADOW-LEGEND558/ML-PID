"""Append the new experiment while preserving all earlier notebook sections."""
from pathlib import Path
import nbformat

MARKER='# New experiment: 20,000 plants'


def append(path,mlp):
    notebook=nbformat.read(path,as_version=4)
    previous=next((i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith(MARKER)),len(notebook.cells))
    notebook.cells=notebook.cells[:previous]
    cells=[]
    def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):cells.append(nbformat.v4.new_code_cell(text))
    names=['random_forest','xgboost','standard_mlp'] if mlp else ['linear','ridge','polynomial_ridge','random_forest','xgboost']
    prefix='large_mlp' if mlp else 'large_base'
    md(MARKER+'''

The earlier notebook sections above are preserved and describe the old 1,000-row experiment.
**Everything below uses `data/pid_dataset_20000.csv` and the frozen 15,000 training / 2,500 validation / 2,500 test split.**

The final test set was scored after settings and weights were frozen. These are new test results, not the earlier development-fold scores. No post-test tuning is performed.
All models use the same five empirical features and normalized targets. The standard MLP uses log-space supervised loss; the physics loss is disabled.

Restart the kernel and Run All to refresh imports and saved results. No model training is run by this notebook.''')
    code(f'''from pathlib import Path
import sys
large_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "scripts" / "report_20000.py").exists())
sys.path.insert(0, str(large_root / "scripts"))
sys.path.insert(0, str(large_root / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from large_experiment import load_models_results, OUT as LARGE_RESULTS
from report_20000 import learning_plot, mlp_history_plot
from plot_baselines import (performance_plot, prediction_plot, failure_plot, objective_plot,
                           itae_plot, overshoot_plot, overshoot_bins_plot, overshoot_summary,
                           example_rows, response_plot, revalidate)
{prefix}_names = {names!r}
{prefix}_metrics, {prefix}_predictions, {prefix}_protocol = load_models_results({prefix}_names)
print("Training:", len({prefix}_protocol["train"]), "Validation:", len({prefix}_protocol["validation"]), "Test:", len({prefix}_protocol["test"]))
print("Dataset SHA-256:", {prefix}_protocol["dataset_sha256"])
display({prefix}_metrics[["success_pct", "stable_pct", "bounds_violation_pct", "r2", "median_J_ratio", "ratio_pairs"]].round(3))''')
    md('''## New test performance and gain predictions

Gain errors compare decoded physical PID gains with optimized targets. Full validation additionally checks bounds, stable poles, refined overshoot <=10%, tail error <=0.001, and ITAE convergence. No prediction clipping or fallback is applied.''')
    code(f'fig = performance_plot({prefix}_metrics)\nplt.show()')
    code(f'fig = prediction_plot({prefix}_predictions)\nplt.show()')
    md('''## ITAE: how much did tracking change from the optimized controller?

The scatter compares raw ITAE on matching plants; the diagonal means equality.
The ratio is `ITAE_predicted / ITAE_target`; 1 means equal. Percentage change is `100*(ratio-1)`; 0% means equal and +20% means 20% larger time-weighted error.

**Only validated prediction/reference pairs are plotted**, with counts shown. Failed/unstable/unresolved controllers remain failures and are never assigned zero error. Raw ITAE and regularized J are different: lower ITAE alone may require larger gains and need not improve J. Target gains come from local optimization, so a prediction can occasionally improve upon them.''')
    code(f'''fig = itae_plot({prefix}_predictions)
plt.show()
display(pd.read_csv(LARGE_RESULTS / "itae_summary.csv").query("model in @{prefix}_names").round(3))''')
    md('''## Regularized objective J

J includes normalized ITAE plus the squared dimensionless-gain penalty at lambda=0.01. Ratios use accepted pairs only and their coverage differs by model. Do not rank models by these conditional medians alone.''')
    code(f'fig = objective_plot({prefix}_predictions)\nplt.show()')
    md('''## Failures and overshoot severity

Below-limit overshoot alone is not full validation. Stable finite-window measurements for rejected controllers are diagnostic; missing/unstable cases are counted separately.''')
    code(f'fig = failure_plot({prefix}_predictions)\nplt.show()\nfig = overshoot_bins_plot({prefix}_predictions)\nplt.show()')
    code(f'fig = overshoot_plot({prefix}_predictions)\nplt.show()\ndisplay(overshoot_summary({prefix}_predictions).round(3))')
    md('''## Learning curves from the larger training set

Each model uses the same nested random training subsets (1,000 / 5,000 / 10,000 / 15,000) and fixed selected settings.
The curves use the **same 500 stratified validation plants**, not the test set. The MLP uses the full 2,500 validation set for early stopping at every size. These are development diagnostics; settings and stopping used validation data.''')
    code(f'''fig = learning_plot({prefix}_names)
plt.show()
display(pd.read_csv(LARGE_RESULTS / "learning_curves.csv").query("model in @{prefix}_names")[["model", "training_size", "success_pct", "r2", "median_J_ratio"]].sort_values(["model", "training_size"]).round(3))''')
    if mlp:
        md('''### Standard MLP training history

The architecture remains 5-128-64-32-3. Batch size is 256 for efficient training on 15,000 plants. Early stopping restores the best validation-loss checkpoint. Input and target scales come only from training data.''')
        code('fig = mlp_history_plot()\nplt.show()')
    md('''## Shared test-plant response examples

Examples are selected transparently from random forest results: median accepted cost ratio and median stable overshoot failure. They are illustrations, not a balanced test sample. Display windows can be shorter than full validation horizons.''')
    code(f'{prefix}_examples = example_rows({prefix}_predictions)\ndisplay(pd.Series({prefix}_examples, name="20k CSV row ID"))')
    code(f'fig = response_plot({prefix}_predictions, {prefix}_examples["Representative RF success"])\nplt.show()')
    code(f'fig = response_plot({prefix}_predictions, {prefix}_examples["Representative RF overshoot failure"])\nplt.show()')
    code(f'''print("Some available NEW test row IDs:", {prefix}_protocol["test"][:25])
# To explore a different held-out plant, choose one of the new test IDs:
# new_source_row = {prefix}_protocol["test"][0]
# fig = response_plot({prefix}_predictions, new_source_row)
# plt.show()''')
    md('''### Recheck the illustrated predictions

This repeats the shared simulation checks without fitting models or changing settings. It is nominal ideal-PID/Padé validation, not a hardware or exact-delay guarantee.''')
    code(f'''{prefix}_checks = revalidate({prefix}_predictions, {prefix}_protocol,
    source_rows=[{prefix}_examples["Representative RF success"], {prefix}_examples["Representative RF overshoot failure"]])
display({prefix}_checks)
print("Matching saved outcomes:", {prefix}_checks.matches.sum(), "/", len({prefix}_checks))''')
    md('''The full new-data report is at `results/dataset_20000/report.md`.
The source CSV and all earlier experiments remain unchanged. To reproduce this new experiment, run `python scripts/train_20000.py`, `python scripts/report_20000.py`, then `python scripts/append_20000_notebooks.py`. Training commands overwrite this experiment's result directory.''')
    notebook.cells+=cells;nbformat.write(notebook,path)
    print('Appended new-data section:',path)


if __name__=='__main__':
    append(Path('notebooks/baseline_validation.ipynb'),False)
    append(Path('notebooks/mlp_validation.ipynb'),True)

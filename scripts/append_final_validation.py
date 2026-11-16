"""Append executed final-cohort sections; preserve all earlier notebook cells.

Only the appended sections are executed. They read frozen result artifacts and
draw plots; they do not fit models, select settings or rescore the final test set.
"""
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TAG = "final_10000"


def section(names, mlp=False):
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md("# Final study: fresh 10,000-plant dataset\n\n"
        "This section is separate from every earlier experiment above. The dataset has **7,000 training / 1,500 validation / 1,500 held-out test plants**. "
        "All models use the same five empirical inputs and normalized gain targets. Hyperparameters and model weights were locked before test scoring. "
        "These cells load the saved final results; they never retrain or select a model from test outcomes."),
        code("%matplotlib inline\nfrom pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
             "FINAL_ROOT = Path.cwd()\nif not (FINAL_ROOT / 'src').is_dir():\n    FINAL_ROOT = FINAL_ROOT.parent\n"
             "sys.path.insert(0, str(FINAL_ROOT / 'scripts'))\nsys.path.insert(0, str(FINAL_ROOT / 'src'))\n"
             "from report_final_10000 import load_final, learning_plot as final_learning_plot, history_plot as final_history_plot\n"
             "from plot_baselines import (style, performance_plot, prediction_plot, failure_plot, objective_plot, itae_plot,\n"
             "    overshoot_plot, overshoot_bins_plot, example_rows, response_plot)\n"
             f"final_names = {names!r}\n"
             "FINAL_RESULTS = FINAL_ROOT / 'results' / 'dataset_final_10000'\n"
             "final_metrics, final_predictions, final_protocol = load_final(final_names)\nstyle()\n"
             "display(final_metrics[['passes','plants','success_pct','stable_pct','r2','median_ITAE_ratio','median_J_ratio','ratio_pairs']])\n"
             "display(pd.read_csv(FINAL_RESULTS / 'selected_before_test.csv').query('model in @final_names')"
             "[['model','setting','success_pct','median_J_ratio']])\n"
             "print('Dataset SHA-256:', final_protocol['dataset_sha256'])"),
        md("## Regression accuracy and controller acceptance\n\n"
           "Gain errors use all 1,500 test plants, including failed controllers. Acceptance requires finite bounded gains, stable poles, "
           "refined overshoot ≤10%, a settled tail and resolved numerical checks. Predictions are evaluated without clipping or a fallback."),
        code("# Gain agreement does not by itself certify a usable controller.\n"
             "for final_plotter in [performance_plot, prediction_plot, failure_plot]:\n"
             "    final_figure = final_plotter(final_metrics if final_plotter is performance_plot else final_predictions)\n"
             "    plt.show()\n    plt.close(final_figure)\n"
             "display(pd.read_csv(FINAL_RESULTS / 'gain_errors.csv').query('model in @final_names'))"),
        md("## ITAE and regularized cost relative to numerical tuning\n\n"
           "**ITAE ratio = predicted ITAE / reference ITAE.** One means equal tracking cost. Percentage change is 100 × (ratio − 1). "
           "Ratios use only pairs where both controllers pass; coverage is shown and differs across models. Inspect upper tails as well as medians. "
           "The teacher minimizes J = ωₙ² ITAE + 0.01 times a dimensionless squared-gain penalty, so lower raw ITAE need not imply lower J."),
        code("for final_plotter in [itae_plot, objective_plot]:\n"
             "    final_figure = final_plotter(final_predictions)\n    plt.show()\n    plt.close(final_figure)\n"
             "display(pd.read_csv(FINAL_RESULTS / 'itae_summary.csv').query('model in @final_names'))"),
        md("## Overshoot severity and error groups\n\n"
           "The threshold curve uses **all 1,500 test plants** as its denominator. Unstable and unmeasured responses are separate, not assigned zero overshoot. "
           "Overshoot below 10% alone is not full acceptance. Physical delay/damping groups are descriptive audits, not prediction inputs or grounds for retuning."),
        code("for final_plotter in [overshoot_plot, overshoot_bins_plot]:\n"
             "    final_figure = final_plotter(final_predictions)\n    plt.show()\n    plt.close(final_figure)\n"
             "display(pd.read_csv(FINAL_RESULTS / 'overshoot_summary.csv').query('model in @final_names'))\n"
             "display(pd.read_csv(FINAL_RESULTS / 'failure_counts.csv').query('model in @final_names'))\n"
             "display(pd.read_csv(FINAL_RESULTS / 'group_results.csv').query('model in @final_names'))"),
        md("## Learning curves: validation data only\n\n"
           "The same nested 1,000 / 3,000 / 5,000 / 7,000 training subsets are used across models, with fixed selected settings. "
           "Every point uses the same 500 stratified validation plants. MLP stopping uses the full validation partition. "
           "These are development diagnostics, not independent holdout estimates; the final test set is never used for the curves."),
        code("final_figure = final_learning_plot(final_names)\nplt.show()\nplt.close(final_figure)\n"
             "display(pd.read_csv(FINAL_RESULTS / 'learning_curves.csv').query('model in @final_names')"
             ".sort_values(['model','training_size'])[['model','training_size','success_pct','r2','median_J_ratio']])")]
    if mlp:
        cells += [md("## MLP training and early stopping\n\n"
            "Three hidden layers (128, 64, 32), SiLU, dropout 0.05, positive normalized outputs and standardized log loss. "
            "Training restores the best validation-loss epoch; these curves do not alter the frozen model."),
            code("final_figure = final_history_plot()\nplt.show()\nplt.close(final_figure)")]
    cells += [md("## Shared response examples\n\n"
        "These examples are selected transparently from random-forest outcomes (median accepted J ratio and median stable overshoot miss). "
        "They illustrate behavior rather than estimate overall accuracy. Plotting stored gains does not rerun model selection or full test validation."),
        code("final_examples = example_rows(final_predictions)\ndisplay(pd.Series(final_examples, name='Final-cohort CSV row'))\n"
             "for final_case in ['Representative RF success','Representative RF overshoot failure']:\n"
             "    final_figure = response_plot(final_predictions, final_examples[final_case])\n"
             "    plt.show()\n    plt.close(final_figure)"),
        md("## Verification and limits\n\n"
           "The numerical target controllers were revalidated, all saved models reproduce their scored gains, and locked model/settings/data hashes remain unchanged. "
           "The held-out scores concern one split and seed within the accepted-label SOPDT population. Ideal PID and second-order Padé delay are approximations; "
           "there is no exact-delay, actuator/noise or hardware safety guarantee. Earlier 20K results use a different test cohort and are not a controlled sample-size comparison.\n\n"
           "Full report: `results/dataset_final_10000/report.md`. Reproduce training with `python scripts/train_final_10000.py` "
           "in a separate unscored workspace; the runner deliberately refuses to overwrite this locked experiment."),
        code("display(pd.read_csv(FINAL_RESULTS / 'acceptance_intervals.csv').query('model in @final_names'))\n"
             "display(json.loads((FINAL_RESULTS / 'verification.json').read_text(encoding='utf-8')))")]
    for cell in cells:
        cell.metadata[TAG] = True
    return cells


def append_execute():
    outputs = {}
    for filename, names, mlp in [
        ("baseline_validation.ipynb", ["linear","ridge","polynomial_ridge","random_forest","xgboost"], False),
        ("mlp_validation.ipynb", ["random_forest","xgboost","standard_mlp"], True)]:
        path = ROOT / "notebooks" / filename
        notebook = nbformat.read(path, as_version=4)
        old = [c for c in notebook.cells if not c.metadata.get(TAG)]
        old_digest = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        added = nbformat.v4.new_notebook(cells=section(names, mlp))
        print("Executing only new final-cohort cells:", filename, flush=True)
        NotebookClient(added, timeout=240, kernel_name="ml-pid-review",
                       resources={"metadata":{"path":str(ROOT)}}).execute()
        assert not any(o.output_type == "error" for c in added.cells if c.cell_type == "code" for o in c.outputs)
        notebook.cells = old + added.cells
        nbformat.validate(notebook)
        assert hashlib.sha256(json.dumps(notebook.cells[:len(old)], sort_keys=True, ensure_ascii=True).encode()).hexdigest() == old_digest
        nbformat.write(notebook, path)
        outputs[filename] = {"preserved_cells":len(old), "added_cells":len(added.cells), "new_cells_executed_without_errors":True}
        print("Saved", filename, flush=True)
    (ROOT / "results/dataset_final_10000/notebook_verification.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    append_execute()

# Model validation notebook

Open `baseline_validation.ipynb` in VS Code or Jupyter. Saved outputs already show
nine figures. To rerun, choose the Python environment `.venv-review/Scripts/python.exe`
and Run All. This environment contains the plotting dependencies used to verify
this notebook. For another environment, install the project scientific dependencies
plus matplotlib and ipykernel.

The notebook uses saved results in `results/regularized_1000`; it does not retrain
models. It shows performance, predicted-vs-target gains, validation outcomes,
regularized objective ratios, and three shared-plant response comparisons. It
also revalidates 15 illustrated model/plant predictions. All 15 matched saved
outcomes in the verified run. Full 1,000-prediction revalidation is optional.

For plot windows from a terminal at the project root:

```powershell
python scripts/plot_baselines.py
```

Use `--no-show` to save only, or `--row 2` for a custom held-out plant. PNG outputs
are in `results/regularized_1000/plots`. A custom row is a zero-based CSV row ID
that must belong to the saved test split. Objective plots include accepted pairs
only, with coverage shown; failures remain in overall pass-rate denominators.

The overshoot section includes threshold sensitivity, full measured distributions, bins around 10%, and the worst random-forest cases. Unstable and unmeasured cases are accounted for separately. Excess is measured in percentage points above the 10% limit.

## MLP comparison

`mlp_validation.ipynb` reads the standard and physics-informed MLP experiment, including training curves, seed sensitivity, decoded gain predictions, overshoot distributions, cost comparisons, and shared response examples. It does not train models or score the excluded 200 test plants. Use the same project kernel, with PyTorch installed.

## 20,000-row results appended

Both notebooks now preserve their original sections and end with **New experiment: 20,000 plants**. This uses the new exact 15,000/2,500/2,500 split and models in `results/dataset_20000`, including explicit ITAE comparisons against optimized controllers. Saved outputs are included; restart the project kernel and Run All to regenerate them.

## Rejected-generation stress set

`rejected_plants_stress_test.ipynb` is a separate experiment on 142 recovered rejected plants. It shows original-budget versus larger-sampling-budget outcomes, response examples, and repeat validation checks. No main model or accepted dataset was changed.

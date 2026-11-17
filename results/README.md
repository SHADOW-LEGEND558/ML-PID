# Which results are final?

**`dataset_final_10000/` is authoritative.** Its split, selected settings and main scores are frozen. `selection_lock.json` records historical checkpoint/protocol hashes. Original Python checkpoints are not distributed; the six `model_testing/` notebooks reconstruct training and controller evaluation from the CSV without saving external models or result files. Website model assets remain separately included.

- Main `report.md`, `test_metrics.csv` and per-model prediction CSVs: held-out comparison.
- `website_extension/`: common-validation curves through 20K training plants and 83 rejected-generation plants.
- `tuning_speed/`: warm timing on 20 validation plants.
- `stress_tests/`: five post-test robustness experiments and independent true-delay cross-checks.
- `final_checks/`: converged metrics above the overshoot threshold and tolerance sensitivity.

Older result folders and their checkpoints have been removed without archiving. The small `dataset_final_10000/development_reference/` collection retains only the 20K comparison table and evidence supporting the historical physics-loss and gain-penalty decisions. Its provenance manifest records the original files and hashes. These different populations and numerical budgets are not mixed into final accuracy.

The current scientific reproduction commands write new outputs under ignored `temp/`. Keep these published records for comparison. The original Python checkpoint copies and obsolete website, cleanup and environment audit records have been removed. Scientific protocols, data checks, predictions and experiment records are retained.

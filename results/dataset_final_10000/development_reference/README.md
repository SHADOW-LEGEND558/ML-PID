# Small development references still used by the final study

The older experiment directories and checkpoints were deleted during cleanup at the project owner's request; no historical archive was created.

- `20000_test_metrics.csv`: frozen six-model development scores used by the final report's cross-cohort comparison. The populations and training sizes differ; this is not a controlled final-test comparison.
- `physics_comparison/`: seed-level evidence, settings and original report supporting the website's historical standard-versus-physics-loss comparison. Fold outputs, histories, plots and weights were removed.
- `gain_penalty_calibration/`: matched 200-plant calibration inputs, trial outcomes and summaries supporting the choice of lambda 0.01. Redundant derived datasets and partial trial files were removed.

`provenance.json` records original paths and SHA-256 hashes. Reports are original historical context and can mention outputs that are no longer distributed. These small references support stated methodological choices; they are not part of the final 1,500-plant test score. Core final reproduction does not load old checkpoints.

To repeat the selected standard/physics-loss comparison over the three seeds and recorded folds, run `python scripts/compare_physics_loss.py --jobs 4 --output temp/physics_comparison` from the repository root. The original physics report also describes its broader candidate search and historical checkpoint creation; the current runner repeats the selected comparison and writes new results separately.

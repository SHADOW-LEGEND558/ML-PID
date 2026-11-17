# Scientific reproduction commands

Run from the repository root after installing `requirements-notebooks.txt`. Model-dependent experiments reconstruct the published selected models from the fixed training/validation split in memory; no original checkpoint is required. Fresh experiment outputs default to ignored `temp/` paths, leaving published evidence separate.

| Script | Experiment |
|---|---|
| `prepare_final_dataset.py` | Dataset/feature/label audit and fixed 70/15/15 split |
| `train_final_10000.py` | Validation-based model selection, held-out evaluation and learning curves in a new output directory |
| `report_final_10000.py` | Main tables and scientific figures from saved evidence |
| `run_stress_suite.py` | Noise, uncertainty, boundaries, true delay and disturbance tests |
| `check_true_delay_stress.py` | Independent RK4/history cross-check of the delay solver |
| `report_stress_suite.py` | Stress summaries, plant-cluster intervals and scientific plots |
| `final_controller_diagnostics.py` | Complete metrics for stable overshoot misses and tolerance sensitivity |
| `extend_final_study.py` | Curves through 20K training plants and the final rejected-plant experiment |
| `benchmark_final_speed.py` | Warm model inference/verification versus numerical tuning |
| `compare_gain_penalty.py` | Matched regularization-weight calibration |
| `compare_physics_loss.py` | Historical selected standard/physics-loss comparison over three seeds |

Exact commands and runtime notes are in the root README. `src/study.py` holds shared specifications/fitting/evaluation, and `src/plotting.py` holds shared graphics. Website-only exporters/checks and one-time migration helpers are intentionally absent.

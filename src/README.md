# Reusable scientific implementation

- `data_gen.py`: stable SOPDT plants, analytical observation, five measured features, ideal-PID/Padé simulation, feasible multistart SLSQP and independent validation.
- `MLP.py`: positive normalized-output MLP and the separately studied optional physics loss. The final model uses physics-loss weight zero.
- `benchmark.py`: data loading, gain errors, grouped splitting and raw-controller validation.
- `target_normalization.py`: empirical target scaling and multiprocessing controller tasks.
- `study.py`: selected configurations, 70/15/15 split validation, model reconstruction and experiment summaries. `selected_models()` fits in memory without writing checkpoints/results.
- `data_validation.py`: independent label and feature audit.
- `rejected_plants.py`: rejection-log recovery and standard/expanded numerical-budget validation.
- `plotting.py`: shared comparison/response/overshoot figures.

The six model notebooks show training directly; scientific runners reuse these modules. Physical parameters are evaluator metadata, not model features. No prediction clipping or hidden substitute controller is used.

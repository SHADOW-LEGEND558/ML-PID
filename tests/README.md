# Scientific tests

Run `python -m unittest discover -s tests -v` from the repository root. Tests cover PID stability/optimization, exact-zero and positive delay, convergence, replacement sampling, feature schema, target normalization, MLP loss/checkpoint round trips in temporary directories, true-delay/disturbance numerics, stable-controller diagnostics, fixed splits and the training-notebook output contract.

Website/export checks are not part of this suite. Full held-out evaluation is in the model notebooks; the five robustness experiments are separate, more expensive script runs. Repository scratch work belongs in ignored `temp/`.

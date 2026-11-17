# Static research website

Serve with `python -m http.server 8000 --directory docs`, then open http://localhost:8000/. No build step, exporter, Node installation or Python backend is needed. The site includes its published evidence, generated Python snapshot and portable browser model assets. Keep these shipped assets together.

- `index.html`: the six project sections and scientific explanations.
- `styles.css`: the consolidated stylesheet, including both themes.
- `app.js`: navigation, simulators, data-generation walkthrough, evidence plots and prediction UI.
- `simulation-workers.js`: lazy background Pyodide tuning/verification.
- `theme.js`: applies the saved theme before paint.
- `research-data.js`, `supplement-data.js`, `generator-source.js`: included evidence/simulation snapshots.
- `model-data.js`, `model-inference.js`, `baseline-manifest.js`, `baseline-inference.js`, `models/`: included browser prediction assets.

The optional live demos load pinned Pyodide/NumPy/SciPy from jsDelivr on first use. Inputs and computation stay local to the browser. Saved evidence remains readable if the live runtime cannot load. MLP is preferred; Random Forest and XGBoost are comparison baselines.

Scientific reproduction lives in `model_testing/`, `src/`, `scripts/` and `tests/`. The root README gives commands for retraining, full controller evaluation, stress tests, diagnostics, learning curves, rejected plants and timing. Website-only exporters/check scripts are intentionally removed. Publishing remains owner-controlled.

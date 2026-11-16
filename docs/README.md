# ML-PID research website

A buildless, static website with six sections: the study, system basics, data
generation, learning to tune, results and limitations. Serve it over HTTP from
the repository root:

```powershell
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/. File URLs cannot run the tuning worker reliably.
No npm packages or Python backend are required. The optional Google Fonts
stylesheet has local font fallbacks. System Basics works without a Python runtime.
The Data Generation walkthrough loads pinned Pyodide 314.0.7, NumPy, and SciPy
from jsDelivr on its first run, so tuning requires an internet connection.
Computation happens in a module worker; plant parameters are not sent to a server.

- `index.html`: system and PID explanations, two explorers, and generation pipeline.
- `styles.css`: responsive visual theme.
- `app.js`: tab navigation, analytical open-loop response, and numerical PID simulation.
- `generation.js` / `generation.css`: sample controls and the four-stage walkthrough.
- `generation-worker.js`: runs actual Python tuning and validation off the UI thread.
- `generator-source.js`: generated snapshot of the relevant classes in `src/data_gen.py`.

The System Basics explorers are browser-side illustrations, not the dataset
generator or automatic PID tuner. They do not load the CSV or present illustrative curves as measured
results. Open-loop output is analytical with true delay. Closed-loop output uses
ideal PID and a second-order Pade delay, matching `src/data_gen.py`; JavaScript
constructs the same transfer-function polynomials and integrates canonical state
space using RK4. Zero initial conditions and a unit setpoint are assumed. The
ideal derivative produces an input impulse of weight Kd at time zero; no finite
input spike is substituted. Displayed sampled metrics do not certify stability.

Data Generation uses the actual `PlantSimulator`, `FeatureExtractor`, and
`PIDOptimizer` classes exported from the generator, including feasible-start screening and the up-to-four-start
SLSQP search, refined validation, bounds, and resource limits. It exports a row
only when a candidate is accepted. Browser controls round parameter choices to
0.01; batch Python sampling is continuous for positive delays and the other parameters. Delay zero can be selected manually. Batch generation draws exact zero delay with configurable probability (default 10%); the existing dataset retains its original delays.
Open-loop features use measured-tail convergence from 10 to 640 seconds at 0.01-second intervals. The CSV includes the observation duration as metadata. AUC is excluded; only five empirical features are used.
Charts are downsampled for display; validated metrics use the full Python grids.
Library versions are shown because results can differ slightly between runtimes.

After changing `src/data_gen.py`, refresh the checked-in browser snapshot:

```powershell
python scripts/export_web_generator.py
```

This standard-library-only command runs from the repository root. A regression
test checks the snapshot hash to catch stale exports. Then refresh the browser.

All asset links are relative, so the directory can be hosted under a GitHub Pages
repository path. Publishing and Git configuration have intentionally not been set
up for this local draft.

Tuning uses the dimensionless squared-gain penalty with weight 0.01. Selection
uses the combined objective, while displayed ITAE remains the raw tracking metric.
This is not an actuator-energy model.

The website reports the frozen final 10,000-row study (7,000 / 1,500 / 1,500).
Extended learning curves use the same final-validation subset and independent
development training plants; the final models and test scores are unchanged. Results and saved response examples are native SVG graphics,
with model selection, overshoot-threshold exploration and plant-group breakdowns.
Light mode is the default; the header switch saves a dark-mode preference locally.

- `research.css` / `research.js`: study layout and result graphics.
- `research-data.js`: generated, compact saved evidence; no live model fitting.
- `theme.css` / `theme.js`: persistent light/dark palette.

Refresh saved website evidence from the repository root with:

```powershell
python scripts/export_website_results.py
```

The exporter reads the frozen final experiment artifacts and uses the shared
Python simulator only to produce downsampled display curves. It never trains,
selects or retunes a model. Its Python environment needs the project's NumPy,
pandas and SciPy dependencies. Reported performance remains the original full
validation output, not metrics recalculated from display curves.


The live prediction explorer uses the frozen final MLP exported to native JavaScript.
Five empirical inputs predict three gains without a Python runtime. Physical-input
mode uses the shared generator to measure those same five features. Optional known
plants permit independent response validation and a fresh numerical comparison in
a lazy Pyodide worker. Inputs remain local. Feature-only mode does not simulate
an unidentified physical plant.

Refresh model assets and check export parity:

```powershell
python scripts/export_browser_model.py
node scripts/check_browser_model.cjs
python scripts/export_website_results.py
```

`model-data.js` and `model-inference.js` contain the frozen model and forward pass;
`prediction-worker.js` and `prediction.js` handle physical verification and the UI.
The browser assessor is exported verbatim from `src/benchmark.py`; after changing
the simulator/evaluator, refresh both generator and model exports.

Additional reproducible evidence: `scripts/extend_final_study.py` and
`scripts/benchmark_final_speed.py`. Their artifacts live in
`results/dataset_final_10000/website_extension` and `tuning_speed`. The stress set
contains 83 rejected plants; standard and expanded grid-budget results are separate.

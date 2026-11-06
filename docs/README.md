# ML-PID website draft

A buildless, static website with System Basics and Data Generation tabs. Open
`index.html` in a browser, or serve this directory with any static HTTP server.
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

The explorers are browser-side illustrations, not a dataset generator or automatic
PID tuner. They do not load the CSV or present illustrative curves as measured
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
0.01; batch Python sampling is continuous. Delay zero can be selected manually.
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

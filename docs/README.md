# ML-PID website draft

A buildless, static website with System Basics and Data Generation tabs. Open
`index.html` in a browser, or serve this directory with any static HTTP server.
No npm packages, Python backend, or build step are required. The optional Google
Fonts stylesheet has local font fallbacks.

- `index.html`: system and PID explanations, two explorers, and generation pipeline.
- `styles.css`: responsive visual theme.
- `app.js`: tab navigation, analytical open-loop response, and numerical PID simulation.

The explorers are browser-side illustrations, not a dataset generator or automatic
PID tuner. They do not load the CSV or present illustrative curves as measured
results. The time axis automatically adjusts to plant speed. PID feedback uses
a 0.05-second derivative filter, a controller sample interval no greater than
0.001 seconds, RK4 plant integration with held input, and interpolated input
history for delay. Zero initial conditions and a unit setpoint are assumed.
The plots report finite-window sampled metrics, not stability guarantees.

All asset links are relative, so the directory can be hosted under a GitHub Pages
repository path. Publishing and Git configuration have intentionally not been set
up for this local draft.

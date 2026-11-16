"""Export frozen final-study evidence for the static site; never fit or select models.

Run from the repository root. Display curves use the shared plant simulator, while
reported metrics remain those from the original full validation artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_gen import Plant, PIDGains, PlantSimulator

RESULTS = ROOT / "results/dataset_final_10000"
EXTENDED = RESULTS / "website_extension"
NAMES = {"linear": "Linear regression", "ridge": "Ridge", "polynomial_ridge": "Polynomial ridge",
         "random_forest": "Random forest", "xgboost": "XGBoost", "standard_mlp": "MLP"}
metrics = pd.read_csv(RESULTS / "test_metrics.csv")
learning = pd.read_csv((EXTENDED if (EXTENDED / "verification.json").exists() else RESULTS) / "learning_curves.csv")
overshoot = pd.read_csv(RESULTS / "overshoot_summary.csv")
timing_path = RESULTS / "tuning_speed/summary.csv"
if not timing_path.exists(): timing_path = ROOT / "results/dataset_20000/tuning_speed/summary.csv"
timing = pd.read_csv(timing_path)
predictions = {key: pd.read_csv(RESULTS / f"{key}_test_predictions.csv") for key in NAMES}
shared_rows = predictions["standard_mlp"].sample(180, random_state=68).source_row.tolist()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 8) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

data = {"scope": "Final 10K study; 7000 training, 1500 validation, 1500 held-out test",
        "metadata": json.loads((ROOT / "data/pid_dataset_final_10000.meta.json").read_text()),
        "sources": ["results/dataset_final_10000/test_metrics.csv", "results/dataset_final_10000/*_test_predictions.csv",
                    "results/dataset_final_10000/website_extension/learning_curves.csv", "results/dataset_final_10000/overshoot_summary.csv",
                    "results/dataset_final_10000/tuning_speed/summary.csv"],
        "models": [], "cases": [], "finalReady": (ROOT / "data/pid_dataset_final_10000.meta.json").exists(),
        "finalEvaluated": (ROOT / "results/dataset_final_10000/verification.json").exists()}
for key, label in NAMES.items():
    frame = predictions[key]
    accepted = frame[frame.accepted & frame.reference_accepted]
    # Stable but unmeasured outcomes are omitted from the CDF, never replaced by zero.
    measured = frame.loc[frame.stable & np.isfinite(frame.overshoot_pct), "overshoot_pct"]
    sample = frame.set_index("source_row").loc[shared_rows]
    points = [{"row": row, "pass": bool(r.accepted),
               "actual": [r[f"target_{g}"] for g in ("Kp", "Ki", "Kd")],
               "predicted": [r[f"predicted_{g}"] for g in ("Kp", "Ki", "Kd")]}
              for row, r in sample.iterrows()]
    ratios = accepted.itae_ratio.dropna().to_numpy()
    quantiles = np.linspace(0, 1, 201)
    groups = {}
    for field, bins in {
        "delay": [("τ = 0", frame.tau == 0), ("0 < τ ≤ 0.1 s", (frame.tau > 0) & (frame.tau <= .1)),
                  ("0.1 < τ ≤ 0.5 s", (frame.tau > .1) & (frame.tau <= .5)), ("τ > 0.5 s", frame.tau > .5)],
        "damping": [("ζ < 0.5", frame.zeta < .5), ("0.5 ≤ ζ < 1", (frame.zeta >= .5) & (frame.zeta < 1)),
                    ("ζ ≥ 1", frame.zeta >= 1)]}.items():
        groups[field] = [{"name": name, "n": int(mask.sum()), "passes": int(frame.loc[mask, "accepted"].sum()),
                          "pct": float(100 * frame.loc[mask, "accepted"].mean())} for name, mask in bins]
    data["models"].append({"id": key, "name": label, "groups": groups,
        "metrics": metrics.loc[metrics.model == key].iloc[0].to_dict(),
        "overshoot": overshoot.loc[overshoot.model == key].iloc[0].to_dict(),
        "timing": timing.loc[timing.model == key].iloc[0].to_dict(),
        "reasons": {k: int(v) for k, v in frame.reason.value_counts().items()},
        "measuredOvershoot": sorted(measured.tolist()), "scatter": points,
        "itaeCdf": [[float(np.quantile(ratios, p)), float(100 * p)] for p in quantiles],
        "itaeP90": float(np.quantile(ratios, .9)), "itaeMax": float(np.max(ratios)),
        "itaeOver2": int(np.sum(ratios > 2)),
        "learning": learning.loc[learning.model == key].sort_values("training_size")[["training_size", "success_pct"]].values.tolist()})

mlp = predictions["standard_mlp"].set_index("source_row")
rf = predictions["random_forest"].set_index("source_row")
def middle_row(frame, col):
    return int(frame.sort_values(col).index[len(frame) // 2])

selected = [
    ("Typical accepted controller", middle_row(mlp[mlp.accepted & (mlp.K > 1.2)], "objective_ratio")),
    ("Forest miss, MLP pass", middle_row(mlp[~rf.accepted & mlp.accepted], "overshoot_pct")),
    ("An MLP overshoot miss", middle_row(mlp[~mlp.accepted & (mlp.reason == "overshoot")], "overshoot_pct")),
    ("Exactly zero dead time", middle_row(mlp[mlp.tau == 0], "objective_ratio")),
    ("Lightly damped plant", middle_row(mlp[(mlp.zeta < .2) & mlp.accepted], "objective_ratio")),
    ("Accepted, but high ITAE", int(mlp[mlp.accepted].itae_ratio.idxmax())),
]
sim = PlantSimulator()
for label, rowid in selected:
    row = mlp.loc[rowid]
    plant = Plant(zeta=row.zeta, wn=row.wn, tau=row.tau, K=row.K)
    duration = min(100., max(3., 1.65 * row.t_settling))
    if label == "Accepted, but high ITAE":
        # This deliberately selected long-tail case needs its actual slow mode.
        gains = PIDGains(*[row[f"predicted_{g}"] for g in ("Kp", "Ki", "Kd")])
        poles = np.roots(sim.closed_loop_coefficients(plant, gains)[1])
        duration = min(100., max(duration, 6 / abs(poles.real.max())))
    t = np.linspace(0., duration, 601)
    target = PIDGains(*[row[f"target_{g}"] for g in ("Kp", "Ki", "Kd")])
    _, y_target = sim.close_loop(plant, target, t)
    _, y_open = sim.open_loop(plant, t)
    item = {"name": label, "sourceRow": rowid, "plant": {k: row[k] for k in ("K", "wn", "zeta", "tau")},
            "features": {k:row[k] for k in ["t_rise","t_settling","y_peak","y_ss","tau_est"]}, "t": t.tolist(), "open": y_open.tolist(), "reference": y_target.tolist(), "models": {}}
    for key, frame in predictions.items():
        r = frame.set_index("source_row").loc[rowid]
        gains = [r[f"predicted_{g}"] for g in ("Kp", "Ki", "Kd")]
        y = None
        if r.stable and np.all(np.isfinite(gains)):
            _, result = sim.close_loop(plant, PIDGains(*gains), t)
            if np.isfinite(result).all() and np.max(np.abs(result)) < 20:
                y = result.tolist()
        item["models"][key] = {"y": y, "gains": gains,
             "accepted": bool(r.accepted), "reason": r.reason,
             "itae": r.itae, "referenceItae": r.reference_itae,
             "overshoot": r.overshoot_pct, "itaeRatio": r.itae_ratio if r.accepted else None,
             "objectiveRatio": r.objective_ratio if r.accepted else None}
    data["cases"].append(item)

data["testCount"] = 1500
data["timingScope"] = "Final validation cohort" if "dataset_final_10000" in str(timing_path) else "20K development timing (separate experiment)"
optimizer_path = timing_path.parent / "optimizer.csv"
data["optimizerMedianMs"] = float(pd.read_csv(optimizer_path).optimizer_seconds.median()*1000)
data["learningExtended"] = (EXTENDED / "verification.json").exists()
if (EXTENDED / "rejected_summary.csv").exists():
    data["rejected"] = pd.read_csv(EXTENDED / "rejected_summary.csv").to_dict("records")
    data["rejectedProvenance"] = json.loads((EXTENDED / "verification.json").read_text(encoding="utf-8"))
destination = ROOT / "docs/research-data.js"
destination.write_text("// Generated from saved experiment artifacts; see scripts/export_website_results.py.\n"
                       + "window.PID_STUDY = " + json.dumps(clean(data), separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
print(f"Exported {len(data['models'])} models, {len(data['cases'])} cases, {destination.stat().st_size:,} bytes.")

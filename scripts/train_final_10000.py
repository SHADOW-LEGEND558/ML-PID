"""Evaluate the fresh 10K cohort using its already-frozen 7000/1500/1500 split.

Reuses the established estimators and controller checks. Test scores are written
only after candidate selection and model-file hashes have been locked.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import study as shared
from benchmark import FEATURES, TARGETS, load_frame
from study import summarize
from provenance import matches_recorded_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dataset_final_10000"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(jobs=12, output=None):
    global OUT
    if output is not None:
        OUT = Path(output).resolve()
    started = perf_counter()
    protocol_path = OUT / "protocol.json"
    split = json.loads(protocol_path.read_text(encoding="utf-8"))
    data = ROOT / split["data_path"]
    if (OUT / "selection_lock.json").exists() or (OUT / "test_metrics.csv").exists():
        raise RuntimeError("Final study already locked/scored. Refusing to overwrite the holdout experiment.")
    frame = load_frame(data)
    metadata = json.loads(data.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert len(frame) == 10000
    assert matches_recorded_hash(data, split["dataset_sha256"])
    assert matches_recorded_hash(data, metadata["dataset_sha256"])
    assert metadata["gain_penalty"] == .01 and not frame.duplicated(["zeta", "wn", "tau", "K"]).any()
    train, validation, test = (split[k] for k in ("train", "validation", "test"))
    assert [len(train), len(validation), len(test)] == [7000, 1500, 1500]
    assert len(set(train) | set(validation) | set(test)) == 10000
    assert not set(train) & set(validation) and not set(train) & set(test) and not set(validation) & set(test)
    curve_validation = list(map(int, train_test_split(validation, train_size=500, random_state=44,
                                       stratify=frame.loc[validation, "tau"].eq(0))[0]))
    protocol = {**split, "curve_validation": curve_validation, "learning_sizes": [1000, 3000, 5000, 7000],
        "test_policy": "Score once after all selected models/settings are locked; no post-test retuning.",
        "status": "Training and validation; test not scored yet.",
        "target_normalization": "q=[y_ss*Kp, y_ss*t_rise*Ki, y_ss*Kd/t_rise]",
        "learning_policy": "Nested training subsets; fixed 500 validation plants; selected settings fixed.",
        "mlp_settings": {"architecture": [5,128,64,32,3], "batch_size": 256, "max_epochs": 500,
                         "patience": 50, "dropout": .05, "weight_decay": .0001, "beta": 0, "seed": 42},
        "versions": {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "xgboost", "torch"]}}
    # Keep the original preparation manifest unchanged. Record this run separately.
    (OUT / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    shared.OUT = OUT  # Shared fitting helpers save histories in this cohort only.
    models, templates, candidates, selected, curves = {}, {}, [], [], []
    print("Verified frozen 7000/1500/1500 split; five empirical inputs, normalized targets.", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        val_refs = shared.reference_rows(validation, pool, frame, .01)
        pd.DataFrame(val_refs.values()).to_csv(OUT / "validation_references.csv", index=False)
        print("Validation numerical-reference controllers rechecked.", flush=True)
        for family, settings in shared.specs().items():
            choices = []
            for setting, estimator in settings:
                label = f"{family}_{setting}"
                print("Fitting", label, flush=True)
                fit_start = perf_counter()
                model = shared.fit_model(family, estimator, train, validation, frame, label)
                fit_seconds = perf_counter() - fit_start
                result, latency = shared.evaluate(model, validation, pool, frame, val_refs, .01, label)
                result.to_csv(OUT / f"{label}_validation.csv", index=False)
                record = {"model": family, "setting": setting, "candidate": label,
                          "fit_seconds": fit_seconds, "batch_prediction_ms_per_plant": latency, **summarize(result)}
                candidates.append(record)
                choices.append((record, model, estimator))
                pd.DataFrame(candidates).to_csv(OUT / "validation_candidates.csv", index=False)
                print(label, "validation acceptance", record["success_pct"], flush=True)
            winner, model, template = min(choices, key=lambda item: shared.key(item[0]))
            models[family], templates[family] = model, template
            selected.append(winner)
            if family == "standard_mlp":
                model.save(OUT / f"{family}.pt", metadata={"protocol": protocol, "selected": winner})
            else:
                joblib.dump({"pipeline": model, "features": FEATURES, "targets": TARGETS,
                             "protocol": protocol, "selected": winner}, OUT / f"{family}.joblib")
            chosen = pd.read_csv(OUT / f"{winner['candidate']}_validation.csv")
            chosen.to_csv(OUT / f"{family}_validation_predictions.csv", index=False)
            curves.append({"model": family, "training_size": 7000,
                           **summarize(chosen[chosen.source_row.isin(curve_validation)])})

        pd.DataFrame(selected).to_csv(OUT / "selected_before_test.csv", index=False)
        artifact_paths = [OUT / f"{name}.{'pt' if name == 'standard_mlp' else 'joblib'}" for name in models]
        artifact_paths += [OUT / "selected_before_test.csv", OUT / "evaluation_protocol.json", protocol_path]
        locked = {p.name: sha(p) for p in artifact_paths}
        (OUT / "selection_lock.json").write_text(json.dumps({"dataset_sha256": sha(data),
            "phase": "Saved before any test reference or prediction scoring", "files": locked}, indent=2), encoding="utf-8")
        print("All six models and settings locked. Beginning one-time held-out test scoring.", flush=True)
        test_refs = shared.reference_rows(test, pool, frame, .01)
        pd.DataFrame(test_refs.values()).to_csv(OUT / "test_references.csv", index=False)
        metrics, reproducibility = [], {}
        for family, model in models.items():
            result, latency = shared.evaluate(model, test, pool, frame, test_refs, .01, f"{family} TEST")
            result.to_csv(OUT / f"{family}_test_predictions.csv", index=False)
            metrics.append({"model": family, "batch_prediction_ms_per_plant": latency, **summarize(result)})
            pd.DataFrame(metrics).to_csv(OUT / "test_metrics.csv", index=False)
            # Check serialization reproduces the already-scored gains, without
            # additional simulation, setting changes or clipping predictions.
            if family == "standard_mlp":
                from MLP import MLPModel
                restored = MLPModel.load(OUT / f"{family}.pt")
            else:
                restored = joblib.load(OUT / f"{family}.joblib")["pipeline"]
            difference = float(np.max(np.abs(restored.predict(frame.loc[test, FEATURES]) -
                result[[f"predicted_{g}" for g in TARGETS]].to_numpy())))
            assert difference < 1e-6
            reproducibility[family] = difference
            print(family, "test results saved; settings unchanged.", flush=True)

        # Development learning curves never evaluate the final holdout again.
        nested = np.random.default_rng(45).permutation(train)
        for family in models:
            for size in [1000, 3000, 5000]:
                print("Learning curve", family, size, flush=True)
                model = shared.fit_model(family, clone(templates[family]) if family != "standard_mlp" else None,
                    nested[:size], validation, frame, f"{family}_learning{size}")
                result, _ = shared.evaluate(model, curve_validation, pool, frame, val_refs, .01,
                                            f"{family} learning {size}")
                result.to_csv(OUT / f"{family}_learning{size}_validation.csv", index=False)
                curves.append({"model": family, "training_size": size, **summarize(result)})
                pd.DataFrame(curves).to_csv(OUT / "learning_curves.csv", index=False)
    assert all(sha(OUT / filename) == digest for filename, digest in locked.items())
    verification = {"split_sizes": {k: len(split[k]) for k in ["train", "validation", "test"]},
        "dataset_hash_unchanged": matches_recorded_hash(data, protocol["dataset_sha256"]),
        "all_locked_artifact_hashes_unchanged": True, "saved_prediction_max_abs_difference": reproducibility,
        "validation_reference_accepted": sum(r["accepted"] for r in val_refs.values()),
        "test_reference_accepted": sum(r["accepted"] for r in test_refs.values()),
        "test_scored_once_per_model": True, "seconds": perf_counter() - started}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("Final 10,000-row study complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Prepared result directory; use a new directory for reproduction.")
    args = parser.parse_args()
    main(args.jobs, args.output)

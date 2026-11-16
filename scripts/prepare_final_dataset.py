"""Check a fresh 10K dataset and freeze its 70/15/15 split, without training.

The spot audit reuses the independent label/feature auditor. This does not score
any learned model or select settings using the final test partition.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_gen import Plant, PIDGains, PlantSimulator
from audit_dataset import audit


def prepare(path, destination, jobs=4):
    frame = pd.read_csv(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(frame) == metadata["samples"] == 10000
    assert metadata["gain_penalty"] == .01
    assert digest == metadata["dataset_sha256"]
    assert np.isfinite(frame.to_numpy()).all()
    assert "AUC" not in frame
    plant_columns = ["zeta", "wn", "tau", "K"]
    assert not frame.duplicated(plant_columns).any()
    for key, low, high in [("K", .5, 2), ("wn", .5, 10), ("zeta", .1, 2), ("tau", 0, 1),
                            ("Kp", .001, 100), ("Ki", .001, 100), ("Kd", 0, 50)]:
        assert frame[key].between(low - 1e-8, high + 1e-8).all(), key
    assert frame.t_rise.gt(0).all() and frame.y_ss.gt(0).all()

    old = pd.read_csv(ROOT / "data/pid_dataset_20000.csv", usecols=plant_columns)
    old_keys = set(map(tuple, old[plant_columns].round(12).to_numpy()))
    overlaps = sum(tuple(row) in old_keys for row in frame[plant_columns].round(12).to_numpy())
    assert overlaps == 0, "Fresh data overlaps the development plants."

    # Fixed exact sizes and delay strata; no model predictions are involved.
    indices = np.arange(len(frame))
    train, held = train_test_split(indices, train_size=7000, random_state=42, stratify=frame.tau.eq(0))
    validation, test = train_test_split(held, test_size=1500, random_state=43, stratify=frame.loc[held, "tau"].eq(0))
    split = {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()}
    assert len(set(train) | set(validation) | set(test)) == 10000
    assert not set(train) & set(validation) and not set(train) & set(test) and not set(validation) & set(test)
    protocol = {**split, "data_path": path.relative_to(ROOT).as_posix(), "dataset_sha256": digest,
                "seed": 42, "gain_penalty": .01,
                "features": ["t_rise", "t_settling", "y_peak", "y_ss", "tau_est"],
                "targets": ["Kp", "Ki", "Kd"],
                "selection": "validation full-pass count, then median accepted J ratio",
                "test_policy": "Untouched by models; lock settings before scoring once.",
                "status": "Dataset and split prepared; no final models trained or evaluated.",
                "strata_counts": {key: {"zero": int(frame.loc[value, "tau"].eq(0).sum()),
                                       "positive": int(frame.loc[value, "tau"].gt(0).sum())} for key, value in split.items()}}

    # A fresh sample of each delay stratum checks labels and acquisition logic.
    sampled = np.concatenate([frame[frame.tau.eq(0)].sample(16, random_state=48).index,
                              frame[frame.tau.gt(0)].sample(16, random_state=49).index])
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        checks = list(pool.map(audit, [(int(i), frame.loc[i].to_dict(), .01) for i in sampled]))
    checked = pd.DataFrame(checks)
    assert checked.validation.eq("accepted").all()
    assert checked.max_feature_difference.max() < 1e-8
    assert checked.duration_difference.max() < 1e-8
    assert checked.dense_overshoot_pct.le(10 + 1e-6).all()
    assert checked.dense_itae_relative_change.le(.001).all()

    # An inexpensive pole check covers every stored label, beyond the spot audit.
    simulator = PlantSimulator()
    max_real = max(float(np.roots(simulator.closed_loop_coefficients(
        Plant(**{k: r[k] for k in plant_columns}), PIDGains(r.Kp, r.Ki, r.Kd))[1]).real.max())
        for _, r in frame.iterrows())
    assert max_real < -1e-8
    upper_hits = np.isclose(frame[["Kp", "Ki", "Kd"]], [100, 100, 50], rtol=0, atol=1e-5).any(axis=1)
    summary = {"rows": len(frame), "attempts": metadata["attempts"], "dataset_sha256": digest,
               "seed": metadata["seed"], "zero_delay_rows": int(frame.tau.eq(0).sum()),
               "nonfinite": 0, "duplicate_plants": 0, "development_overlap_at_12_decimals": overlaps,
               "upper_bound_hits": int(upper_hits.sum()), "largest_pole_real_part": max_real,
               "spot_audit_rows": len(checked), "spot_audit_accepted": int(checked.validation.eq("accepted").sum()),
               "max_spot_feature_difference": float(checked.max_feature_difference.max()),
               "max_spot_dense_overshoot_pct": float(checked.dense_overshoot_pct.max()),
               "max_spot_dense_itae_relative_change": float(checked.dense_itae_relative_change.max()),
               "split_sizes": {key: len(value) for key, value in split.items()},
               "strata_counts": protocol["strata_counts"], "model_evaluation": "Not performed"}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (destination / "data_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    checked.to_csv(destination / "spot_audit.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pid_dataset_final_10000.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dataset_final_10000")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.output.resolve(), args.jobs)

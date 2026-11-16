"""Shared empirical-only training, held-out evaluation, and artifact export."""
import argparse
import hashlib
import json
from importlib.metadata import version
from time import perf_counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data_gen import Plant, PIDGains, PlantSimulator, PIDOptimizer

FEATURES = ['t_rise', 't_settling', 'y_peak', 'y_ss', 'tau_est']
TARGETS = ['Kp', 'Ki', 'Kd']
PLANT_COLUMNS = ['zeta', 'wn', 'tau', 'K']  # Evaluation metadata, never model inputs.


def load_frame(path):
    frame = pd.read_csv(path)
    columns = PLANT_COLUMNS + FEATURES + TARGETS
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('Dataset contains missing or nonfinite required values.')
    if (frame[['zeta', 'wn', 'K']] <= 0).any().any() or (frame.tau < 0).any():
        raise ValueError('Invalid plant parameters in evaluation metadata.')
    return frame


def split_plants(frame, test_size=0.2, seed=42):
    # Identical plants stay together even when their rows or labels differ.
    groups = frame.groupby(PLANT_COLUMNS, sort=True).ngroup().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(frame, groups=groups))
    if len(test) < 2 or len(np.unique(groups[train])) < 2:
        raise ValueError('Need at least two test rows and two distinct training plants.')
    return train, test, groups


def gain_metrics(targets, predictions):
    if predictions.shape != targets.shape or not np.isfinite(predictions).all():
        raise ValueError('Predictions must be finite and match the target shape.')
    result = {}
    for i, name in enumerate(TARGETS):
        result[f'{name}_mae'] = float(mean_absolute_error(targets[:, i], predictions[:, i]))
        result[f'{name}_rmse'] = float(np.sqrt(mean_squared_error(targets[:, i], predictions[:, i])))
        result[f'{name}_r2'] = float(r2_score(targets[:, i], predictions[:, i]))
    result['r2'] = float(r2_score(targets, predictions, multioutput='uniform_average'))
    return result


def assess_controller(plant, vector, gain_penalty=0.01):
    """Raw predictions: never clip gains or silently substitute a controller."""
    sim = PlantSimulator()
    optimizer = PIDOptimizer(sim, gain_penalty=gain_penalty)
    result = {'stable': False, 'within_bounds': False, 'accepted': False,
              'overshoot_pct': None, 'itae': None, 'objective': None,
              'normalized_itae': None, 'gain_cost': None, 'reason': 'nonfinite_gains'}
    if not np.isfinite(vector).all():
        return result
    result['within_bounds'] = bool(np.all(vector >= optimizer.BOUNDS[:, 0])
                                   and np.all(vector <= optimizer.BOUNDS[:, 1]))
    gains = PIDGains(*map(float, vector))
    try:
        _, den = sim.closed_loop_coefficients(plant, gains)
        poles = np.roots(den)
        result['stable'] = bool(np.isfinite(poles).all() and np.max(poles.real) < -1e-8)
        if not result['stable']:
            result['reason'] = 'unstable'
            return result
        grid = sim.get_dynamic_t_eval(plant)
        metrics, reason = optimizer._validate_candidate(plant, gains, grid)
        result['reason'] = reason
        if metrics is not None:
            result.update(accepted=True, overshoot_pct=metrics['overshoot_pct'],
                          itae=metrics['itae'])
            result.update(optimizer.objective_terms(plant, vector, metrics['itae']))
        else:
            # Diagnostics only: these finite-window numbers are NOT validated.
            duration = max(float(grid[-1]), 12 / -float(np.max(poles.real)))
            n = max(8001, int(np.ceil(duration * np.max(abs(poles.imag)) * 32 / np.pi)) + 1)
            if duration <= optimizer.max_duration and n <= optimizer.max_validation_points:
                observed = optimizer._response_metrics(plant, gains, duration, n)
                result['overshoot_pct'] = observed['overshoot_pct']
        return result
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        result['reason'] = 'simulation_failure'
        return result


def run_experiment(models, family):
    parser = argparse.ArgumentParser(description=f'Five-feature {family} PID benchmarks')
    parser.add_argument('--data', type=Path, default=Path('data/pid_dataset.csv'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('results') / family)
    args = parser.parse_args()
    if args.cv < 2:
        parser.error('--cv must be at least 2')
    frame = load_frame(args.data)
    train, test, groups = split_plants(frame, args.test_size, args.seed)
    X, y = frame[FEATURES], frame[TARGETS].to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    metadata_path = args.data.with_suffix('.meta.json')
    if not metadata_path.exists():
        parser.error('Dataset metadata is required to identify the tuning objective and weight.')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['dataset_sha256'] != digest:
        parser.error('Dataset hash does not match its metadata.')
    weight = float(metadata['gain_penalty'])
    manifest = {'dataset_sha256': digest, 'features': FEATURES, 'targets': TARGETS,
                'seed': args.seed, 'test_size': args.test_size,
                'gain_penalty': weight, 'objective': metadata['objective'],
                'train_rows': train.tolist(), 'test_rows': test.tolist(),
                'versions': {package: version(package) for package in
                             ['numpy', 'scipy', 'pandas', 'scikit-learn', 'joblib']},
                'evaluation': 'Ideal PID, second-order Pade delay; raw unmodified gains.'}
    (args.output / 'split.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    references = []
    for row in frame.iloc[test].itertuples():
        plant = Plant(row.zeta, row.wn, row.tau, row.K)
        references.append(assess_controller(plant, np.array([row.Kp, row.Ki, row.Kd]), weight))
    summaries = []
    for name, (estimator, grid, polynomial, scale) in models(args.seed).items():
        steps = []
        if polynomial:
            steps.append(('polynomial', PolynomialFeatures(2, include_bias=False)))
        if scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', estimator))
        pipeline = Pipeline(steps)
        cv = GroupKFold(n_splits=min(args.cv, len(np.unique(groups[train]))))
        search = GridSearchCV(pipeline, grid or {}, scoring='neg_mean_squared_error',
                              cv=cv, n_jobs=1, error_score='raise')
        print(f'Training {name} ({len(train)} train / {len(test)} test rows)...', flush=True)
        started = perf_counter()
        search.fit(X.iloc[train], y[train], groups=groups[train])
        fit_seconds = perf_counter() - started
        pd.DataFrame(search.cv_results_).to_csv(args.output / f'{name}_cv.csv', index=False)
        started = perf_counter()
        predictions = search.predict(X.iloc[test])
        prediction_seconds = perf_counter() - started
        summary = {'model': name, **gain_metrics(y[test], predictions),
                   'fit_cv_seconds': fit_seconds,
                   'batch_prediction_ms_per_plant': 1000*prediction_seconds/len(test),
                   'train_r2': gain_metrics(y[train], search.predict(X.iloc[train]))['r2']}
        records = []
        for i, row in enumerate(frame.iloc[test].itertuples()):
            assessment = assess_controller(Plant(row.zeta, row.wn, row.tau, row.K), predictions[i], weight)
            ref = references[i]
            ratio = (assessment['itae'] / ref['itae']
                     if assessment['accepted'] and ref['accepted'] and ref['itae'] > 0 else None)
            record = {'source_row': int(test[i]), **assessment,
                      'reference_reason': ref['reason'], 'reference_itae': ref['itae'],
                      'reference_objective': ref['objective'],
                      'objective_ratio': (assessment['objective']/ref['objective']
                          if assessment['accepted'] and ref['accepted'] and ref['objective'] > 0 else None),
                      'itae_ratio': ratio}
            record.update({key: getattr(row, key) for key in PLANT_COLUMNS + FEATURES})
            record.update({f'predicted_{key}': float(predictions[i, j]) for j, key in enumerate(TARGETS)})
            record.update({f'target_{key}': float(y[test[i], j]) for j, key in enumerate(TARGETS)})
            records.append(record)
        details = pd.DataFrame(records)
        summary.update(stable_pct=100 * float(details.stable.mean()),
                       bounds_violation_pct=100 * float((~details.within_bounds).mean()),
                       validated_success_pct=100 * float(details.accepted.mean()),
                       measured_overshoot_within_limit_pct=100 * float((details.overshoot_pct <= 10 + 1e-6).mean()),
                       overshoot_measured_count=int(details.overshoot_pct.notna().sum()),
                       itae_ratio_count=int(details.itae_ratio.notna().sum()),
                       reference_validated_count=int((details.reference_reason == 'accepted').sum()),
                       objective_ratio_count=int(details.objective_ratio.notna().sum()),
                       median_objective_ratio=details.objective_ratio.median(),
                       p90_objective_ratio=details.objective_ratio.quantile(.9),
                       median_itae_ratio=details.itae_ratio.median())
        details.to_csv(args.output / f'{name}_predictions.csv', index=False)
        joblib.dump({'pipeline': search.best_estimator_, 'features': FEATURES,
                     'targets': TARGETS, 'best_params': search.best_params_, **manifest},
                    args.output / f'{name}.joblib')
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(args.output / 'metrics.csv', index=False)
        print(f"  R2={summary['r2']:.3f}; stable={summary['stable_pct']:.1f}%; "
              f"validated={summary['validated_success_pct']:.1f}%", flush=True)
    print(f'Results and fitted pipelines saved to {args.output}')

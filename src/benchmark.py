"""Five-feature data loading, gain metrics and raw-controller validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupShuffleSplit
from data_gen import Plant,PIDGains,PlantSimulator,PIDOptimizer

FEATURES=['t_rise','t_settling','y_peak','y_ss','tau_est']
TARGETS=['Kp','Ki','Kd']
PLANT_COLUMNS=['zeta','wn','tau','K']

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

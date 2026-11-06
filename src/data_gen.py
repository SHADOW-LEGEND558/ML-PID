import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.signal as sps
from tqdm import tqdm


@dataclass(frozen=True)
class Plant:
    zeta: float
    wn: float
    tau: float
    K: float


@dataclass(frozen=True)
class PIDGains:
    Kp: float
    Ki: float
    Kd: float


class PlantSimulator:
    def get_dynamic_t_eval(self, plant: Plant, n_points: int = 1000) -> np.ndarray:
        """Computes a dynamic simulation time vector suited to plant dynamics."""
        self.validate_plant(plant)
        rate = plant.zeta * plant.wn
        if plant.zeta > 1.0:
            rate = plant.wn / (plant.zeta + np.sqrt(plant.zeta**2 - 1.0))
        t_max = max(10.0, 12.0 / rate + 3.0 * plant.tau)
        return np.linspace(0, t_max, n_points)

    @staticmethod
    def validate_plant(plant: Plant) -> None:
        values = [plant.zeta, plant.wn, plant.tau, plant.K]
        if not np.all(np.isfinite(values)) or min(plant.zeta, plant.wn, plant.K) <= 0 or plant.tau < 0:
            raise ValueError("Plant requires finite zeta, wn, K > 0 and tau >= 0.")

    def open_loop(self, plant: Plant, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized analytical open-loop unit step response."""
        t = t_eval
        tau, K, wn, zeta = plant.tau, plant.K, plant.wn, plant.zeta

        y = np.zeros_like(t)
        mask = t >= tau
        t_shifted = t[mask] - tau

        if len(t_shifted) == 0:
            return t, y

        if zeta < 0.999:  # Underdamped
            wd = wn * np.sqrt(1.0 - zeta**2)
            phi = np.arccos(zeta)
            y_shifted = K * (1.0 - (np.exp(-zeta * wn * t_shifted) / np.sqrt(1.0 - zeta**2)) * np.sin(wd * t_shifted + phi))
        elif zeta > 1.001:  # Overdamped
            s1 = wn * (zeta - np.sqrt(zeta**2 - 1.0))
            s2 = wn * (zeta + np.sqrt(zeta**2 - 1.0))
            y_shifted = K * (1.0 - (s2 * np.exp(-s1 * t_shifted) - s1 * np.exp(-s2 * t_shifted)) / (s2 - s1))
        else:  # Critically damped
            y_shifted = K * (1.0 - (1.0 + wn * t_shifted) * np.exp(-wn * t_shifted))

        y[mask] = y_shifted
        return t, y

    def closed_loop_coefficients(self, plant: Plant, gains: PIDGains) -> tuple[np.ndarray, np.ndarray]:
        """Ideal PID with unity negative feedback and second-order Pade delay."""
        self.validate_plant(plant)
        if not np.all(np.isfinite([gains.Kp, gains.Ki, gains.Kd])):
            raise ValueError("PID gains must be finite.")
        
        # Plant G(s) = K * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        num_g = [plant.K * (plant.wn**2)]
        den_g = [1.0, 2.0 * plant.zeta * plant.wn, plant.wn**2]

        # 2nd-Order Pade delay approximation: e^(-s*tau)
        if plant.tau > 1e-5:
            tau2 = plant.tau**2
            num_p = [tau2 / 12.0, -plant.tau / 2.0, 1.0]
            den_p = [tau2 / 12.0, plant.tau / 2.0, 1.0]
            num_gp = np.polymul(num_g, num_p)
            den_gp = np.polymul(den_g, den_p)
        else:
            num_gp, den_gp = num_g, den_g

        # Controller C(s) = (Kd*s^2 + Kp*s + Ki) / s
        # Avoid an artificial integrator pole for P/PD control.
        num_c = [gains.Kd, gains.Kp, gains.Ki] if gains.Ki != 0 else [gains.Kd, gains.Kp]
        den_c = [1.0, 0.0] if gains.Ki != 0 else [1.0]

        # Loop transfer function L(s) = C(s) * G(s)
        num_l = np.polymul(num_gp, num_c)
        den_l = np.polymul(den_gp, den_c)

        # Closed-loop T(s) = L(s) / (1 + L(s))
        max_len = max(len(den_l), len(num_l))
        num_l_padded = np.pad(num_l, (max_len - len(num_l), 0), 'constant')
        den_l_padded = np.pad(den_l, (max_len - len(den_l), 0), 'constant')

        den_cl = np.polyadd(den_l_padded, num_l_padded)
        num_cl = np.trim_zeros(num_l, 'f')
        if num_cl.size == 0:
            return np.array([0.0]), np.asarray(den_gp, dtype=float)

        # Normalize transfer-function coefficients.
        norm_factor = den_cl[0]
        return np.asarray(num_cl / norm_factor), np.asarray(den_cl / norm_factor)

    def close_loop(self, plant: Plant, gains: PIDGains, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-loop output to a unit reference step (zero initial state)."""
        num, den = self.closed_loop_coefficients(plant, gains)
        if not np.any(num):
            return t_eval, np.zeros_like(t_eval)
        T_cl = self._state_space(num, den)
        t = t_eval
        t_out, y_out = sps.step(T_cl, T=t)
        return t_out, y_out

    @staticmethod
    def _state_space(num: np.ndarray, den: np.ndarray) -> sps.StateSpace:
        """Canonical realization without dropping small leading coefficients.

        scipy TransferFunction normalization can discard small derivative
        coefficients during finite differencing. Preserve them explicitly.
        """
        num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
        num, den = num / den[0], den / den[0]
        order = len(den) - 1
        padded = np.pad(num, (len(den) - len(num), 0))
        direct = padded[0]
        a = np.zeros((order, order))
        a[0] = -den[1:]
        a[1:, :-1] = np.eye(order - 1)
        b = np.zeros((order, 1))
        b[0, 0] = 1.0
        c = (padded[1:] - direct * den[1:]).reshape(1, -1)
        return sps.StateSpace(a, b, c, np.array([[direct]]))

    def control_effort(self, plant: Plant, gains: PIDGains, t_eval: np.ndarray) -> tuple[np.ndarray, float]:
        """Regular part of unit-step control input and impulse weight at t=0.

        U/R = C*(1-Y/R). An ideal derivative on a reference step produces
        Kd*delta(t); that impulse cannot be plotted as a finite sample.
        """
        num, den = self.closed_loop_coefficients(plant, gains)
        c_num = [gains.Kd, gains.Kp, gains.Ki] if gains.Ki != 0 else [gains.Kd, gains.Kp]
        c_den = [1.0, 0.0] if gains.Ki != 0 else [1.0]
        step_num = np.polymul(c_num, np.polysub(den, num))
        step_den = np.polymul(np.polymul(c_den, den), [1.0, 0.0])
        quotient, remainder = np.polydiv(step_num, step_den)
        remainder = np.trim_zeros(remainder, 'f')
        regular = np.zeros_like(t_eval)
        if remainder.size and np.any(remainder):
            _, regular = sps.impulse(self._state_space(remainder, step_den), T=t_eval)
        return regular, float(quotient[-1])


class FeatureExtractor:
    """Extracts step response features from open-loop data."""

    def extract(self, t: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_ss = float(y[-1]) if len(y) > 0 else 0.0
        y_peak = float(np.max(y)) if len(y) > 0 else 0.0

        val_10, val_90 = 0.1 * y_ss, 0.9 * y_ss
        idx_10 = np.where(y >= val_10)[0]
        idx_90 = np.where(y >= val_90)[0]

        t_10 = t[idx_10[0]] if len(idx_10) > 0 else 0.0
        t_90 = t[idx_90[0]] if len(idx_90) > 0 else 0.0
        t_rise = max(0.0, float(t_90 - t_10))

        band = 0.02 * abs(y_ss) if abs(y_ss) > 1e-5 else 0.02
        out_of_bounds = np.where(np.abs(y - y_ss) > band)[0]
        t_settling = float(t[out_of_bounds[-1]]) if len(out_of_bounds) > 0 else 0.0

        auc = float(spi.simpson(y, x=t))
        idx_delay = np.where(y >= 0.01 * y_ss)[0]
        tau_est = float(t[idx_delay[0]]) if len(idx_delay) > 0 else 0.0

        return {
            "t_rise": t_rise,
            "t_settling": t_settling,
            "y_peak": y_peak,
            "y_ss": y_ss,
            "AUC": auc,
            "tau_est": tau_est
        }


class PIDOptimizer:
    """Multi-start local ITAE minimization followed by independent validation.

    All stability/response checks concern the second-order Pade model, not
    an exact-delay guarantee. Resource limits fail closed with diagnostics.
    """

    BOUNDS = np.array([(0.001, 100.0), (0.001, 100.0), (0.0, 50.0)])

    def __init__(self, simulator: PlantSimulator, n_starts: int = 4,
                 max_duration: float = 1000.0, max_validation_points: int = 65537) -> None:
        if n_starts < 1 or max_duration <= 0 or max_validation_points < 8001:
            raise ValueError("Invalid optimization/validation limits.")
        self.simulator = simulator
        self.n_starts = n_starts
        self.max_duration = max_duration
        self.max_validation_points = max_validation_points
        self.last_validation: Optional[dict[str, float]] = None
        self.diagnostics: list[dict] = []
        self.initial_guesses: list[np.ndarray] = []
        self._cache: dict = {}

    def _get_initial_guess(self, plant: Plant) -> list[float]:
        """SIMC-inspired starting heuristic, not a general SIMC tuning rule."""
        tau1 = 1.0 / (plant.zeta * plant.wn)
        tau_c = max(0.1, plant.tau)
        kp_init = max(0.1, tau1 / (plant.K * (tau_c + plant.tau)))
        ki_init = max(0.01, kp_init / min(tau1, 4.0 * (tau_c + plant.tau)))
        kd_init = max(0.0, kp_init * (0.5 / plant.wn))
        return [kp_init, ki_init, kd_init]

    def _get_initial_guesses(self, plant: Plant, t_eval: Optional[np.ndarray] = None) -> list[np.ndarray]:
        """Find distinct stable, sampled-feasible seeds before launching SLSQP.

        Back off heuristic gains and probe conservative PI alternatives. This
        screening is not final validation: it only makes the search starts
        feasible on the optimization grid, with a little overshoot headroom.
        """
        if t_eval is None:
            t_eval = self.simulator.get_dynamic_t_eval(plant)
        base = np.asarray(self._get_initial_guess(plant))
        factors = [np.ones(3), np.array([0.3, 0.1, 0.5]),
                   np.array([0.7, 0.3, 1.5]), np.array([1.5, 0.7, 2.0])]
        candidates = []
        # Try decreasing aggressiveness; preserve the heuristic's relative shape.
        for backoff in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
            candidates.extend(base * factor * backoff for factor in factors)
        # Independent, plant-scaled PI seeds prevent every start inheriting the
        # same problematic derivative/ integral proportions from the heuristic.
        for kp in [0.01, 0.03, 0.1, 0.3]:
            for ki in [0.01, 0.03, 0.1, 0.3]:
                candidates.append(np.array([kp / plant.K, ki * plant.wn / plant.K, 0.0]))
        rng = np.random.default_rng(42)
        # Additional deterministic candidates support user-requested larger pools.
        for _ in range(min(128, max(0, self.n_starts * 4 - 16))):
            candidates.append(base * np.exp(rng.uniform(np.log(0.005), np.log(1.5), 3)))
        pool = []
        seen = set()
        for candidate in candidates:
            candidate = np.clip(candidate, self.BOUNDS[:, 0], self.BOUNDS[:, 1])
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            metrics = self._evaluate(candidate, plant, t_eval)
            if (metrics['pole_real'] < -1e-8 and np.isfinite(metrics['itae'])
                    and metrics['overshoot'] <= 9.5):
                pool.append((candidate, metrics['itae']))
        if not pool:
            return []
        # Prefer useful candidates, then spread starts in log-gain space.
        pool.sort(key=lambda item: item[1])
        shortlist = pool[:max(12, self.n_starts * 3)]
        selected = [shortlist.pop(0)[0]]
        while shortlist and len(selected) < self.n_starts:
            distances = [min(float(np.linalg.norm(
                np.log(np.maximum(item[0], 1e-6)) - np.log(np.maximum(start, 1e-6))))
                for start in selected) for item in shortlist]
            selected.append(shortlist.pop(int(np.argmax(distances)))[0])
        return selected

    def _evaluate(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> dict:
        """Share simulation results between objective and both constraints."""
        key = (plant, tuple(float(x) for x in gains_vec), t_eval.size,
               float(t_eval[-1]))
        if key in self._cache:
            return self._cache[key]
        result = {'itae': 1e12, 'overshoot': np.inf, 'pole_real': np.inf}
        try:
            gains = PIDGains(*map(float, gains_vec))
            _, den = self.simulator.closed_loop_coefficients(plant, gains)
            pole_real = float(np.max(np.roots(den).real))
            result['pole_real'] = pole_real
            if pole_real < -1e-8:
                t, y = self.simulator.close_loop(plant, gains, t_eval)
                if np.all(np.isfinite(y)):
                    score = float(spi.simpson(t * np.abs(1.0 - y), x=t))
                    if np.isfinite(score):
                        result.update(itae=score, overshoot=max(0.0, float(y.max()) - 1.0) * 100)
            elif np.isfinite(pole_real):
                result['itae'] = 1e8 * (1 + max(0.0, pole_real))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        if len(self._cache) >= 512:
            self._cache.clear()
        self._cache[key] = result
        return result

    def _itae_objective(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> float:
        return self._evaluate(gains_vec, plant, t_eval)['itae']

    def _scaled_itae_objective(self, gains_vec: np.ndarray, plant: Plant,
                               t_eval: np.ndarray, scale: float) -> float:
        # A positive constant scale preserves the minimizer while avoiding huge
        # first SLSQP steps for conservative controllers with large initial ITAE.
        return self._itae_objective(gains_vec, plant, t_eval) / scale

    def _overshoot_constraint(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> float:
        overshoot = self._evaluate(gains_vec, plant, t_eval)['overshoot']
        return 10.0 - overshoot if np.isfinite(overshoot) else -1e4

    def _stability_constraint(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> float:
        pole = self._evaluate(gains_vec, plant, t_eval)['pole_real']
        return -pole - 1e-8 if np.isfinite(pole) else -1e4

    def _response_metrics(self, plant: Plant, gains: PIDGains,
                          duration: float, n_points: int) -> dict[str, float]:
        t = np.linspace(0, duration, n_points)
        _, y = self.simulator.close_loop(plant, gains, t)
        if not np.all(np.isfinite(y)):
            raise ValueError("Nonfinite validation response.")
        peak = float(max(1.0, y.max()))
        # Refine the largest sampled local maxima using exact LTI evaluations
        # at arbitrary times (two-point step computes exp(A*t), not interpolation).
        local = sps.find_peaks(y)[0]
        intervals = [(t[0], t[1])]
        intervals.extend((t[i - 1], t[i + 1])
                         for i in sorted(local, key=lambda index: y[index], reverse=True)[:8]
                         if y[i] >= 1.0)
        for left, right in intervals:
            def negative_output(time: float) -> float:
                return -float(self.simulator.close_loop(
                    plant, gains, np.array([0.0, time]))[1][-1])
            refined = spo.minimize_scalar(negative_output,
                bounds=(left, right), method='bounded',
                options={'xatol': max(1e-10, duration * 1e-10)})
            if refined.success and np.isfinite(refined.fun):
                peak = max(peak, -float(refined.fun))
        error = np.abs(1.0 - y)
        score = float(spi.simpson(t * error, x=t))
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation ITAE.")
        return {'itae': score,
                'overshoot_pct': (peak - 1.0) * 100,
                'final_error': float(error[-1]),
                'tail_error': float(np.max(error[int(0.9 * n_points):]))}

    def _validate_candidate(self, plant: Plant, gains: PIDGains,
                            t_eval: np.ndarray) -> tuple[Optional[dict], str]:
        vector = np.array([gains.Kp, gains.Ki, gains.Kd])
        if not np.all(np.isfinite(vector)) or np.any(vector < self.BOUNDS[:, 0]) or np.any(vector > self.BOUNDS[:, 1]):
            return None, 'invalid_gains'
        try:
            _, den = self.simulator.closed_loop_coefficients(plant, gains)
            poles = np.roots(den)
            if not np.all(np.isfinite(poles)) or np.max(poles.real) >= -1e-8:
                return None, 'unstable'
            duration = max(float(t_eval[-1]), 12.0 / -float(np.max(poles.real)))
            previous = None
            for _ in range(5):
                if duration > self.max_duration:
                    return None, 'duration_limit'
                # At least 32 samples per fastest oscillatory period.
                n = max(4001, int(np.ceil(duration * np.max(abs(poles.imag)) * 16 / np.pi)) + 1)
                while 2 * n - 1 <= self.max_validation_points:
                    coarse = self._response_metrics(plant, gains, duration, n)
                    fine = self._response_metrics(plant, gains, duration, 2 * n - 1)
                    if fine['overshoot_pct'] > 10.0 + 1e-6:
                        return None, 'overshoot'
                    score_tolerance = max(1e-8, 1e-3 * fine['itae'])
                    resolved = (abs(fine['itae'] - coarse['itae']) <= score_tolerance
                                and abs(fine['overshoot_pct'] - coarse['overshoot_pct']) <= 1e-3)
                    if resolved:
                        break
                    n = 2 * n - 1
                else:
                    return None, 'resolution_limit'
                # Compare two horizons as well as two grids. A small final error
                # alone does not establish that omitted time-weighted error is small.
                if previous is not None and fine['tail_error'] <= 1e-3 and abs(fine['itae'] - previous) <= score_tolerance:
                    fine.update(duration=duration, n_points=2 * n - 1,
                                max_pole_real=float(np.max(poles.real)))
                    return fine, 'accepted'
                previous = fine['itae']
                duration *= 2
            return None, 'tail_not_converged'
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None, 'simulation_failure'

    def optimize(self, plant: Plant, t_eval: np.ndarray) -> Optional[PIDGains]:
        self.simulator.validate_plant(plant)
        if (len(t_eval) < 2 or t_eval[0] != 0 or not np.all(np.isfinite(t_eval))
                or np.any(np.diff(t_eval) <= 0)
                or not np.allclose(np.diff(t_eval), t_eval[1] - t_eval[0])):
            raise ValueError("Optimization time grid must be uniform, finite, increasing, and start at zero.")
        self._cache.clear()
        self.last_validation = None
        self.diagnostics = []
        self.initial_guesses = self._get_initial_guesses(plant, t_eval)
        best_gains, best_score = None, np.inf
        constraints = [
            {'type': 'ineq', 'fun': self._overshoot_constraint, 'args': (plant, t_eval)},
            {'type': 'ineq', 'fun': self._stability_constraint, 'args': (plant, t_eval)}]
        for start_index, x0 in enumerate(self.initial_guesses):
            objective_scale = max(1.0, self._itae_objective(x0, plant, t_eval))
            res = spo.minimize(self._scaled_itae_objective, x0=x0, args=(plant, t_eval, objective_scale),
                method='SLSQP', bounds=self.BOUNDS, constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-6})
            record = {'start': start_index, 'solver_success': bool(res.success),
                      'message': str(res.message), 'iterations': int(res.nit),
                      'initial_gains': x0.tolist()}
            if res.success and np.all(np.isfinite(res.x)):
                gains = PIDGains(*map(float, res.x))
                metrics, reason = self._validate_candidate(plant, gains, t_eval)
                record['validation'] = reason
                if metrics is not None:
                    record['itae'] = metrics['itae']
                    if metrics['itae'] < best_score:
                        best_gains, best_score = gains, metrics['itae']
                        self.last_validation = metrics
            else:
                record['validation'] = 'optimizer_failure'
            self.diagnostics.append(record)
        return best_gains


def _process_single_plant(plant: Plant, n_starts: int = 4) -> Optional[dict[str, float]]:
    simulator = PlantSimulator()
    extractor = FeatureExtractor()
    optimizer = PIDOptimizer(simulator, n_starts=n_starts)

    try:
        t_eval = simulator.get_dynamic_t_eval(plant)
        t_ol, y_ol = simulator.open_loop(plant, t_eval)
        features = extractor.extract(t_ol, y_ol)
        gains = optimizer.optimize(plant, t_eval)

        if gains is None:
            reasons = Counter(record['validation'] for record in optimizer.diagnostics)
            if not optimizer.initial_guesses:
                reasons['no_feasible_start'] += 1
            print(f"Discarded {plant}: {dict(reasons)}", flush=True)
            return None  # Filter out unoptimized / failed plants

        return {
            "zeta": plant.zeta,
            "wn": plant.wn,
            "tau": plant.tau,
            "K": plant.K,
            **features,
            "Kp": gains.Kp,
            "Ki": gains.Ki,
            "Kd": gains.Kd
        }
    except Exception as exc:
        print(f"Failed {plant}: {type(exc).__name__}: {exc}", flush=True)
        return None


class DatasetGenerator:
    def __init__(self, n_samples: int = 5000, seed: int = 42, output_path: str = "data/pid_dataset.csv", n_starts: int = 4, max_attempts: Optional[int] = None) -> None:
        if n_samples < 1 or n_starts < 1:
            raise ValueError("Samples and starting points must be positive.")
        self.n_samples = n_samples
        self.seed = seed
        self.output_path = output_path
        self.n_starts = n_starts
        self.max_attempts = max_attempts if max_attempts is not None else 10 * n_samples
        if self.max_attempts < n_samples:
            raise ValueError("Maximum attempts must be at least the requested sample count.")

    def _sample_plants(self, count: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> list[Plant]:
        count = self.n_samples if count is None else count
        rng = np.random.default_rng(self.seed) if rng is None else rng
        zetas = rng.uniform(0.1, 2.0, count)
        wns = rng.uniform(0.5, 10.0, count)
        taus = rng.uniform(0.0, 1.0, count)
        Ks = rng.uniform(0.5, 2.0, count)

        return [
            Plant(zeta=float(z), wn=float(w), tau=float(t), K=float(k))
            for z, w, t, k in zip(zetas, wns, taus, Ks)
        ]

    def run(self, n_jobs: int = -1) -> pd.DataFrame:
        # Keep one RNG stream across replacement batches; never retry the same
        # failed plant by accidentally resetting the seed.
        rng = np.random.default_rng(self.seed)
        data = []
        attempted = 0

        print(f"Generating {self.n_samples} accepted systems with {self.n_starts} PID starts per plant using multiprocessing...")
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            with tqdm(total=self.n_samples, desc="Accepted Plants") as progress:
                while len(data) < self.n_samples:
                    count = min(self.n_samples - len(data), self.max_attempts - attempted)
                    if count == 0:
                        raise RuntimeError(
                            f"Only {len(data)}/{self.n_samples} valid plants after {attempted} attempts. "
                            "No output was written. Inspect tuning failures or increase --max-attempts.")
                    plants = self._sample_plants(count=count, rng=rng)
                    futures = {executor.submit(_process_single_plant, plant, self.n_starts): i
                               for i, plant in enumerate(plants)}
                    accepted = {}
                    for future in as_completed(futures):
                        res = future.result()
                        attempted += 1
                        if res is not None:
                            accepted[futures[future]] = res
                            progress.update(1)
                        progress.set_postfix(attempted=attempted, rejected=attempted - len(data) - len(accepted))
                    # Stable CSV ordering regardless of worker completion order.
                    data.extend(accepted[i] for i in sorted(accepted))

        df = pd.DataFrame(data)
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully saved to {output_file} ({len(df)} valid records from {attempted} attempts).")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PID training dataset.")
    parser.add_argument("--samples", type=int, default=5, help="Number of accepted rows to save; failed plants are replaced")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/pid_dataset.csv", help="Output CSV path")
    parser.add_argument("--starts", type=int, default=4, help="Deterministic PID starting points per plant")
    parser.add_argument("--jobs", type=int, default=-1, help="Worker count; -1 uses the executor default")
    parser.add_argument("--max-attempts", type=int, default=None, help="Total plant attempt limit (default: 10 times --samples)")
    args = parser.parse_args()

    generator = DatasetGenerator(n_samples=args.samples, seed=args.seed, output_path=args.output, n_starts=args.starts, max_attempts=args.max_attempts)
    generator.run(n_jobs=args.jobs)

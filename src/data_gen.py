import argparse
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
        # Settling time estimate for 2nd order system: 4 / (zeta * wn)
        tau_dom = 1.0 / (plant.zeta * plant.wn)
        t_settle_est = 8.0 * tau_dom + 3.0 * plant.tau
        t_max = max(10.0, min(100.0, t_settle_est))
        return np.linspace(0, t_max, n_points)

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

    def close_loop(self, plant: Plant, gains: PIDGains, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-loop unit step response using explicit 2nd-order Pade approximation."""
        t = t_eval
        
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
        num_c = [gains.Kd, gains.Kp, gains.Ki]
        den_c = [1.0, 0.0]

        # Loop transfer function L(s) = C(s) * G(s)
        num_l = np.polymul(num_gp, num_c)
        den_l = np.polymul(den_gp, den_c)

        # Closed-loop T(s) = L(s) / (1 + L(s))
        max_len = max(len(den_l), len(num_l))
        num_l_padded = np.pad(num_l, (max_len - len(num_l), 0), 'constant')
        den_l_padded = np.pad(den_l, (max_len - len(den_l), 0), 'constant')

        den_cl = np.polyadd(den_l_padded, num_l_padded)
        num_cl = num_l_padded

        # Normalize filter coefficients to avoid numerical ill-conditioning
        norm_factor = den_cl[0]
        T_cl = sps.TransferFunction(num_cl / norm_factor, den_cl / norm_factor)
        t_out, y_out = sps.step(T_cl, T=t)
        return t_out, y_out


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
    """Optimizes PID gains using SLSQP initialized with heuristic SIMC parameters."""

    def __init__(self, simulator: PlantSimulator) -> None:
        self.simulator = simulator

    def _get_initial_guess(self, plant: Plant) -> list[float]:
        """Analytical SIMC heuristic formula for reliable initial guess x0."""
        tau1 = 1.0 / (plant.zeta * plant.wn)
        tau_c = max(0.1, plant.tau)
        kp_init = max(0.1, (1.0 / plant.K) * (tau1 / (tau_c + plant.tau)))
        ki_init = max(0.01, kp_init / min(tau1, 4.0 * (tau_c + plant.tau)))
        kd_init = max(0.0, kp_init * (0.5 / plant.wn))
        return [kp_init, ki_init, kd_init]

    def _itae_objective(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> float:
        gains = PIDGains(Kp=gains_vec[0], Ki=gains_vec[1], Kd=gains_vec[2])
        try:
            t, y_cl = self.simulator.close_loop(plant, gains, t_eval)
            if np.any(np.isnan(y_cl)) or np.any(np.isinf(y_cl)) or np.max(y_cl) > 10.0:
                return 1e6
            error = np.abs(1.0 - y_cl)
            itae = float(spi.simpson(t * error, x=t))
            return itae if not np.isnan(itae) else 1e6
        except Exception:
            return 1e6

    def _overshoot_constraint(self, gains_vec: np.ndarray, plant: Plant, t_eval: np.ndarray) -> float:
        gains = PIDGains(Kp=gains_vec[0], Ki=gains_vec[1], Kd=gains_vec[2])
        try:
            _, y_cl = self.simulator.close_loop(plant, gains, t_eval)
            y_peak = np.max(y_cl)
            overshoot_pct = max(0.0, (y_peak - 1.0) * 100.0)
            return 10.0 - overshoot_pct  # Overshoot <= 10%
        except Exception:
            return -100.0

    def optimize(self, plant: Plant, t_eval: np.ndarray) -> Optional[PIDGains]:
        x0 = self._get_initial_guess(plant)
        bounds = [(0.001, 100.0), (0.001, 100.0), (0.0, 50.0)]
        constraints = {'type': 'ineq', 'fun': self._overshoot_constraint, 'args': (plant, t_eval)}

        res = spo.minimize(
            self._itae_objective,
            x0=x0,
            args=(plant, t_eval),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-4}
        )

        # Return gains ONLY if optimizer successfully converged
        if res.success and not np.isnan(res.x[0]):
            return PIDGains(Kp=float(res.x[0]), Ki=float(res.x[1]), Kd=float(res.x[2]))
        return None  # Return None to discard failed optimizations and preserve dataset quality


def _process_single_plant(plant: Plant) -> Optional[dict[str, float]]:
    simulator = PlantSimulator()
    extractor = FeatureExtractor()
    optimizer = PIDOptimizer(simulator)

    try:
        t_eval = simulator.get_dynamic_t_eval(plant)
        t_ol, y_ol = simulator.open_loop(plant, t_eval)
        features = extractor.extract(t_ol, y_ol)
        gains = optimizer.optimize(plant, t_eval)

        if gains is None:
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
    except Exception:
        return None


class DatasetGenerator:
    def __init__(self, n_samples: int = 5000, seed: int = 42, output_path: str = "data/pid_dataset.csv") -> None:
        self.n_samples = n_samples
        self.seed = seed
        self.output_path = output_path

    def _sample_plants(self) -> list[Plant]:
        rng = np.random.default_rng(self.seed)
        zetas = rng.uniform(0.1, 2.0, self.n_samples)
        wns = rng.uniform(0.5, 10.0, self.n_samples)
        taus = rng.uniform(0.0, 1.0, self.n_samples)
        Ks = rng.uniform(0.5, 2.0, self.n_samples)

        return [
            Plant(zeta=float(z), wn=float(w), tau=float(t), K=float(k))
            for z, w, t, k in zip(zetas, wns, taus, Ks)
        ]

    def run(self, n_jobs: int = -1) -> pd.DataFrame:
        plants = self._sample_plants()
        data = []

        print(f"Generating dataset with {self.n_samples} systems using multiprocessing...")
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = [executor.submit(_process_single_plant, plant) for plant in plants]
            for future in tqdm(as_completed(futures), total=self.n_samples, desc="Processing Plants"):
                res = future.result()
                if res is not None:
                    data.append(res)

        df = pd.DataFrame(data)
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Dataset successfully saved to {output_file} ({len(df)} valid records).")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PID training dataset.")
    parser.add_argument("--samples", type=int, default=5, help="Number of plants to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/pid_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    generator = DatasetGenerator(n_samples=args.samples, seed=args.seed, output_path=args.output)
    generator.run()
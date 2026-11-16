"""Numerical regression checks; run with python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_gen import Plant, PIDGains, PlantSimulator, PIDOptimizer


class PIDOptimizationTests(unittest.TestCase):
    def test_regularized_objective_is_dimensionless_and_decomposed(self):
        sim = PlantSimulator()
        opt = PIDOptimizer(sim, gain_penalty=.01)
        p = Plant(.7,2,0,1)
        terms = opt.objective_terms(p, np.array([2.,4.,1.]), .5)
        self.assertAlmostEqual(terms['normalized_itae'], 2)
        self.assertAlmostEqual(terms['gain_cost'], .12)
        self.assertAlmostEqual(terms['objective'], 2.12)
        scaled = opt.objective_terms(Plant(.7,4,0,2), np.array([1.,4.,.25]), .125)
        self.assertAlmostEqual(scaled['objective'], terms['objective'])
        self.assertAlmostEqual(PIDOptimizer(sim, gain_penalty=0).objective_terms(p,
            np.array([2.,4.,1.]), .5)['objective'], 2)
        with self.assertRaises(ValueError):
            PIDOptimizer(sim, gain_penalty=-.01)

    def test_zero_delay_regularization_avoids_upper_bounds(self):
        sim = PlantSimulator()
        opt = PIDOptimizer(sim)
        gains = opt.optimize(Plant(.7,2,0,1), sim.get_dynamic_t_eval(Plant(.7,2,0,1)))
        self.assertIsNotNone(gains)
        if gains is not None:
            self.assertTrue(np.all(np.array([gains.Kp,gains.Ki,gains.Kd]) < opt.BOUNDS[:,1] - 1e-5))

    def setUp(self):
        self.sim = PlantSimulator()

    def test_overdamped_horizon_reaches_true_dc_gain(self):
        plant = Plant(2, 0.5, 0, 1)
        t = self.sim.get_dynamic_t_eval(plant)
        self.assertGreater(t[-1], 80)
        self.assertAlmostEqual(self.sim.open_loop(plant, t)[1][-1], 1, places=4)

    def test_ideal_pid_cancellation_matches_analytic_first_order(self):
        plant = Plant(0.7, 2, 0, 1)
        # C = (s^2 + 2.8s + 4)/s gives T = 4/(s+4).
        t = np.linspace(0, 5, 2001)
        _, y = self.sim.close_loop(plant, PIDGains(2.8, 4, 1), t)
        np.testing.assert_allclose(y, 1 - np.exp(-4 * t), atol=1e-10)

    def test_pd_and_zero_controller_have_no_artificial_integrator(self):
        plant = Plant(0.7, 2, 0, 1)
        t = np.linspace(0, 50, 3000)
        _, den = self.sim.closed_loop_coefficients(plant, PIDGains(2, 0, 0))
        self.assertEqual(len(den), 3)
        y = self.sim.close_loop(plant, PIDGains(2, 0, 0), t)[1]
        self.assertAlmostEqual(y[-1], 2 / 3, places=6)
        np.testing.assert_array_equal(
            self.sim.close_loop(plant, PIDGains(0, 0, 0), t)[1], 0)

    def test_control_effort_separates_derivative_impulse(self):
        plant = Plant(0.7, 2, 0, 1)
        t = np.linspace(0, 100, 10001)
        u, impulse = self.sim.control_effort(plant, PIDGains(1, 0.5, 0.2), t)
        self.assertAlmostEqual(impulse, 0.2)
        self.assertAlmostEqual(u[0], 0.84, places=10)
        self.assertAlmostEqual(u[-1], 1, places=6)

    def test_rejects_unstable_and_overshooting_controllers(self):
        opt = PIDOptimizer(self.sim)
        plant = Plant(0.2, 2, 1, 1)
        _, reason = opt._validate_candidate(plant, PIDGains(20, 10, 0),
                                            self.sim.get_dynamic_t_eval(plant))
        self.assertEqual(reason, 'unstable')
        plant = Plant(0.2, 2, 0, 1)
        _, reason = opt._validate_candidate(plant, PIDGains(10, 1, 0),
                                            self.sim.get_dynamic_t_eval(plant))
        self.assertEqual(reason, 'overshoot')

    def test_validation_resource_limit_fails_closed(self):
        plant = Plant(0.7, 2, 0.2, 1)
        opt = PIDOptimizer(self.sim, max_duration=1)
        metrics, reason = opt._validate_candidate(plant, PIDGains(1, 0.5, 0.2),
                                                   self.sim.get_dynamic_t_eval(plant))
        self.assertIsNone(metrics)
        self.assertEqual(reason, 'duration_limit')

    def test_objective_and_constraint_reuse_one_simulation(self):
        plant = Plant(0.7, 2, 0.2, 1)
        opt = PIDOptimizer(self.sim)
        t = self.sim.get_dynamic_t_eval(plant)
        calls = []
        original = self.sim.close_loop
        def counted(*args):
            calls.append(1)
            return original(*args)
        self.sim.close_loop = counted
        gains = np.array([1, 0.5, 0.2])
        opt._itae_objective(gains, plant, t)
        opt._overshoot_constraint(gains, plant, t)
        opt._stability_constraint(gains, plant, t)
        self.assertEqual(len(calls), 1)

    def test_multistart_returns_independently_validated_gains(self):
        plant = Plant(0.7, 2, 0.2, 1)
        opt = PIDOptimizer(self.sim, n_starts=2)
        gains = opt.optimize(plant, self.sim.get_dynamic_t_eval(plant))
        self.assertIsNotNone(gains)
        self.assertEqual(len(opt.diagnostics), 2)
        validation = opt.last_validation
        if validation is None:
            self.fail('Expected validation metrics for the accepted controller.')
        self.assertLessEqual(validation['overshoot_pct'], 10 + 1e-6)
        self.assertLess(validation['tail_error'], 1e-3)
        self.assertLess(validation['max_pole_real'], 0)
        accepted = [r['objective'] for r in opt.diagnostics if r['validation'] == 'accepted']
        self.assertEqual(validation['objective'], min(accepted))

    def test_previously_unstable_starts_are_replaced_by_feasible_seeds(self):
        plant = Plant(0.11, 4.77, 0.56, 1.72)
        opt = PIDOptimizer(self.sim)
        t = self.sim.get_dynamic_t_eval(plant)
        starts = opt._get_initial_guesses(plant, t)
        self.assertEqual(len(starts), 4)
        self.assertEqual(len({tuple(start) for start in starts}), 4)
        for start in starts:
            _, denominator = self.sim.closed_loop_coefficients(plant, PIDGains(*start))
            self.assertLess(np.max(np.roots(denominator).real), -1e-8)
            _, y = self.sim.close_loop(plant, PIDGains(*start), t)
            self.assertLessEqual(float(y.max()), 1.095 + 1e-10)
        gains = opt.optimize(plant, t)
        self.assertIsNotNone(gains)
        self.assertTrue(any(record['validation'] == 'accepted' for record in opt.diagnostics))

    def test_feasible_start_selection_is_repeatable(self):
        plant = Plant(0.7, 2, 0.2, 1)
        opt = PIDOptimizer(self.sim)
        first = opt._get_initial_guesses(plant)
        second = PIDOptimizer(self.sim)._get_initial_guesses(plant)
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()

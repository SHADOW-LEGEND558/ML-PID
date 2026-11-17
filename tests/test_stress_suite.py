"""Independent checks for the added delay and disturbance simulators."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_stress_suite import TrueDelayResponse, disturbance_task, exact_task
from data_gen import Plant, PIDGains, PlantSimulator


class StressSimulatorTests(unittest.TestCase):
    def test_zero_delay_matches_transfer_function_with_derivative_impulse(self):
        plant=Plant(.7,2.,0.,1.3)
        gains=[.8,.5,.2]
        times=np.linspace(0,15,3001)
        response=TrueDelayResponse(plant,gains,15,1e-10)
        expected=PlantSimulator().close_loop(plant,PIDGains(*gains),times)[1]
        np.testing.assert_allclose(response.output(times),expected,atol=2e-8,rtol=2e-8)

    def test_positive_delay_has_no_anticipatory_output_and_correct_impulse(self):
        plant=Plant(.7,2.,.3,1.3)
        gains=[.8,.5,.2]
        response=TrueDelayResponse(plant,gains,2,1e-10)
        np.testing.assert_array_equal(response.output([0,.1,.299]),[0,0,0])
        self.assertAlmostEqual(float(response.solutions[0](.3)[1]),1.3*4*.2,places=10)
        self.assertGreater(response.output([.301])[0],0)

    def test_true_delay_zero_gains_reduces_to_open_loop_p_action(self):
        # Feedback-free first delay interval: exactly the plant's step response.
        plant=Plant(.7,2.,.3,1.3)
        response=TrueDelayResponse(plant,[.8,0.,0.],.59,1e-10)
        times=np.linspace(.3,.59,51)
        expected=PlantSimulator().open_loop(plant,times)[1]*.8
        np.testing.assert_allclose(response.output(times),expected,atol=2e-8,rtol=2e-8)

    def test_small_true_delay_tends_to_zero_delay(self):
        gains=[.8,.5,.2]
        times=np.linspace(0,2,1001)
        delayed=TrueDelayResponse(Plant(.7,2.,.001,1.3),gains,2,1e-9)
        nominal=TrueDelayResponse(Plant(.7,2.,0.,1.3),gains,2,1e-9)
        self.assertIsNone(delayed.failed)
        self.assertLess(float(np.max(abs(delayed.output(times)-nominal.output(times)))),.003)

    def test_disturbance_input_normalization_survives_delay(self):
        no_delay=disturbance_task(({'test':'check'},[.7,2.,0.,1.3],[.8,.5,.2]))
        delayed=disturbance_task(({'test':'check'},[.7,2.,.05,1.3],[.8,.5,.2]))
        self.assertTrue(no_delay['recovered'])
        self.assertTrue(delayed['recovered'])
        self.assertLess(abs(delayed['max_deviation']/no_delay['max_deviation']-1),.05)
        self.assertLess(abs(delayed['disturbance_itae']/no_delay['disturbance_itae']-1),.01)


if __name__=='__main__':unittest.main()

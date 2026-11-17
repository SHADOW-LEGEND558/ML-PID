"""Checks for five-feature training and honest control evaluation."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from benchmark import FEATURES, gain_metrics, split_plants, assess_controller
from data_gen import Plant, PlantSimulator, FeatureExtractor, DatasetGenerator


class BaselineTests(unittest.TestCase):
    def test_mean_predictor_has_zero_r2(self):
        y = np.array([[1.,100.,.1],[2.,200.,.2],[3.,300.,.3]])
        prediction=np.tile(y.mean(axis=0),(len(y),1))
        self.assertAlmostEqual(gain_metrics(y,prediction)['r2'],0)

    def test_identical_plants_never_cross_split(self):
        frame = pd.DataFrame({'zeta': np.repeat(np.arange(10)+1,2),
                              'wn': 2., 'tau': .2, 'K': 1.})
        train,test,groups = split_plants(frame, .2, 42)
        self.assertFalse(set(groups[train]) & set(groups[test]))
        np.testing.assert_array_equal(test, split_plants(frame,.2,42)[1])

    def test_observation_and_schema(self):
        sim = PlantSimulator()
        durations = []
        for plant in [Plant(.7,2,.2,1), Plant(.1,.5,1,2), Plant(2,.5,0,1)]:
            t,y = sim.observe_open_loop(plant)
            self.assertAlmostEqual(t[1]-t[0], .01)
            self.assertLess(abs(y[-1]-plant.K), .002*plant.K)
            self.assertEqual(set(FeatureExtractor().extract(t,y)), set(FEATURES))
            durations.append(t[-1])
        self.assertGreater(durations[1],durations[0])
        with self.assertRaises(ValueError):
            sim.observe_open_loop(Plant(.1,.5,1,2), max_duration=10)

    def test_zero_delay_sampling(self):
        self.assertTrue(all(p.tau == 0 for p in DatasetGenerator(20,zero_delay_fraction=1)._sample_plants()))
        self.assertTrue(all(p.tau > 0 for p in DatasetGenerator(20,zero_delay_fraction=0)._sample_plants()))

    def test_control_failure_and_success_are_not_hidden(self):
        plant = Plant(.7,2,0,1)
        self.assertTrue(assess_controller(plant,np.array([2.8,4.,1.]))['accepted'])
        bad = assess_controller(plant,np.array([-2.,-1.,0.]))
        self.assertFalse(bad['stable'])
        self.assertFalse(bad['within_bounds'])
        self.assertFalse(bad['accepted'])
        finite_but_invalid = assess_controller(plant,np.array([200.,1.,0.]))
        self.assertFalse(finite_but_invalid['within_bounds'])
        self.assertFalse(finite_but_invalid['accepted'])
        self.assertEqual(assess_controller(plant,np.array([np.nan,1.,0.]))['reason'],'nonfinite_gains')

    def test_controller_objective_respects_weight(self):
        plant = Plant(.7,2,0,1)
        vector = np.array([2.8,4.,1.])
        raw = assess_controller(plant, vector, gain_penalty=0)
        regularized = assess_controller(plant, vector, gain_penalty=.01)
        self.assertTrue(raw['accepted'] and regularized['accepted'])
        self.assertAlmostEqual(raw['itae'], regularized['itae'])
        self.assertAlmostEqual(raw['objective'], 4*raw['itae'])
        self.assertAlmostEqual(regularized['objective']-raw['objective'],
                               .01*(2.8**2+2**2+2**2))


if __name__ == '__main__':
    unittest.main()

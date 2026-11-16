import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_baselines import overshoot_summary


class OvershootSummaryTests(unittest.TestCase):
    def test_missing_and_unstable_are_not_counted_as_passes(self):
        frame=pd.DataFrame({'stable':[True]*8+[False],
            'overshoot_pct':[0,10,10.5,12,15,20,30,np.nan,np.nan],
            'accepted':[True,True]+[False]*7})
        s=overshoot_summary({'example':frame}).iloc[0]
        self.assertEqual(s.at_most_10,2)
        self.assertEqual(s.over_10_to_12,2)
        self.assertEqual(s.over_12_to_15,1)
        self.assertEqual(s.over_15_to_20,1)
        self.assertEqual(s.over_20,1)
        self.assertEqual(s.stable_unmeasured,1)
        self.assertEqual(s.unstable,1)
        self.assertEqual(s.measured_stable,7)
        self.assertEqual(s.full_validation_passes,2)
        self.assertEqual(s.median_excess_pp,5)


if __name__=='__main__': unittest.main()

"""Checks that descriptive diagnostics retain stable overshoot misses."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from final_controller_diagnostics import diagnostic


class FinalDiagnosticTests(unittest.TestCase):
    def test_exact_first_order_itae_and_settling_time(self):
        # Controller cancels the plant denominator: T(s)=1/(s+1).
        r=diagnostic(('test',0,[.7,2.,0.,1.],[.7,1.,.25],True))
        self.assertTrue(r['response_resolved'])
        self.assertAlmostEqual(r['itae'],1.,places=6)
        self.assertAlmostEqual(r['settling_time_2pct'],-np.log(.02),places=7)
        self.assertTrue(r['original_strict_accepted'])

    def test_stable_overshoot_miss_receives_complete_metrics(self):
        r=diagnostic(('test',0,[.3,2.,0.,1.],[1.,1.,0.],False))
        self.assertTrue(r['stable'])
        self.assertTrue(r['response_resolved'])
        self.assertGreater(r['overshoot_pct'],10)
        self.assertGreater(r['itae'],0)
        self.assertGreater(r['settling_time_2pct'],0)
        self.assertFalse(r['original_strict_accepted'])

    def test_unstable_response_is_not_given_converged_metrics(self):
        r=diagnostic(('test',0,[.2,3.,1.,1.],[100.,100.,0.],False))
        self.assertEqual(r['reason'],'unstable')
        self.assertFalse(r['response_resolved'])


if __name__=='__main__':unittest.main()

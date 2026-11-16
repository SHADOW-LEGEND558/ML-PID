import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from large_experiment import make_split,summarize


class LargeSplitTests(unittest.TestCase):
    def test_exact_disjoint_reproducible_stratified_split(self):
        frame=pd.DataFrame({'zeta':np.arange(20000)+1,'wn':2.,'tau':np.where(np.arange(20000)%10==0,0.,.2),'K':1.})
        split=make_split(frame)
        self.assertEqual([len(split[k]) for k in ['train','validation','test']],[15000,2500,2500])
        self.assertEqual(len(set(split['train'])|set(split['validation'])|set(split['test'])),20000)
        self.assertFalse(set(split['train'])&set(split['test']))
        self.assertEqual(split,make_split(frame))
        self.assertEqual(int(frame.loc[split['test'],'tau'].eq(0).sum()),250)

    def test_cost_ratios_keep_failed_references_out_of_comparisons(self):
        frame=pd.DataFrame({'accepted':[True,True,False],'stable':[True]*3,
            'within_bounds':[True]*3,'reference_accepted':[True,False,True],
            'objective_ratio':[1.2,np.nan,np.nan],'itae_ratio':[1.1,np.nan,np.nan]})
        for gain in ['Kp','Ki','Kd']:
            frame['target_'+gain]=[1.,2.,3.]
            frame['predicted_'+gain]=[1.,2.,3.]
        result=summarize(frame)
        self.assertEqual(result['passes'],2)
        self.assertEqual(result['ratio_pairs'],1)
        self.assertAlmostEqual(result['median_ITAE_ratio'],1.1)
        self.assertEqual(result['r2'],1.)


if __name__=='__main__':unittest.main()

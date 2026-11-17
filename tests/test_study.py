import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from study import validate_split,summarize,model_fingerprints,specs


class StudyTests(unittest.TestCase):
    def test_model_fingerprint_is_stable_and_detects_changed_weights(self):
        from sklearn.base import clone
        features=np.arange(40,dtype=float).reshape(8,5)+1
        targets=np.column_stack([features[:,0],features[:,0]*2,features[:,0]*.1])
        model=clone(specs()['linear'][0][1]).fit(features,targets)
        before=model_fingerprints({'linear':model})
        model.predict(features)
        self.assertEqual(before,model_fingerprints({'linear':model}))
        model.estimator_.steps[-1][1].coef_[0,0]+=1
        self.assertNotEqual(before,model_fingerprints({'linear':model}))

    def test_exact_disjoint_final_split(self):
        protocol={'train':list(range(7000)),'validation':list(range(7000,8500)),'test':list(range(8500,10000))}
        self.assertEqual(list(map(len,validate_split(protocol))),[7000,1500,1500])
        protocol['test'][0]=0
        with self.assertRaises(ValueError):validate_split(protocol)

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

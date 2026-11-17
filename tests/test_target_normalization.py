import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyRegressor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from target_normalization import EmpiricalTargetRegressor


class TargetNormalizationTests(unittest.TestCase):
    def test_fit_and_inverse_prediction_use_only_empirical_features(self):
        X=np.array([[2,4,1,1,.2],[1,3,2,2,.1]],dtype=float)
        y=np.array([[3,4,5],[1.5,4,1.25]])
        model=EmpiricalTargetRegressor(DummyRegressor(),normalized=True).fit(X,y)
        np.testing.assert_allclose(model.predict(X),y)
        raw=EmpiricalTargetRegressor(DummyRegressor(),normalized=False).fit(X,y)
        np.testing.assert_allclose(raw.predict(X),np.tile(y.mean(axis=0),(2,1)))
        with self.assertRaises(ValueError):model.factors(np.zeros((2,5)))


if __name__=='__main__':unittest.main()

"""Empirical target scaling and training-only control evaluation helpers."""
from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np
from benchmark import assess_controller, TARGETS
from data_gen import Plant


class EmpiricalTargetRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, normalized=False):
        self.estimator=estimator
        self.normalized=normalized

    @staticmethod
    def factors(X):
        # Only the five model inputs: rise time at index 0, final value at 3.
        array=np.asarray(X,dtype=float)
        times=array[:,0];gain=array[:,3]
        if np.any(times<=0) or np.any(gain<=0):
            raise ValueError('Target normalization requires positive rise time and final output.')
        return np.column_stack([gain,gain*times,gain/times])

    def fit(self,X,y):
        self.estimator_=clone(self.estimator)
        self.estimator_.fit(X,np.asarray(y)*self.factors(X) if self.normalized else y)
        return self

    def predict(self,X):
        prediction=self.estimator_.predict(X)
        return prediction/self.factors(X) if self.normalized else prediction


def evaluate_task(task):
    row_id,plant_parameters,prediction,weight=task
    result=assess_controller(Plant(*plant_parameters),np.asarray(prediction),weight)
    return {'source_row':int(row_id),**result}

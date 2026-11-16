"""Random forest and XGBoost empirical PID baselines; no substitute estimator."""
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from Baselines import Model, Dataset
from benchmark import run_experiment

# 2. Random Forest Regressor Model Baseline
class RandomForestModel(Model):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    @property
    def name(self) -> str:
        depth_str = f", max_depth={self.max_depth}" if self.max_depth else ""
        return f"Random Forest Regressor (trees={self.n_estimators}{depth_str})"

    def fit(self, dataset: Dataset) -> None:
        self._model.fit(dataset.features, dataset.targets)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.features)


# 3. XGBoost Multi-Output Regressor Model Baseline
class XGBoostModel(Model):
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.08, max_depth: int = 6, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        if not HAS_XGBOOST:
            raise ImportError('XGBoost is required. Install it with: python -m pip install xgboost')
        base_estimator = xgb.XGBRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state, n_jobs=1)
        self._model = MultiOutputRegressor(base_estimator)

    @property
    def name(self) -> str:
        tag = "XGBoost"
        return f"{tag} Regressor (trees={self.n_estimators}, lr={self.learning_rate}, depth={self.max_depth})"

    def fit(self, dataset: Dataset) -> None:
        self._model.fit(dataset.features, dataset.targets)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.features)



def model_specs(seed):
    return {
        'random_forest': (
            RandomForestModel(random_state=seed)._model,
            {'model__max_depth': [6, 12, None], 'model__min_samples_leaf': [1, 3]}, False, False),
        'xgboost': (
            XGBoostModel(random_state=seed)._model,
            {'model__estimator__max_depth': [3, 6],
             'model__estimator__learning_rate': [0.03, 0.08]}, False, False),
    }


if __name__ == '__main__':
    run_experiment(model_specs, 'trees')

"""Linear, ridge, and polynomial-ridge empirical PID baselines."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from Baselines import Model, Dataset
from benchmark import run_experiment

# 2. Linear Regression Model Baseline
class LinearRegressionModel(Model):
    def __init__(self):
        self._model = LinearRegression()

    @property
    def name(self) -> str:
        return "Multiple Linear Regression"

    def fit(self, dataset: Dataset) -> None:
        self._model.fit(dataset.features, dataset.targets)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.features)


# 3. Ridge Regression Model Baseline
class RidgeRegressionModel(Model):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model = Ridge(alpha=self.alpha)

    @property
    def name(self) -> str:
        return f"Ridge Regression (alpha={self.alpha})"

    def fit(self, dataset: Dataset) -> None:
        self._model.fit(dataset.features, dataset.targets)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.features)


# 4. Polynomial Ridge Regression Model Baseline
class PolynomialRegressionModel(Model):
    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self._model = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=self.alpha)
        )

    @property
    def name(self) -> str:
        return f"Polynomial Ridge Regression (degree={self.degree})"

    def fit(self, dataset: Dataset) -> None:
        self._model.fit(dataset.features, dataset.targets)

    def predict(self, dataset: Dataset) -> np.ndarray:
        return self._model.predict(dataset.features)



def model_specs(seed):
    return {
        'linear': (LinearRegression(), {}, False, True),
        'ridge': (Ridge(), {'model__alpha': [0.1, 1.0, 10.0, 100.0]}, False, True),
        'polynomial_ridge': (Ridge(), {'model__alpha': [0.1, 1.0, 10.0, 100.0]}, True, True),
    }


if __name__ == '__main__':
    run_experiment(model_specs, 'regression')

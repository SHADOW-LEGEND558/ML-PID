from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Dataset:
    features: np.ndarray
    targets: np.ndarray

@dataclass
class EvaluationResult:
    model_name: str
    mse: float
    mae: float
    r2_score: float = 0.0


# 2. Data Conditioner Abstract Interface
class DataConditioner(ABC):
    
    @abstractmethod
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit scaler/imputer parameters and transform data."""
        pass

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """Transform data using pre-fitted parameters."""
        pass

# 3. Polymorphic Model Base Interface
class Model(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string."""
        pass

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """Train the model on given dataset."""
        pass

    @abstractmethod
    def predict(self, dataset: Dataset) -> np.ndarray:
        """Generate target predictions for given dataset."""
        pass

# 4. Model Evaluator
class EvaluateModel:
    def evaluate(self, model: Model, test_dataset: Dataset) -> EvaluationResult:
        predictions = model.predict(test_dataset)
        targets = test_dataset.targets

        mse = float(np.mean((predictions - targets) ** 2))
        mae = float(np.mean(np.abs(predictions - targets)))
        
        # R2 score calculation
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        return EvaluationResult(
            model_name=model.name,
            mse=mse,
            mae=mae,
            r2_score=r2
        )


# 5. Benchmark Orchestrator (Runner)
class BenchmarkRunner:
    def __init__(self, conditioner: DataConditioner):
        self.conditioner = conditioner
        self.models: list[Model] = []
        self.evaluator = EvaluateModel()

    def add_model(self, model: Model) -> None:
        """Register any model implementing the Model interface."""
        self.models.append(model)

    def run_benchmark(self, raw_train: Dataset, raw_test: Dataset) -> list[EvaluationResult]:
        """Runs the complete conditioning, training, and evaluation pipeline across all models."""
        # 1. Condition Data
        clean_train = self.conditioner.fit_transform(raw_train)
        clean_test = self.conditioner.transform(raw_test)

        results = []
        # 2. Iterate polymorphically through models
        for model in self.models:
            print(f"Training {model.name}...")
            model.fit(clean_train)
            res = self.evaluator.evaluate(model, clean_test)
            results.append(res)

        return results
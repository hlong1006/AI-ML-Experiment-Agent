"""
Base Trainer — abstract class, tất cả trainers kế thừa từ đây.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from agent.config import settings
from agent.schemas import Experiment, ExperimentMetrics, TaskType
from ml.evaluators.metrics import compute_metrics


class BaseTrainer(ABC):
    """
    Abstract base trainer.
    Subclasses implement _build_model() and optionally override fit().
    """

    model_name: str = "base"

    def __init__(self, params: Dict[str, Any], task_type: TaskType):
        self.params = params
        self.task_type = task_type
        self.model: Any = None
        self._is_fitted: bool = False

    @abstractmethod
    def _build_model(self) -> Any:
        """Instantiate and return the sklearn-compatible estimator."""
        ...

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseTrainer":
        """Train the model. Override for early-stopping support."""
        self.model = self._build_model()
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        self._check_fitted()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ExperimentMetrics:
        """Compute metrics on the given split."""
        self._check_fitted()
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return compute_metrics(y, y_pred, y_prob, self.task_type)

    def save(self, path: Optional[Path] = None, experiment_id: str = "") -> str:
        """Persist the trained model to disk."""
        self._check_fitted()
        fname = f"{self.model_name}_{experiment_id or 'model'}.pkl"
        save_path = path or (settings.MODELS_DIR / fname)
        joblib.dump(self.model, save_path)
        return str(save_path)

    @classmethod
    def load(cls, path: Path) -> Any:
        """Load a persisted model from disk."""
        return joblib.load(path)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")

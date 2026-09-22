"""
LightGBM Trainer — fast gradient boosting optimized for tabular data.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np

from agent.schemas import TaskType
from ml.trainers.base_trainer import BaseTrainer


class LightGBMTrainer(BaseTrainer):
    model_name = "lightgbm"

    def __init__(self, params: Dict[str, Any], task_type: TaskType):
        defaults = {
            "n_estimators": 300,
            "max_depth": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        if task_type == TaskType.BINARY_CLASSIFICATION:
            defaults["objective"] = "binary"
            defaults["metric"] = "binary_logloss"
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            defaults["objective"] = "multiclass"
            defaults["metric"] = "multi_logloss"
        elif task_type == TaskType.REGRESSION:
            defaults["objective"] = "regression"
            defaults["metric"] = "rmse"

        defaults.update(params)
        super().__init__(defaults, task_type)

    def _build_model(self):
        if self.task_type == TaskType.REGRESSION:
            return lgb.LGBMRegressor(**self.params)
        return lgb.LGBMClassifier(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMTrainer":
        self.model = self._build_model()
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_X"] = X_val
            fit_kwargs["eval_y"] = y_val
            fit_kwargs["callbacks"] = callbacks
        self.model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True
        return self

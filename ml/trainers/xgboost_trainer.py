"""
XGBoost Trainer — supports early stopping with eval_set.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb

from agent.schemas import TaskType
from ml.trainers.base_trainer import BaseTrainer


class XGBoostTrainer(BaseTrainer):
    model_name = "xgboost"

    def __init__(self, params: Dict[str, Any], task_type: TaskType):
        defaults = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": -1,
        }
        # Task-specific objective
        if task_type == TaskType.BINARY_CLASSIFICATION:
            defaults["objective"] = "binary:logistic"
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            defaults["objective"] = "multi:softprob"
        elif task_type == TaskType.REGRESSION:
            defaults["objective"] = "reg:squarederror"
            defaults["eval_metric"] = "rmse"

        defaults.update(params)
        super().__init__(defaults, task_type)

    def _build_model(self):
        # Remove keys not accepted by XGBClassifier
        safe_params = {k: v for k, v in self.params.items()
                       if k != "use_label_encoder"}
        if self.task_type == TaskType.REGRESSION:
            return xgb.XGBRegressor(**safe_params)
        return xgb.XGBClassifier(**safe_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostTrainer":
        self.model = self._build_model()
        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        self.model.fit(X_train, y_train, **fit_kwargs)
        self._is_fitted = True
        return self

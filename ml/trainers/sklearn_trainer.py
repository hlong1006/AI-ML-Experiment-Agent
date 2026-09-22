"""
Sklearn Trainers — Logistic Regression & Random Forest.
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from agent.schemas import TaskType
from ml.trainers.base_trainer import BaseTrainer


class LogisticRegressionTrainer(BaseTrainer):
    model_name = "logistic_regression"

    def __init__(self, params: Dict[str, Any], task_type: TaskType):
        defaults = {
            "max_iter": 1000,
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": None,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(defaults, task_type)

    def _build_model(self) -> LogisticRegression:
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            self.params["multi_class"] = "multinomial"
            self.params["solver"] = "lbfgs"
        return LogisticRegression(**self.params)


class RandomForestTrainer(BaseTrainer):
    model_name = "random_forest"

    def __init__(self, params: Dict[str, Any], task_type: TaskType):
        defaults = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight": None,
            "n_jobs": -1,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(defaults, task_type)

    def _build_model(self):
        if self.task_type == TaskType.REGRESSION:
            return RandomForestRegressor(**{
                k: v for k, v in self.params.items()
                if k not in ("class_weight",)
            })
        return RandomForestClassifier(**self.params)

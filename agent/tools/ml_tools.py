"""
ML Tools — Sprint v1-ml-tools
@tool functions: preprocess_data, train_model, evaluate_model, save_model,
                 get_experiment_history, compare_experiments.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from langchain_core.tools import tool

from agent.config import settings
from agent.schemas import (
    Experiment,
    ExperimentMetrics,
    ExperimentStatus,
    ModelName,
    TaskType,
)
from agent.tools.data_tools import _session_df, _session_info
from ml.evaluators.metrics import (
    compute_metrics,
    get_confusion_matrix,
    get_full_classification_report,
)
from ml.preprocessors.preprocessor import DataPreprocessor
from ml.trainers.base_trainer import BaseTrainer
from ml.trainers.lgbm_trainer import LightGBMTrainer
from ml.trainers.sklearn_trainer import LogisticRegressionTrainer, RandomForestTrainer
from ml.trainers.xgboost_trainer import XGBoostTrainer


# ── Session-scoped state ──────────────────────────────────────────────────────
_preprocessor: Optional[DataPreprocessor] = None
_splits: Optional[Dict[str, np.ndarray]] = None
_trained_models: Dict[str, BaseTrainer] = {}
_experiment_history: List[Experiment] = []


def _get_trainer(model_name: str, params: Dict[str, Any], task_type: TaskType) -> BaseTrainer:
    """Factory — returns the correct trainer for the given model name."""
    mapping = {
        ModelName.LOGISTIC_REGRESSION: LogisticRegressionTrainer,
        ModelName.RANDOM_FOREST: RandomForestTrainer,
        ModelName.XGBOOST: XGBoostTrainer,
        ModelName.LIGHTGBM: LightGBMTrainer,
    }
    trainer_cls = mapping.get(model_name)  # type: ignore
    if trainer_cls is None:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(mapping.keys())}")
    return trainer_cls(params, task_type)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def preprocess_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
    scale_numerical: bool = True,
    random_state: int = 42,
) -> str:
    """
    Preprocess the loaded dataset: impute missing values, encode categoricals,
    scale numericals, and split into train/val/test sets.

    Args:
        test_size: Fraction for test split (default 0.15).
        val_size: Fraction for validation split (default 0.15).
        scale_numerical: Whether to StandardScale numerical features.
        random_state: Random seed for reproducibility.

    Returns:
        JSON string with split sizes and feature count.
    """
    global _preprocessor, _splits

    # Import current session state from data_tools
    from agent.tools import data_tools as dt
    df = dt._session_df
    info = dt._session_info

    if df is None or info is None:
        return json.dumps({"error": "No dataset loaded. Call load_dataset() first."})

    _preprocessor = DataPreprocessor(info)
    X_train, X_val, X_test, y_train, y_val, y_test = _preprocessor.fit_transform(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    _splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    # Save preprocessor
    prep_path = _preprocessor.save()

    return json.dumps({
        "status": "preprocessed",
        "n_features": X_train.shape[1],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "preprocessor_path": prep_path,
        "feature_names": _preprocessor.feature_names_out[:10],  # first 10 for brevity
    })


@tool
def train_model(
    model: str,
    params: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
) -> str:
    """
    Train a machine learning model on the preprocessed data.

    Args:
        model: Model name. One of: 'logistic_regression', 'random_forest',
               'xgboost', 'lightgbm'.
        params: Dictionary of model hyperparameters (optional).
        experiment_id: Optional ID to track this experiment.

    Returns:
        JSON string with experiment ID, metrics, and training time.
    """
    global _trained_models, _experiment_history

    if _splits is None:
        return json.dumps({"error": "Data not preprocessed. Call preprocess_data() first."})

    from agent.tools import data_tools as dt
    info = dt._session_info
    if info is None:
        return json.dumps({"error": "No dataset info. Call load_dataset() first."})

    params = params or {}
    exp_id = experiment_id or f"EXP_{uuid4().hex[:6].upper()}"
    exp_num = len(_experiment_history) + 1

    exp = Experiment(
        experiment_id=exp_id,
        experiment_number=exp_num,
        model_name=model,  # type: ignore
        parameters=params,
        status=ExperimentStatus.RUNNING,
    )

    start_time = time.time()
    try:
        trainer = _get_trainer(model, params, info.task_type)
        trainer.fit(
            _splits["X_train"], _splits["y_train"],
            _splits["X_val"],   _splits["y_val"],
        )
        elapsed = round(time.time() - start_time, 2)

        # Evaluate on validation set
        val_metrics = trainer.evaluate(_splits["X_val"], _splits["y_val"])

        exp.metrics = val_metrics
        exp.training_time = elapsed
        exp.status = ExperimentStatus.COMPLETED
        exp.features_used = getattr(_preprocessor, "feature_names_out", [])

        _trained_models[exp_id] = trainer
        _experiment_history.append(exp)

        return json.dumps({
            "experiment_id": exp_id,
            "experiment_number": exp_num,
            "model": model,
            "params": params,
            "val_metrics": {
                "accuracy": val_metrics.accuracy,
                "precision": val_metrics.precision,
                "recall": val_metrics.recall,
                "f1": val_metrics.f1,
                "roc_auc": val_metrics.roc_auc,
            },
            "training_time_seconds": elapsed,
            "status": "completed",
        })

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        exp.status = ExperimentStatus.FAILED
        exp.failure_reason = str(e)
        _experiment_history.append(exp)
        return json.dumps({
            "experiment_id": exp_id,
            "model": model,
            "status": "failed",
            "error": str(e),
            "training_time_seconds": elapsed,
        })


@tool
def evaluate_model(experiment_id: str, split: str = "test") -> str:
    """
    Evaluate a trained model on val or test split.

    Args:
        experiment_id: The experiment ID returned by train_model().
        split: 'val' or 'test' (default 'test').

    Returns:
        JSON string with detailed metrics including confusion matrix.
    """
    if experiment_id not in _trained_models:
        return json.dumps({"error": f"Model {experiment_id} not found. Train it first."})
    if _splits is None:
        return json.dumps({"error": "No preprocessed data available."})

    from agent.tools import data_tools as dt
    info = dt._session_info

    trainer = _trained_models[experiment_id]
    X = _splits[f"X_{split}"]
    y = _splits[f"y_{split}"]

    y_pred = trainer.predict(X)
    y_prob = trainer.predict_proba(X)
    metrics = compute_metrics(y, y_pred, y_prob, info.task_type)
    cm = get_confusion_matrix(y, y_pred)
    report = get_full_classification_report(y, y_pred)

    return json.dumps({
        "experiment_id": experiment_id,
        "split": split,
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
        },
        "confusion_matrix": cm["confusion_matrix"],
        "classification_report": report,
    }, default=str)


@tool
def save_model(experiment_id: str) -> str:
    """
    Persist a trained model to disk.

    Args:
        experiment_id: The experiment ID of the model to save.

    Returns:
        JSON string with the saved model path.
    """
    if experiment_id not in _trained_models:
        return json.dumps({"error": f"Model {experiment_id} not found."})

    trainer = _trained_models[experiment_id]
    path = trainer.save(experiment_id=experiment_id)

    # Update experiment record
    for exp in _experiment_history:
        if exp.experiment_id == experiment_id:
            exp.model_artifact_path = path
            break

    return json.dumps({"status": "saved", "path": path, "experiment_id": experiment_id})


@tool
def get_experiment_history() -> str:
    """
    Return the full history of all experiments run in this session.

    Returns:
        JSON string listing all experiments with metrics and status.
    """
    if not _experiment_history:
        return json.dumps({"experiments": [], "message": "No experiments run yet."})

    history = []
    for exp in _experiment_history:
        history.append({
            "experiment_id": exp.experiment_id,
            "experiment_number": exp.experiment_number,
            "model": exp.model_name,
            "params": exp.parameters,
            "val_f1": exp.metrics.f1,
            "val_accuracy": exp.metrics.accuracy,
            "val_precision": exp.metrics.precision,
            "val_recall": exp.metrics.recall,
            "val_roc_auc": exp.metrics.roc_auc,
            "training_time": exp.training_time,
            "status": exp.status,
        })

    # Sort by F1 descending
    history.sort(key=lambda x: x["val_f1"], reverse=True)
    return json.dumps({"experiments": history, "total": len(history)})


@tool
def compare_experiments() -> str:
    """
    Compare all completed experiments and identify the best model.

    Returns:
        JSON string with ranked experiments and the best model recommendation.
    """
    completed = [e for e in _experiment_history if e.status == ExperimentStatus.COMPLETED]
    if not completed:
        return json.dumps({"error": "No completed experiments to compare."})

    ranked = sorted(completed, key=lambda e: e.metrics.f1, reverse=True)
    best = ranked[0]

    rows = []
    for exp in ranked:
        rows.append({
            "rank": ranked.index(exp) + 1,
            "experiment_id": exp.experiment_id,
            "model": exp.model_name,
            "val_f1": exp.metrics.f1,
            "val_accuracy": exp.metrics.accuracy,
            "val_roc_auc": exp.metrics.roc_auc,
            "training_time": exp.training_time,
        })

    return json.dumps({
        "best_model": best.model_name,
        "best_experiment_id": best.experiment_id,
        "best_f1": best.metrics.f1,
        "best_params": best.parameters,
        "ranking": rows,
    })


# ── Export ────────────────────────────────────────────────────────────────────
ML_TOOLS = [
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    get_experiment_history,
    compare_experiments,
]

__all__ = ["ML_TOOLS", "_experiment_history", "_trained_models", "_splits", "_preprocessor"]

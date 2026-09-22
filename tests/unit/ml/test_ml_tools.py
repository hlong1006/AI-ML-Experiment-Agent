"""Unit tests for ML tools."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent.tools.data_tools import load_dataset
from agent.tools.ml_tools import (
    preprocess_data,
    train_model,
    evaluate_model,
    save_model,
    get_experiment_history,
    compare_experiments,
)

SAMPLE_CSV = Path(__file__).parent / "sample_ml.csv"


@pytest.fixture(scope="module", autouse=True)
def setup_dataset():
    """Load a sample dataset into session before tests."""
    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=8, n_informative=5,
                                random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    df["target"] = y.astype(str)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)

    # Load into session
    load_dataset.invoke({"path": str(SAMPLE_CSV), "target_column": "target"})
    preprocess_data.invoke({})

    yield

    if SAMPLE_CSV.exists():
        SAMPLE_CSV.unlink()


def test_train_logistic_regression():
    result = json.loads(train_model.invoke({"model": "logistic_regression"}))
    assert result["status"] == "completed"
    assert "val_metrics" in result
    assert result["val_metrics"]["f1"] > 0


def test_train_random_forest():
    result = json.loads(train_model.invoke({
        "model": "random_forest",
        "params": {"n_estimators": 50},
    }))
    assert result["status"] == "completed"
    assert result["val_metrics"]["f1"] > 0


def test_train_xgboost():
    result = json.loads(train_model.invoke({
        "model": "xgboost",
        "params": {"n_estimators": 50, "max_depth": 3},
    }))
    assert result["status"] == "completed"
    assert result["val_metrics"]["f1"] > 0


def test_train_lightgbm():
    result = json.loads(train_model.invoke({
        "model": "lightgbm",
        "params": {"n_estimators": 50},
    }))
    assert result["status"] == "completed"
    assert result["val_metrics"]["f1"] > 0


def test_evaluate_model():
    history = json.loads(get_experiment_history.invoke({}))
    exp_id = history["experiments"][0]["experiment_id"]
    result = json.loads(evaluate_model.invoke({"experiment_id": exp_id, "split": "test"}))
    assert "metrics" in result
    assert result["split"] == "test"


def test_save_model():
    history = json.loads(get_experiment_history.invoke({}))
    exp_id = history["experiments"][0]["experiment_id"]
    result = json.loads(save_model.invoke({"experiment_id": exp_id}))
    assert result["status"] == "saved"
    assert Path(result["path"]).exists()


def test_compare_experiments():
    result = json.loads(compare_experiments.invoke({}))
    assert "best_model" in result
    assert "ranking" in result
    assert len(result["ranking"]) >= 1

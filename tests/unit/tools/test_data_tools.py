"""Unit tests for data_tools."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent.tools.data_tools import (
    load_dataset,
    inspect_dataset,
    analyze_missing_values,
    analyze_target_distribution,
    get_dataset_summary,
)


SAMPLE_CSV = Path(__file__).parent / "sample_churn.csv"


@pytest.fixture(scope="module", autouse=True)
def create_sample_csv():
    """Create a tiny sample dataset for testing."""
    data = {
        "tenure": [1, 12, 24, 36, 48, 60, 6, 18, 30, 42],
        "MonthlyCharges": [29.9, 55.0, 70.0, 90.0, None, 45.0, 35.0, 60.0, 80.0, 95.0],
        "TotalCharges": [29.9, 660.0, 1680.0, 3240.0, None, 2700.0, 210.0, 1080.0, 2400.0, 3990.0],
        "Contract": ["Month-to-month","One year","Two year","One year",
                     "Month-to-month","Two year","Month-to-month","One year",
                     "Two year","One year"],
        "Churn": ["Yes","No","No","No","Yes","No","Yes","No","No","No"],
    }
    df = pd.DataFrame(data)
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)
    yield
    # cleanup
    if SAMPLE_CSV.exists():
        SAMPLE_CSV.unlink()


def test_load_dataset():
    result = json.loads(load_dataset.invoke({
        "path": str(SAMPLE_CSV),
        "target_column": "Churn",
    }))
    assert result["status"] == "loaded"
    assert result["rows"] == 10
    assert result["target_column"] == "Churn"
    assert result["task_type"] == "binary_classification"
    assert "MonthlyCharges" in result["missing_values"]


def test_inspect_dataset():
    result = json.loads(inspect_dataset.invoke({}))
    assert result["shape"] == [10, 5]
    assert "feature_details" in result
    assert len(result["feature_details"]) == 5


def test_analyze_missing_values():
    result = json.loads(analyze_missing_values.invoke({}))
    assert result["columns_with_missing"] >= 1
    col_names = [d["column"] for d in result["details"]]
    assert "MonthlyCharges" in col_names


def test_analyze_target_distribution():
    result = json.loads(analyze_target_distribution.invoke({}))
    assert "class_distribution" in result
    assert "Yes" in result["class_distribution"] or "No" in result["class_distribution"]
    assert "recommendations" in result


def test_get_dataset_summary():
    result = json.loads(get_dataset_summary.invoke({}))
    assert result["rows"] == 10
    assert result["target"] == "Churn"

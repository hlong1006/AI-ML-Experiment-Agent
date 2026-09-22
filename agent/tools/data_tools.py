"""
Data Tools — Sprint v1-data-tools
Tất cả @tool functions liên quan đến dataset: load, inspect, EDA, visualize.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from agent.config import settings
from agent.schemas import DatasetInfo, FeatureInfo, TaskType


# ── Internal state (session-scoped) ──────────────────────────────────────────
_session_df: Optional[pd.DataFrame] = None
_session_info: Optional[DatasetInfo] = None


def _get_df() -> pd.DataFrame:
    if _session_df is None:
        raise RuntimeError("No dataset loaded. Call load_dataset() first.")
    return _session_df


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def load_dataset(path: str, target_column: str, task_description: str = "") -> str:
    """
    Load a CSV dataset from the given path.

    Args:
        path: Absolute or relative path to the CSV file.
        target_column: Name of the target/label column.
        task_description: Natural language description of the ML task.

    Returns:
        JSON string with basic dataset info.
    """
    global _session_df, _session_info

    file_path = Path(path)
    if not file_path.exists():
        # Try under data/raw/
        alt = settings.DATA_DIR / "raw" / path
        if alt.exists():
            file_path = alt
        else:
            return json.dumps({"error": f"File not found: {path}"})

    df = pd.read_csv(file_path)
    _session_df = df

    # Detect task type
    n_unique_target = df[target_column].nunique()
    if n_unique_target == 2:
        task_type = TaskType.BINARY_CLASSIFICATION
    elif n_unique_target <= 20:
        task_type = TaskType.MULTICLASS_CLASSIFICATION
    else:
        task_type = TaskType.REGRESSION

    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_column in numerical:
        numerical.remove(target_column)
    if target_column in categorical:
        categorical.remove(target_column)

    feature_names = [c for c in df.columns if c != target_column]
    class_dist: Dict[str, int] = {}
    if task_type != TaskType.REGRESSION:
        class_dist = df[target_column].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

    missing = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    imbalance_ratio = None
    if task_type == TaskType.BINARY_CLASSIFICATION and len(class_dist) == 2:
        counts = list(class_dist.values())
        imbalance_ratio = round(max(counts) / min(counts), 2)

    _session_info = DatasetInfo(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_column=target_column,
        task_type=task_type,
        feature_names=feature_names,
        numerical_features=numerical,
        categorical_features=categorical,
        class_distribution=class_dist,
        missing_values=missing,
        imbalance_ratio=imbalance_ratio,
        dataset_path=str(file_path),
    )

    result = {
        "status": "loaded",
        "rows": len(df),
        "columns": len(df.columns),
        "target_column": target_column,
        "task_type": task_type.value,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "class_distribution": class_dist,
        "missing_values": missing,
        "imbalance_ratio": imbalance_ratio,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def inspect_dataset() -> str:
    """
    Return detailed statistics about the currently loaded dataset.
    Includes dtypes, describe(), and sample rows.

    Returns:
        JSON string with detailed statistics.
    """
    df = _get_df()

    describe = df.describe(include="all").fillna("").to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    sample = df.head(3).fillna("").to_dict(orient="records")

    feature_details: List[Dict] = []
    for col in df.columns:
        feature_details.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "n_unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist(),
        })

    result = {
        "shape": list(df.shape),
        "dtypes": dtypes,
        "feature_details": feature_details,
        "sample_rows": sample,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def analyze_missing_values() -> str:
    """
    Analyze missing values in the dataset.
    Returns per-column missing count, percentage, and recommended strategy.

    Returns:
        JSON string with missing value analysis.
    """
    df = _get_df()
    total = len(df)
    report: List[Dict] = []

    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        if n_miss == 0:
            continue
        pct = round(n_miss / total * 100, 2)
        dtype = str(df[col].dtype)

        if pct > 50:
            strategy = "drop_column"
        elif "float" in dtype or "int" in dtype:
            strategy = "fill_median"
        else:
            strategy = "fill_mode"

        report.append({
            "column": col,
            "n_missing": n_miss,
            "pct_missing": pct,
            "dtype": dtype,
            "recommended_strategy": strategy,
        })

    return json.dumps({
        "total_rows": total,
        "columns_with_missing": len(report),
        "details": report,
    }, ensure_ascii=False)


@tool
def analyze_target_distribution() -> str:
    """
    Analyze the distribution of the target column.
    Detects imbalance and recommends handling strategies.

    Returns:
        JSON string with class distribution and recommendations.
    """
    if _session_info is None:
        return json.dumps({"error": "No dataset loaded."})

    df = _get_df()
    target = _session_info.target_column
    dist = _session_info.class_distribution
    total = _session_info.n_rows

    recommendations: List[str] = []

    if _session_info.task_type == TaskType.BINARY_CLASSIFICATION:
        ratio = _session_info.imbalance_ratio or 1.0
        if ratio > 3:
            recommendations.extend([
                "Use class_weight='balanced' in tree models",
                "Consider SMOTE oversampling",
                "Use F1-score as primary metric (not accuracy)",
                "Try threshold optimization after training",
            ])
        if ratio > 10:
            recommendations.append("Severe imbalance — consider SMOTEENN or ADASYN")

    pct_dist = {k: round(v / total * 100, 2) for k, v in dist.items()}

    return json.dumps({
        "target_column": target,
        "class_distribution": dist,
        "pct_distribution": pct_dist,
        "imbalance_ratio": _session_info.imbalance_ratio,
        "recommendations": recommendations,
    }, ensure_ascii=False)


@tool
def create_visualization(chart_type: str, columns: Optional[List[str]] = None) -> str:
    """
    Create and save a visualization chart.

    Args:
        chart_type: One of: 'correlation_heatmap', 'target_distribution',
                    'missing_values', 'feature_distributions', 'boxplots'
        columns: Optional list of columns to include (default: all numerical)

    Returns:
        JSON string with the saved chart path.
    """
    df = _get_df()
    info = _session_info

    save_dir = settings.REPORTS_DIR / "charts"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{chart_type}.png"

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_theme(style="darkgrid")

    try:
        if chart_type == "correlation_heatmap":
            num_cols = info.numerical_features if info else df.select_dtypes(include=np.number).columns.tolist()
            if columns:
                num_cols = [c for c in columns if c in num_cols]
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                        linewidths=0.5, square=True)
            ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

        elif chart_type == "target_distribution":
            target = info.target_column if info else df.columns[-1]
            if df[target].dtype == "object" or df[target].nunique() <= 20:
                counts = df[target].value_counts()
                sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                            palette="viridis")
                for i, v in enumerate(counts.values):
                    ax.text(i, v + 0.5, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=10)
            else:
                sns.histplot(df[target], bins=30, ax=ax, color="#6366f1")
            ax.set_title(f"Target Distribution: {target}", fontsize=14, fontweight="bold")

        elif chart_type == "missing_values":
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if missing.empty:
                ax.text(0.5, 0.5, "No missing values!", transform=ax.transAxes,
                        ha="center", va="center", fontsize=16)
            else:
                sns.barplot(x=missing.values, y=missing.index.tolist(), ax=ax,
                            palette="Reds_r", orient="h")
                ax.set_xlabel("Missing Count")
            ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")

        elif chart_type == "feature_distributions":
            num_cols = info.numerical_features if info else df.select_dtypes(include=np.number).columns.tolist()
            if columns:
                num_cols = [c for c in columns if c in num_cols]
            num_cols = num_cols[:9]  # max 9 subplots
            plt.close(fig)
            n = len(num_cols)
            cols_grid = min(3, n)
            rows_grid = (n + cols_grid - 1) // cols_grid
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(15, 4 * rows_grid))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                sns.histplot(df[col].dropna(), bins=30, ax=axes[i], color="#6366f1", kde=True)
                axes[i].set_title(col, fontsize=11)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold")

        elif chart_type == "boxplots":
            num_cols = info.numerical_features if info else df.select_dtypes(include=np.number).columns.tolist()
            if columns:
                num_cols = [c for c in columns if c in num_cols]
            target = info.target_column if info else None
            if target and df[target].nunique() <= 5:
                col_to_plot = num_cols[0] if num_cols else None
                if col_to_plot:
                    sns.boxplot(data=df, x=target, y=col_to_plot, ax=ax, palette="Set2")
                    ax.set_title(f"{col_to_plot} by {target}", fontsize=14, fontweight="bold")
            else:
                sns.boxplot(data=df[num_cols], ax=ax, palette="Set2", orient="h")
                ax.set_title("Feature Boxplots", fontsize=14, fontweight="bold")
        else:
            return json.dumps({"error": f"Unknown chart_type: {chart_type}"})

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return json.dumps({
            "status": "created",
            "chart_type": chart_type,
            "path": str(save_path),
        })

    except Exception as e:
        plt.close("all")
        return json.dumps({"error": str(e)})


@tool
def get_dataset_summary() -> str:
    """
    Return a concise summary of the loaded dataset for the agent to reason about.

    Returns:
        JSON string with key dataset characteristics.
    """
    if _session_info is None:
        return json.dumps({"error": "No dataset loaded."})

    info = _session_info
    return json.dumps({
        "rows": info.n_rows,
        "features": len(info.feature_names),
        "target": info.target_column,
        "task_type": info.task_type,
        "numerical_features": info.numerical_features,
        "categorical_features": info.categorical_features,
        "missing_values": info.missing_values,
        "class_distribution": info.class_distribution,
        "imbalance_ratio": info.imbalance_ratio,
    }, ensure_ascii=False)


# ── Export ───────────────────────────────────────────────────────────────────
DATA_TOOLS = [
    load_dataset,
    inspect_dataset,
    analyze_missing_values,
    analyze_target_distribution,
    create_visualization,
    get_dataset_summary,
]

__all__ = ["DATA_TOOLS", "_session_df", "_session_info"]

"""
Shared Pydantic schemas — dùng toàn bộ project.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class ModelName(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    DECISION_TREE = "decision_tree"
    SVM = "svm"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CriticDecision(str, Enum):
    CONTINUE = "continue"
    TUNE = "tune"
    FINISH = "finish"


# ── Dataset ──────────────────────────────────────────────────────────────────

class FeatureInfo(BaseModel):
    name: str
    dtype: str
    n_missing: int = 0
    n_unique: int = 0
    sample_values: List[Any] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    n_rows: int
    n_cols: int
    target_column: str
    task_type: TaskType
    feature_names: List[str] = Field(default_factory=list)
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    feature_details: List[FeatureInfo] = Field(default_factory=list)
    imbalance_ratio: Optional[float] = None
    dataset_path: str = ""
    processed_path: str = ""


# ── Experiments ──────────────────────────────────────────────────────────────

class ExperimentMetrics(BaseModel):
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    val_loss: Optional[float] = None


class Experiment(BaseModel):
    experiment_id: str = Field(default_factory=lambda: f"EXP_{uuid4().hex[:6].upper()}")
    session_id: str = ""
    experiment_number: int = 0
    model_name: ModelName
    parameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_strategy: str = "standard"
    features_used: List[str] = Field(default_factory=list)
    metrics: ExperimentMetrics = Field(default_factory=ExperimentMetrics)
    training_time: float = 0.0
    status: ExperimentStatus = ExperimentStatus.PENDING
    critic_notes: str = ""
    failure_reason: str = ""
    model_artifact_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PlannedExperiment(BaseModel):
    order: int
    model_name: ModelName
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ExperimentPlan(BaseModel):
    experiments: List[PlannedExperiment] = Field(default_factory=list)
    optimization_goal: str = "maximize f1"
    target_metric_value: float = 0.0
    max_experiments: int = 10
    time_budget_minutes: int = 30


# ── Agent Memory ─────────────────────────────────────────────────────────────

class AgentMemory(BaseModel):
    dataset_context: Optional[DatasetInfo] = None
    best_model_name: str = ""
    best_metric_value: float = 0.0
    best_experiment_id: str = ""
    failed_branches: List[str] = Field(default_factory=list)
    proven_strategies: List[str] = Field(default_factory=list)
    current_hypothesis: str = ""
    best_hyperparameters: Dict[str, Any] = Field(default_factory=dict)


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Central shared state flowing through the LangGraph graph."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_path: str = ""
    task_description: str = ""
    target_column: str = ""
    primary_metric: str = "f1"
    target_metric_value: float = 0.0

    dataset_info: Optional[DatasetInfo] = None
    experiment_plan: Optional[ExperimentPlan] = None
    experiment_history: List[Experiment] = Field(default_factory=list)
    current_experiment: Optional[Experiment] = None
    memory: AgentMemory = Field(default_factory=AgentMemory)

    # Control flow
    next_action: str = ""
    critic_decision: CriticDecision = CriticDecision.CONTINUE
    critic_reasoning: str = ""
    experiments_done: int = 0
    max_experiments: int = 10
    time_budget_seconds: float = 1800.0
    time_elapsed: float = 0.0

    # Output
    final_report_path: str = ""
    error_message: str = ""

    class Config:
        use_enum_values = True

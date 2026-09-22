"""
Preprocessor — handles missing values, encoding, scaling, and train/val/test split.
Works with the currently loaded dataset from data_tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from agent.config import settings
from agent.schemas import DatasetInfo


class DataPreprocessor:
    """
    Fits and applies a preprocessing pipeline based on DatasetInfo.

    Pipeline steps:
    1. Numerical: impute median → StandardScaler
    2. Categorical: impute mode  → OneHotEncoder (handle_unknown='ignore')
    3. Target: LabelEncoder (for classification)
    """

    def __init__(self, dataset_info: DatasetInfo):
        self.info = dataset_info
        self.pipeline: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names_out: List[str] = []

    def build_pipeline(self, scale_numerical: bool = True) -> ColumnTransformer:
        """Build the ColumnTransformer pipeline."""
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler() if scale_numerical else "passthrough"),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if self.info.numerical_features:
            transformers.append(("num", num_pipe, self.info.numerical_features))
        if self.info.categorical_features:
            transformers.append(("cat", cat_pipe, self.info.categorical_features))

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        return self.pipeline

    def fit_transform(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the pipeline and split into train / val / test.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        target = self.info.target_column
        X = df.drop(columns=[target])
        y = df[target].values

        # Encode target
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        # Build and fit pipeline
        self.build_pipeline()
        X_all = self.pipeline.fit_transform(X)

        # Save feature names
        self.feature_names_out = self._get_feature_names()

        # Split: train / temp → temp → val / test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_all, y_enc, test_size=test_size + val_size,
            random_state=random_state, stratify=y_enc if len(np.unique(y_enc)) > 1 else None,
        )
        rel_val = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - rel_val,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted yet. Call fit_transform() first.")
        target = self.info.target_column
        X = df.drop(columns=[target], errors="ignore")
        return self.pipeline.transform(X)

    def _get_feature_names(self) -> List[str]:
        """Extract feature names after transformation."""
        names: List[str] = []
        if self.pipeline is None:
            return names
        for name, transformer, cols in self.pipeline.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                enc = transformer.named_steps["encoder"]
                try:
                    names.extend(enc.get_feature_names_out(cols).tolist())
                except Exception:
                    names.extend(cols)
        return names

    def save(self, path: Optional[Path] = None) -> str:
        """Save the fitted pipeline to disk."""
        save_path = path or (settings.MODELS_DIR / "preprocessor.pkl")
        joblib.dump({"pipeline": self.pipeline, "label_encoder": self.label_encoder}, save_path)
        return str(save_path)

    @classmethod
    def load(cls, dataset_info: DatasetInfo, path: Path) -> "DataPreprocessor":
        """Load a fitted pipeline from disk."""
        obj = cls(dataset_info)
        data = joblib.load(path)
        obj.pipeline = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        return obj

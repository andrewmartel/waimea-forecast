"""Class-based estimator for wave height forecasting: fit, predict, save, load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from waimea_forecast.config import (
    ARTIFACT_VERSION,
    TARGET_COLUMN,
    VALIDATION_FRACTION,
)
from waimea_forecast.features.engineering import build_features, prepare_supervised


class WaveHeightEstimator:
    """
    Estimator that encapsulates data preparation, feature engineering, and model.

    Fit on raw wide-format DataFrame; predict from raw DataFrame using the same
    feature pipeline. Persists feature state and model in a single artifact.
    """

    def __init__(
        self,
        *,
        validation_fraction: float = VALIDATION_FRACTION,
        random_state: int | None = None,
    ):
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self._feature_state: dict[str, Any] = {}
        self._model: Ridge | None = None
        self._feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "WaveHeightEstimator":
        """
        Fit feature pipeline and model on raw wide-format DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with 'date' and target column.

        Returns
        -------
        self : WaveHeightEstimator
        """
        featurized, feat_state = build_features(df, fitted_medians=None)
        self._feature_state = feat_state

        X_train, y_train, X_val, y_val, feature_cols, state = prepare_supervised(
            featurized,
            feature_state=feat_state,
            validation_fraction=self.validation_fraction,
        )
        self._feature_state.update(state)
        self._feature_columns = state["feature_column_order"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = Ridge(alpha=1.0, random_state=self.random_state)
        model.fit(X_train_scaled, y_train)

        self._scaler = scaler
        self._model = model

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict next-day wave height for each row using fitted pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw wide-format DataFrame (same schema as training).

        Returns
        -------
        np.ndarray
            Predicted wave heights, one per row (after feature alignment).
        """
        if self._model is None:
            raise ValueError("Estimator not fitted; call fit() first.")

        featurized, _ = build_features(
            df,
            fitted_medians=self._feature_state.get("fitted_medians"),
        )
        feature_cols = self._feature_state.get("feature_column_order", self._feature_columns)
        missing = [c for c in feature_cols if c not in featurized.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")

        X = featurized[feature_cols]
        # Impute any remaining NaN with column median from training
        for c in X.columns:
            if X[c].isna().any():
                med = self._feature_state.get("fitted_medians", {}).get(c, X[c].median())
                X = X.assign(**{c: X[c].fillna(med)})
        X = X[feature_cols]
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def save(self, path: str | Path) -> None:
        """Persist fitted estimator (feature state, scaler, model) to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ARTIFACT_VERSION,
            "feature_state": self._feature_state,
            "feature_columns": self._feature_columns,
            "scaler": self._scaler,
            "model": self._model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "WaveHeightEstimator":
        """Load a persisted estimator from path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        payload = joblib.load(path)
        est = cls()
        est._feature_state = payload.get("feature_state", {})
        est._feature_columns = payload.get("feature_columns", [])
        est._scaler = payload["scaler"]
        est._model = payload["model"]
        return est

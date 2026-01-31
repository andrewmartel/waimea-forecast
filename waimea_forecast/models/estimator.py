"""Class-based estimator for wave height forecasting: fit, predict, save, load."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except ImportError:
    pass
try:
    from sklearn.impute import IterativeImputer
except ImportError as e:
    raise ImportError(
        "MICE imputation requires scikit-learn >= 0.21. "
        "Upgrade with: pip install --upgrade scikit-learn"
    ) from e
try:
    from sklearn.impute import KNNImputer
except ImportError:
    KNNImputer = None  # requires sklearn >= 0.22

from waimea_forecast.config import (
    ARTIFACT_VERSION,
    TARGET_COLUMN,
    VALIDATION_FRACTION,
)
from waimea_forecast.features.engineering import build_features, prepare_supervised


def _make_imputer(
    method: Literal["mice", "knn"],
    random_state: int | None,
    n_neighbors: int = 5,
    max_iter: int = 10,
) -> "IterativeImputer | KNNImputer":
    if method == "mice":
        return IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
        )
    if method == "knn":
        if KNNImputer is None:
            raise ImportError(
                "KNN imputation requires scikit-learn >= 0.22. "
                "Upgrade with: pip install --upgrade scikit-learn"
            )
        return KNNImputer(n_neighbors=n_neighbors)
    raise ValueError(f"imputation must be 'mice' or 'knn', got {method!r}")


class WaveHeightEstimator:
    """
    Estimator that encapsulates data preparation, feature engineering, and model.

    Fit on raw wide-format DataFrame; predict from raw DataFrame using the same
    feature pipeline. Uses MICE (IterativeImputer) or KNN imputation fitted on
    X_train only. Persists feature state, imputer, scaler, and model in one artifact.
    """

    def __init__(
        self,
        *,
        validation_fraction: float = VALIDATION_FRACTION,
        random_state: int | None = None,
        imputation: Literal["mice", "knn"] = "mice",
    ):
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.imputation = imputation
        self._feature_state: dict[str, Any] = {}
        self._feature_columns: list[str] = []
        self._imputer: IterativeImputer | KNNImputer | None = None
        self._model: Ridge | None = None

    def fit(self, df: pd.DataFrame) -> "WaveHeightEstimator":
        """
        Fit feature pipeline, imputer (on X_train only), and model on raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with 'date' and target column.

        Returns
        -------
        self : WaveHeightEstimator
        """
        featurized, feat_state = build_features(df, impute=False)
        self._feature_state = feat_state

        X_train, y_train, X_val, y_val, feature_cols, state = prepare_supervised(
            featurized,
            feature_state=feat_state,
            validation_fraction=self.validation_fraction,
            drop_na_rows=False,
        )
        self._feature_state.update(state)
        self._feature_columns = state["feature_column_order"]

        imputer = _make_imputer(self.imputation, self.random_state)
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        # Drop rows that are still NaN after imputation (e.g. all-NaN rows)
        train_ok = ~np.isnan(X_train_imp).any(axis=1)
        val_ok = ~np.isnan(X_val_imp).any(axis=1)
        X_train_imp = np.where(np.isnan(X_train_imp), 0.0, X_train_imp)
        X_val_imp = np.where(np.isnan(X_val_imp), 0.0, X_val_imp)
        X_train_imp = X_train_imp[train_ok]
        y_train = y_train.iloc[np.where(train_ok)[0]]
        X_val_imp = X_val_imp[val_ok]
        y_val = y_val.iloc[np.where(val_ok)[0]]

        # Store training column medians for fallback (e.g. old artifacts, edge cases)
        X_train_kept = X_train.iloc[np.where(train_ok)[0]]
        self._feature_state["fitted_medians"] = {
            c: float(X_train_kept[c].median()) for c in X_train.columns
        }

        self._imputer = imputer
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
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

        featurized, _ = build_features(df, impute=False)
        feature_cols = self._feature_state.get("feature_column_order", self._feature_columns)
        missing = [c for c in feature_cols if c not in featurized.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")

        X = featurized[feature_cols]

        if self._imputer is not None:
            X_imp = self._imputer.transform(X)
            X_imp = np.where(np.isnan(X_imp), 0.0, X_imp)
        else:
            # Backward compat: old artifacts without imputer
            X_imp = X.copy()
            for c in X_imp.columns:
                if X_imp[c].isna().any():
                    med = self._feature_state.get("fitted_medians", {}).get(c, X_imp[c].median())
                    X_imp[c] = X_imp[c].fillna(med)
            X_imp = X_imp.values

        X_scaled = self._scaler.transform(X_imp)
        return self._model.predict(X_scaled)

    def save(self, path: str | Path) -> None:
        """Persist fitted estimator (feature state, imputer, scaler, model) to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ARTIFACT_VERSION,
            "feature_state": self._feature_state,
            "feature_columns": self._feature_columns,
            "imputer": self._imputer,
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
        est._imputer = payload.get("imputer")
        est._scaler = payload["scaler"]
        est._model = payload["model"]
        return est

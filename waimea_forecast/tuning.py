"""Tuning utilities: select max_features by validation performance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from waimea_forecast.features.engineering import build_features, prepare_supervised
from waimea_forecast.metrics import regression_metrics
from waimea_forecast.models.estimator import WaveHeightEstimator


# Default candidates for max_features: try "all" plus several top-K cutoffs
DEFAULT_MAX_FEATURES_CANDIDATES = [10, 20, 30, 40, 50, 60, 80, 100, None]


def select_max_features_cv(
    df: pd.DataFrame,
    horizon_days: int,
    candidates: list[int | None] | None = None,
    validation_fraction: float = 0.2,
    metric: str = "MAE",
) -> tuple[int | None, list[tuple[int | None, float]]]:
    """
    Choose max_features by validation performance (single train/val split).

    Fits the estimator for each candidate (e.g. K=10, 20, ..., None), evaluates
    on the same held-out validation set, and returns the K with best validation
    metric (default: lowest MAE).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'date' and target column.
    horizon_days : int
        Forecast horizon (1, 7, or 30).
    candidates : list of int or None, optional
        max_features values to try (None = use all features).
        Default: [10, 20, 30, 40, 50, 60, 80, 100, None].
    validation_fraction : float, optional
        Fraction of rows for validation (same as estimator default 0.2).
    metric : str, optional
        Metric to minimize: "MAE", "RMSE", or "MAPE". For R² use "MAE" and
        pick by hand, or extend this to support "R2" (maximize).

    Returns
    -------
    best_k : int or None
        The best max_features (None means use all).
    results : list of (k, value)
        For each candidate, (max_features, validation metric value).
    """
    candidates = candidates or DEFAULT_MAX_FEATURES_CANDIDATES
    minimize = metric.upper() in ("MAE", "RMSE", "MAPE")

    featurized, feat_state = build_features(
        df, horizon_days=horizon_days, impute=False
    )
    _, _, X_val, y_val, _, _ = prepare_supervised(
        featurized,
        feat_state,
        validation_fraction=validation_fraction,
        drop_na_rows=False,
        horizon_days=horizon_days,
    )

    results: list[tuple[int | None, float]] = []
    best_val = np.inf if minimize else -np.inf
    best_k: int | None = None

    for k in candidates:
        est = WaveHeightEstimator(
            horizon_days=horizon_days,
            max_features=k,
            validation_fraction=validation_fraction,
        )
        est.fit(df)

        X_val_imp = est._imputer.transform(X_val)
        val_ok = ~np.isnan(X_val_imp).any(axis=1)
        y_val_clean = y_val.iloc[np.where(val_ok)[0]]
        preds = est.predict(df)
        val_preds = np.array([preds[i] for i in y_val_clean.index if i < len(preds)])

        if len(val_preds) != len(y_val_clean):
            val_metric = np.nan
        else:
            reg = regression_metrics(y_val_clean.values, val_preds)
            val_metric = reg.get(metric.upper(), reg["MAE"])

        results.append((k, float(val_metric)))

        if np.isfinite(val_metric):
            if minimize and val_metric < best_val:
                best_val = val_metric
                best_k = k
            elif not minimize and val_metric > best_val:
                best_val = val_metric
                best_k = k

    return best_k, results

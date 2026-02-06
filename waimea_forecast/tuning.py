"""Tune ETS and ARIMA by holdout MAE over a small grid of configs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from waimea_forecast.statistical import fit_ets, fit_arima, forecast_ets, forecast_arima


# ETS: (trend, seasonal, seasonal_periods); seasonal_periods None => non-seasonal
ETS_GRID: list[dict[str, Any]] = [
    {"trend": "add", "seasonal": "add", "seasonal_periods": 7},
    {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
    {"trend": "add", "seasonal": None, "seasonal_periods": None},
    {"trend": None, "seasonal": "add", "seasonal_periods": 7},
    {"trend": None, "seasonal": None, "seasonal_periods": None},
]

# ARIMA: (order, seasonal_order)
ARIMA_GRID: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((1, 0, 1), (0, 0, 0, 0)),
    ((0, 0, 0), (0, 0, 0, 0)),  # mean
    ((1, 0, 0), (0, 0, 0, 0)),
    ((2, 0, 1), (0, 0, 0, 0)),
    ((1, 1, 1), (0, 0, 0, 0)),
    ((0, 1, 1), (0, 0, 0, 0)),  # simple exponential smoothing-like
    ((1, 0, 1), (1, 0, 0, 7)),
    ((1, 0, 0), (1, 0, 0, 7)),
]


def _holdout_mae(y: pd.Series, n_valid: int, pred: np.ndarray) -> float:
    """MAE between last n_valid actuals and first n_valid predictions."""
    actual = y.iloc[-n_valid:].values
    if len(pred) < n_valid:
        return np.nan
    pred = pred[:n_valid]
    mask = np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.nan
    return np.abs(np.asarray(pred)[mask] - np.asarray(actual)[mask]).mean()


def select_best_ets(
    y: pd.Series,
    n_valid: int = 60,
    metric: str = "mae",
) -> tuple[dict[str, Any], float]:
    """
    Select ETS config with lowest holdout MAE (or RMSE).

    Fits on y[:-n_valid], forecasts n_valid steps, compares to y[-n_valid:].

    Parameters
    ----------
    y : pd.Series
        Full history (will be split into train / holdout).
    n_valid : int
        Holdout length (default 60).
    metric : str
        'mae' or 'rmse'.

    Returns
    -------
    best_kwargs : dict
        Pass to fit_ets(**best_kwargs).
    best_score : float
        Best holdout MAE or RMSE.
    """
    y_ = y.dropna().astype(float)
    if len(y_) < n_valid + 30:
        return {"trend": "add", "seasonal": None, "seasonal_periods": None}, np.nan
    train = y_.iloc[:-n_valid]
    actual = y_.iloc[-n_valid:].values

    def score(pred: np.ndarray) -> float:
        pred = pred[:n_valid]
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            return np.nan
        err = np.asarray(pred)[mask] - np.asarray(actual)[mask]
        if metric == "rmse":
            return float(np.sqrt((err ** 2).mean()))
        return float(np.abs(err).mean())

    best_score = np.inf
    best_kwargs: dict[str, Any] = {"trend": "add", "seasonal": None, "seasonal_periods": None}

    for kw in ETS_GRID:
        seasonal_periods = kw.get("seasonal_periods")
        if seasonal_periods is not None and len(train) < seasonal_periods * 2:
            continue
        try:
            m = fit_ets(train, **kw)
            f = m.forecast(steps=n_valid)
            pred = np.asarray(f)
            s = score(pred)
            if not np.isnan(s) and s < best_score:
                best_score = s
                best_kwargs = dict(kw)
        except Exception:
            continue

    return best_kwargs, best_score


def select_best_arima(
    y: pd.Series,
    n_valid: int = 60,
    metric: str = "mae",
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], float]:
    """
    Select ARIMA (order, seasonal_order) with lowest holdout MAE (or RMSE).

    Parameters
    ----------
    y : pd.Series
        Full history.
    n_valid : int
        Holdout length.
    metric : str
        'mae' or 'rmse'.

    Returns
    -------
    best_order : tuple (p, d, q)
    best_seasonal_order : tuple (P, D, Q, s)
    best_score : float
    """
    y_ = y.dropna().astype(float)
    if len(y_) < n_valid + 50:
        return (1, 0, 1), (0, 0, 0, 0), np.nan
    train = y_.iloc[:-n_valid]
    actual = y_.iloc[-n_valid:].values

    def score(pred: np.ndarray) -> float:
        pred = pred[:n_valid]
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.sum() == 0:
            return np.nan
        err = np.asarray(pred)[mask] - np.asarray(actual)[mask]
        if metric == "rmse":
            return float(np.sqrt((err ** 2).mean()))
        return float(np.abs(err).mean())

    best_score = np.inf
    best_order = (1, 0, 1)
    best_seasonal = (0, 0, 0, 0)

    for order, seasonal_order in ARIMA_GRID:
        if len(train) < max(order[0], order[2]) + (seasonal_order[3] or 1) + 10:
            continue
        if seasonal_order[3] and len(train) < seasonal_order[3] * 2:
            continue
        try:
            m = fit_arima(train, order=order, seasonal_order=seasonal_order)
            f = m.get_forecast(steps=n_valid)
            pred = f.predicted_mean.values
            s = score(pred)
            if not np.isnan(s) and s < best_score:
                best_score = s
                best_order = order
                best_seasonal = seasonal_order
        except Exception:
            continue

    return best_order, best_seasonal, best_score

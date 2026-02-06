"""Robust baselines for rolling 60-day forecasts: persistence, seasonal naive, rolling mean."""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_1_step(series: pd.Series) -> float:
    """
    One-step-ahead forecast: next value = last observed value.

    Parameters
    ----------
    series : pd.Series
        Historical values (e.g. target column up to "today"), index ignored.

    Returns
    -------
    float
        Forecast for the next period. NaN if series is empty.
    """
    if len(series) == 0 or series.iloc[-1] is pd.NA:
        return np.nan
    return float(series.iloc[-1])


def seasonal_naive_1_step(series: pd.Series, season_length: int = 365) -> float:
    """
    One-step-ahead forecast: next value = value one season ago.

    Parameters
    ----------
    series : pd.Series
        Historical values (same length or longer than season_length).
    season_length : int
        Season length in periods (default 365 for daily = same day last year).

    Returns
    -------
    float
        Forecast for the next period. NaN if not enough history.
    """
    if len(series) < season_length:
        return np.nan
    val = series.iloc[-season_length]
    return float(val) if pd.notna(val) else np.nan


def rolling_mean_1_step(series: pd.Series, window: int = 7) -> float:
    """
    One-step-ahead forecast: next value = mean of last `window` values.

    Parameters
    ----------
    series : pd.Series
        Historical values.
    window : int
        Rolling window size (default 7).

    Returns
    -------
    float
        Forecast for the next period. NaN if insufficient or all-NaN window.
    """
    tail = series.iloc[-window:] if len(series) >= window else series
    tail = tail.dropna()
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def baseline_forecast_n_days(
    series: pd.Series,
    n_days: int,
    method: str = "persistence",
    **kwargs: int,
) -> pd.DataFrame:
    """
    Produce a recursive n-day forecast using a baseline method.

    Each day's forecast is used as the "observed" value for the next step
    (for persistence; for seasonal_naive/rolling_mean the extended series
    is built by appending the forecast).

    Parameters
    ----------
    series : pd.Series
        Historical values up to and including "today".
    n_days : int
        Number of days to forecast.
    method : str
        One of "persistence", "seasonal_naive", "rolling_mean".
    **kwargs : int
        Passed to the method (e.g. season_length, window).

    Returns
    -------
    pd.DataFrame
        Columns: step (1..n_days), predicted. No uncertainty intervals
        for baselines in this stub.
    """
    method = method.lower()
    extended = series.copy()
    preds = []

    for _ in range(n_days):
        if method == "persistence":
            pred = persistence_1_step(extended)
        elif method == "seasonal_naive":
            pred = seasonal_naive_1_step(extended, **kwargs)
        elif method == "rolling_mean":
            pred = rolling_mean_1_step(extended, **kwargs)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        preds.append(pred)
        extended = pd.concat([extended, pd.Series([pred])], ignore_index=True)

    return pd.DataFrame({"step": range(1, n_days + 1), "predicted": preds})

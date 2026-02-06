"""Simple statistical models (ETS, ARIMA) with point forecasts and uncertainty intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def _ensure_series(y: pd.Series) -> pd.Series:
    """Drop NaN from end for model fitting; return contiguous series."""
    out = y.dropna()
    return out.astype(float)


def fit_ets(
    y: pd.Series,
    seasonal_periods: int | None = 7,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> Any:
    """
    Fit ETS (Exponential Smoothing) model. Requires statsmodels.

    Parameters
    ----------
    y : pd.Series
        Univariate time series (e.g. daily wave height).
    seasonal_periods : int, optional
        Seasonal period (e.g. 7 for weekly). None for non-seasonal.
    trend : str, optional
        'add', 'mul', or None.
    seasonal : str, optional
        'add', 'mul', or None.

    Returns
    -------
    fitted model
        Fitted statsmodels ExponentialSmoothing result.
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for ETS. pip install statsmodels")
    y_ = _ensure_series(y)
    if len(y_) < 2:
        raise ValueError("Need at least 2 non-NaN observations for ETS")
    kwargs: dict[str, Any] = {"trend": trend or "add", "seasonal": seasonal}
    if seasonal_periods is not None:
        kwargs["seasonal_periods"] = min(seasonal_periods, len(y_) // 2 or 1)
    model = ExponentialSmoothing(y_, **kwargs)
    return model.fit(optimized=True)


def forecast_ets(
    model: Any,
    steps: int,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Forecast with fitted ETS model; return point and intervals.

    Parameters
    ----------
    model : fitted ExponentialSmoothing result
    steps : int
        Forecast horizon.
    alpha : float
        Significance level for intervals (e.g. 0.2 -> 80% interval).

    Returns
    -------
    pd.DataFrame
        Columns: step, predicted, lower, upper (single level).
        For 80% and 95% pass alpha=0.2 and 0.05 in separate calls or extend.
    """
    f = model.forecast(steps=steps)
    # get_forecast may not exist on all versions; use forecast + approximate intervals
    pred = np.asarray(f)
    try:
        stderr = np.sqrt(model.sse / len(model.fittedvalues)) if hasattr(model, "sse") else np.nan
    except Exception:
        stderr = np.nan
    if np.isnan(stderr) or stderr <= 0:
        stderr = np.abs(pred).mean() * 0.1 if np.any(pred != 0) else 0.5
    from scipy import stats as scipy_stats
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    half = z * stderr * np.ones(steps)
    return pd.DataFrame({
        "step": range(1, steps + 1),
        "predicted": pred,
        "lower": pred - half,
        "upper": pred + half,
    })


def fit_arima(
    y: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] | None = (0, 0, 0, 0),
) -> Any:
    """
    Fit ARIMA/SARIMA model. Requires statsmodels.

    Parameters
    ----------
    y : pd.Series
        Univariate time series.
    order : tuple
        (p, d, q) for ARIMA.
    seasonal_order : tuple, optional
        (P, D, Q, s) for seasonal part. (0,0,0,0) for plain ARIMA.

    Returns
    -------
    fitted model
        Fitted SARIMAX result.
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for ARIMA. pip install statsmodels")
    y_ = _ensure_series(y)
    if len(y_) < max(order[0], order[2]) + 10:
        raise ValueError("Need more observations for ARIMA")
    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)
    model = SARIMAX(y_, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)


def forecast_arima(
    model: Any,
    steps: int,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Forecast with fitted ARIMA; return point and prediction intervals.

    Parameters
    ----------
    model : fitted SARIMAX result
    steps : int
        Forecast horizon.
    alpha : float
        Significance (e.g. 0.2 -> 80% interval).

    Returns
    -------
    pd.DataFrame
        Columns: step, predicted, lower, upper.
    """
    f = model.get_forecast(steps=steps)
    pred = f.predicted_mean.values
    ci = f.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values
    return pd.DataFrame({
        "step": range(1, steps + 1),
        "predicted": pred,
        "lower": lower,
        "upper": upper,
    })


def statistical_forecast_n_days(
    y: pd.Series,
    n_days: int,
    method: str = "ets",
    interval_levels: list[float] | None = None,
    **fit_kwargs: Any,
) -> pd.DataFrame:
    """
    Fit a statistical model and return n-day point forecast and intervals.

    Parameters
    ----------
    y : pd.Series
        Historical target series.
    n_days : int
        Forecast horizon.
    method : str
        "ets" or "arima".
    interval_levels : list of float, optional
        Significance levels for intervals (e.g. [0.2, 0.05] for 80% and 95%).
    **fit_kwargs
        Passed to fit_ets or fit_arima.

    Returns
    -------
    pd.DataFrame
        Columns: step, predicted, lower_80, upper_80, lower_95, upper_95 (if requested).
    """
    interval_levels = interval_levels or [0.2, 0.05]
    if method.lower() == "ets":
        m = fit_ets(y, **fit_kwargs)
        out = forecast_ets(m, n_days, alpha=interval_levels[0])
        out = out.rename(columns={"lower": "lower_80", "upper": "upper_80"})
        if len(interval_levels) > 1:
            f2 = forecast_ets(m, n_days, alpha=interval_levels[1])
            out["lower_95"] = f2["lower"].values
            out["upper_95"] = f2["upper"].values
    elif method.lower() == "arima":
        m = fit_arima(y, **fit_kwargs)
        out = forecast_arima(m, n_days, alpha=interval_levels[0])
        out = out.rename(columns={"lower": "lower_80", "upper": "upper_80"})
        if len(interval_levels) > 1:
            f2 = forecast_arima(m, n_days, alpha=interval_levels[1])
            out["lower_95"] = f2["lower"].values
            out["upper_95"] = f2["upper"].values
    else:
        raise ValueError(f"Unknown method: {method}")
    return out

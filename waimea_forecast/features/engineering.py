"""Feature engineering for wave height forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from waimea_forecast.config import EXCLUDED_FEATURE_COLUMNS, TARGET_COLUMN

# Lag days for target and key predictors
LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOW = 7

# Columns to exclude from feature set (target and non-numeric)
DATE_COL = "date"


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns excluding date and target."""
    exclude = {DATE_COL, TARGET_COLUMN}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def _add_lags(df: pd.DataFrame, column: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged columns for a single series (no look-ahead)."""
    out = df.copy()
    for lag in lags:
        out[f"{column}_lag{lag}"] = out[column].shift(lag)
    return out


def _add_rolling(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Add rolling mean and std (computed only on past values)."""
    out = df.copy()
    out[f"{column}_roll_mean{window}"] = out[column].shift(1).rolling(window, min_periods=1).mean()
    out[f"{column}_roll_std{window}"] = out[column].shift(1).rolling(window, min_periods=1).std()
    return out


# Annual cycle length for day-of-year encoding (leap-year aware)
DAYS_PER_YEAR = 365.25


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonality features (day-of-year, month, annual cycle sin/cos)."""
    out = df.copy()
    if "date" not in out.columns:
        return out
    dt = pd.to_datetime(out["date"])
    doy = dt.dt.dayofyear
    out["day_of_year"] = doy
    out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    # Annual seasonality: smooth cyclic encoding so Dec 31 is close to Jan 1
    out["day_of_year_sin"] = np.sin(2 * np.pi * doy / DAYS_PER_YEAR)
    out["day_of_year_cos"] = np.cos(2 * np.pi * doy / DAYS_PER_YEAR)
    return out


def _impute_features(df: pd.DataFrame, fitted_medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Impute numeric feature columns: forward fill then median.
    Returns (df with imputed values, fitted medians for inference).
    """
    feature_cols = _get_feature_columns(df)
    if not feature_cols:
        return df, fitted_medians or {}

    out = df.copy()
    if fitted_medians is None:
        fitted_medians = {}
        for c in feature_cols:
            if c in out.columns:
                fitted_medians[c] = out[c].median()
    for c in feature_cols:
        if c not in out.columns:
            continue
        out[c] = out[c].ffill().bfill()
        out[c] = out[c].fillna(fitted_medians.get(c, out[c].median()))
    return out, fitted_medians


def build_features(
    df: pd.DataFrame,
    *,
    lags: list[int] | None = None,
    rolling_window: int = ROLLING_WINDOW,
    fitted_medians: dict[str, float] | None = None,
    impute: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build feature matrix from raw wide-format DataFrame.

    Adds lags of target and key buoy wave heights, rolling stats, and calendar
    features. Optionally imputes missing values (forward fill then median).
    Set impute=False when the estimator will fit an imputer on X_train only.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with 'date' and TARGET_COLUMN.
    lags : list of int, optional
        Lag days (default [1, 2, 3, 7]).
    rolling_window : int, optional
        Rolling window for mean/std (default 7).
    fitted_medians : dict, optional
        Pre-fitted medians for imputation (inference); used only if impute=True.
    impute : bool, optional
        If True, run ffill/bfill/median imputation and return fitted_medians.
        If False, leave NaNs; state has no fitted_medians (for MICE/KNN path).

    Returns
    -------
    pd.DataFrame
        Featurized DataFrame; feature cols may contain NaNs if impute=False.
    dict
        State: fitted_medians (if impute=True), feature_column_order (set later).
    """
    lags = lags or LAG_DAYS
    out = df.copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    # Lags of target
    out = _add_lags(out, TARGET_COLUMN, lags)

    # Lags of other wave height columns (1-day only; skip excluded columns)
    wave_cols = [
        c
        for c in out.columns
        if c.startswith("wave_height_")
        and c != TARGET_COLUMN
        and c not in EXCLUDED_FEATURE_COLUMNS
    ]
    for col in wave_cols[:5]:  # limit to first 5 to avoid explosion
        if col in out.columns:
            out = _add_lags(out, col, [1])

    # Rolling stats for target
    out = _add_rolling(out, TARGET_COLUMN, rolling_window)

    # Calendar
    out = _add_calendar(out)

    if impute:
        out, medians = _impute_features(out, fitted_medians)
        state = {"fitted_medians": medians}
    else:
        state = {}
    return out, state


def prepare_supervised(
    df: pd.DataFrame,
    feature_state: dict[str, Any] | None = None,
    validation_fraction: float = 0.2,
    drop_na_rows: bool = True,
    horizon_days: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], dict]:
    """
    Prepare aligned X, y and train/validation split by time.

    Target is wave height horizon_days ahead. Drops rows where target is NA.
    When drop_na_rows=False, X_train and X_val may contain NaNs (for imputer to fill).

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features (with feature columns and target).
    feature_state : dict, optional
        State from build_features (for consistent column order).
    validation_fraction : float, optional
        Fraction of rows (by time) for validation (default 0.2).
    drop_na_rows : bool, optional
        If True, drop rows where any feature is NaN. If False, return X with NaNs.
    horizon_days : int, optional
        Forecast horizon in days (default 1). Each row's features predict target at +horizon_days.

    Returns
    -------
    X_train, y_train, X_val, y_val : DataFrame, Series, ...
    feature_columns : list[str]
        Column names for X.
    state : dict
        Updated feature_state with feature_column_order and horizon_days.
    """
    clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    clean = clean.sort_values(DATE_COL).reset_index(drop=True)

    # Target: value horizon_days ahead (today's features -> height at +horizon_days)
    y = clean[TARGET_COLUMN].shift(-horizon_days)
    clean = clean.iloc[:-horizon_days].copy()
    y = y.iloc[:-horizon_days]
    clean = clean.loc[y.notna().index]
    y = y.dropna()

    feature_cols = _get_feature_columns(clean)
    feature_cols = [c for c in feature_cols if c in clean.columns]

    X = clean[feature_cols]
    if drop_na_rows:
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]

    n = len(X)
    val_size = max(1, int(n * validation_fraction))
    train_size = n - val_size

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:]
    y_val = y.iloc[train_size:]

    state = dict(feature_state) if feature_state else {}
    state["feature_column_order"] = feature_cols
    state["horizon_days"] = horizon_days

    return X_train, y_train, X_val, y_val, feature_cols, state

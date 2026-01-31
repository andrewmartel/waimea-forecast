"""Feature engineering for wave height forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from waimea_forecast.config import TARGET_COLUMN

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


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year and month (sin/cos for cyclicity)."""
    out = df.copy()
    if "date" not in out.columns:
        return out
    dt = pd.to_datetime(out["date"])
    out["day_of_year"] = dt.dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
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

    # Lags of other wave height columns (1-day only to keep matrix size reasonable)
    wave_cols = [c for c in out.columns if c.startswith("wave_height_") and c != TARGET_COLUMN]
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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], dict]:
    """
    Prepare aligned X, y and train/validation split by time.

    Drops rows where target is NA. Uses last validation_fraction of time for validation.
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

    Returns
    -------
    X_train, y_train, X_val, y_val : DataFrame, Series, ...
    feature_columns : list[str]
        Column names for X.
    state : dict
        Updated feature_state with feature_column_order.
    """
    clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    clean = clean.sort_values(DATE_COL).reset_index(drop=True)

    # Target: next-day (today's features -> tomorrow's height)
    y = clean[TARGET_COLUMN].shift(-1)
    clean = clean.iloc[:-1].copy()
    y = y.iloc[:-1]
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

    return X_train, y_train, X_val, y_val, feature_cols, state

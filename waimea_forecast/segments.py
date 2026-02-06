"""Hierarchy and segments for rolling forecasts (e.g. by buoy or region)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from waimea_forecast.config import TARGET_COLUMN


# Segment key for "all" (aggregate) vs specific segment id
SEGMENT_ALL = "all"


def get_wave_height_columns(df: pd.DataFrame) -> list[str]:
    """Return list of wave_height_* columns present in df (excluding excluded)."""
    from waimea_forecast.config import EXCLUDED_FEATURE_COLUMNS
    cols = [c for c in df.columns if c.startswith("wave_height_")]
    return [c for c in cols if c not in EXCLUDED_FEATURE_COLUMNS]


def get_segments(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Define segments for hierarchical/segmented forecasting.

    Each segment is a dict with at least:
    - segment_id : str (e.g. "all", "51201h", "51000h")
    - column : str (column name in df for that segment's target)
    - label : str (human-readable, optional)

    Parameters
    ----------
    df : pd.DataFrame
        Wide data with date and wave_height_* columns.

    Returns
    -------
    list of dict
        One entry per segment. First is typically "all" (TARGET_COLUMN).
    """
    wave_cols = get_wave_height_columns(df)
    if not wave_cols:
        return [{"segment_id": SEGMENT_ALL, "column": TARGET_COLUMN, "label": "target"}]
    segments = [{"segment_id": SEGMENT_ALL, "column": TARGET_COLUMN, "label": "target (North Shore)"}]
    for c in wave_cols:
        if c == TARGET_COLUMN:
            continue
        sid = c.replace("wave_height_", "").replace("_", "").lower() or "other"
        segments.append({"segment_id": sid, "column": c, "label": c})
    return segments


def forecast_by_segment(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    target_col: str,
    forecast_fn: Any,
    n_days: int = 60,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Run a univariate forecast per segment and return combined DataFrame.

    forecast_fn(series, n_days, **kwargs) -> DataFrame with at least
    columns: step, predicted, and optionally lower_80, upper_80, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Wide data with date and segment columns.
    as_of_date : pd.Timestamp
        Last date of history (inclusive).
    target_col : str
        Column name for this segment's target.
    forecast_fn : callable
        (series, n_days, **kwargs) -> DataFrame.
    n_days : int
        Forecast horizon.
    **kwargs
        Passed to forecast_fn.

    Returns
    -------
    pd.DataFrame
        Forecast with segment_id column set; step, predicted, and any intervals.
    """
    mask = df["date"] <= as_of_date
    sub = df.loc[mask, ["date", target_col]].sort_values("date")
    series = sub[target_col].copy()
    series.index = sub["date"]
    out = forecast_fn(series, n_days, **kwargs)
    out["segment_id"] = target_col
    return out


def reconcile_bottom_up(
    segment_forecasts: list[pd.DataFrame],
    aggregate_segment_id: str = SEGMENT_ALL,
) -> list[pd.DataFrame]:
    """
    Placeholder: bottom-up reconciliation.

    In bottom-up, the aggregate forecast is the sum (or mean) of the
    segment forecasts. Here we leave segment forecasts unchanged and
    optionally recompute the aggregate from segments if present.

    Parameters
    ----------
    segment_forecasts : list of DataFrame
        One forecast per segment (each has step, predicted, segment_id).
    aggregate_segment_id : str
        Which segment_id is the aggregate (e.g. "all").

    Returns
    -------
    list of DataFrame
        Same list; stub returns inputs unchanged. Full implementation
        would recompute aggregate from segment predictions.
    """
    return segment_forecasts


def reconcile_top_down(
    aggregate_forecast: pd.DataFrame,
    segment_forecasts: list[pd.DataFrame],
    proportions: dict[str, float] | None = None,
) -> list[pd.DataFrame]:
    """
    Placeholder: top-down reconciliation.

    Allocate aggregate forecast to segments by proportions (e.g. historical
    share). Stub returns segment_forecasts unchanged.

    Parameters
    ----------
    aggregate_forecast : DataFrame
        Point (and optional intervals) for aggregate.
    segment_forecasts : list of DataFrame
        Per-segment forecasts (may be overwritten by proportions).
    proportions : dict, optional
        segment_id -> proportion of aggregate.

    Returns
    -------
    list of DataFrame
        Stub returns segment_forecasts unchanged.
    """
    return segment_forecasts

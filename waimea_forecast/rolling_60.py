"""Rolling 60-day forecast orchestrator: baseline + statistical model + segments + intervals."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from waimea_forecast.config import TARGET_COLUMN
from waimea_forecast.baseline import baseline_forecast_n_days
from waimea_forecast.statistical import statistical_forecast_n_days
from waimea_forecast.segments import SEGMENT_ALL, get_segments
from waimea_forecast.tuning import select_best_ets, select_best_arima


def run_rolling_60(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    n_days: int = 60,
    *,
    baseline_method: str = "persistence",
    statistical_method: str = "ets",
    use_segments: bool = False,
    reconcile: str | None = None,
    interval_levels: list[float] | None = None,
    tune_statistical: bool = False,
    n_tune_holdout: int | None = None,
) -> pd.DataFrame:
    """
    Produce a rolling n-day daily forecast from as_of_date.

    Combines:
    - Robust baseline (persistence, seasonal_naive, or rolling_mean)
    - Simple statistical model (ETS or ARIMA) with uncertainty intervals
    - Optional segments (e.g. by buoy) with optional reconciliation

    Parameters
    ----------
    df : pd.DataFrame
        Wide data with 'date' and TARGET_COLUMN (and optionally other wave columns).
    as_of_date : datetime or str
        Last date of history (inclusive). Forecast is for as_of_date+1 .. as_of_date+n_days.
    n_days : int
        Number of days to forecast (default 60).
    baseline_method : str
        "persistence", "seasonal_naive", or "rolling_mean".
    statistical_method : str
        "ets" or "arima".
    use_segments : bool
        If True, run forecasts per segment (from get_segments) and concatenate.
    reconcile : str, optional
        "bottom_up" or "top_down"; applied if use_segments. Stub leaves forecasts unchanged.
    interval_levels : list of float, optional
        For statistical model intervals (e.g. [0.2, 0.05] for 80% and 95%).

    Returns
    -------
    pd.DataFrame
        Columns: date, step, model (baseline | ets | arima), predicted,
        lower_80, upper_80, lower_95, upper_95 (NaN for baseline),
        segment_id (if use_segments).
    """
    as_of = pd.Timestamp(as_of_date)
    df = df.sort_values("date").reset_index(drop=True)
    mask = df["date"] <= as_of
    if not mask.any():
        raise ValueError(f"No rows on or before as_of_date={as_of}")
    hist = df.loc[mask].copy()
    interval_levels = interval_levels or [0.2, 0.05]
    n_holdout = n_tune_holdout if n_tune_holdout is not None else n_days

    # Optional tuning: choose best ETS/ARIMA config by holdout MAE
    stat_kwargs: dict[str, Any] = {}
    if tune_statistical:
        series_tune = hist[TARGET_COLUMN].dropna().astype(float)
        if statistical_method.lower() == "ets":
            best_kw, best_score = select_best_ets(series_tune, n_valid=n_holdout, metric="mae")
            stat_kwargs = best_kw
        elif statistical_method.lower() == "arima":
            best_order, best_seasonal, best_score = select_best_arima(
                series_tune, n_valid=n_holdout, metric="mae"
            )
            stat_kwargs = {"order": best_order, "seasonal_order": best_seasonal}

    # Build forecast dates
    first_fc_date = hist["date"].iloc[-1] + pd.Timedelta(days=1)
    dates = [first_fc_date + pd.Timedelta(days=i) for i in range(n_days)]

    def _baseline_fn(series: pd.Series, steps: int, **kw: Any) -> pd.DataFrame:
        return baseline_forecast_n_days(series, steps, method=baseline_method, **kw)

    def _stat_fn(series: pd.Series, steps: int, **kw: Any) -> pd.DataFrame:
        merged_kw = {**stat_kwargs, **kw}
        return statistical_forecast_n_days(
            series, steps, method=statistical_method,
            interval_levels=interval_levels, **merged_kw
        )

    rows = []

    if use_segments:
        segments = get_segments(df)
        for seg in segments:
            seg_id = seg["segment_id"]
            col = seg["column"]
            if col not in hist.columns or hist[col].dropna().empty:
                continue
            series = hist[col].reset_index(drop=True)
            # Baseline
            bl = _baseline_fn(series, n_days)
            for i, r in bl.iterrows():
                rows.append({
                    "date": dates[int(r["step"]) - 1],
                    "step": int(r["step"]),
                    "model": "baseline",
                    "predicted": r["predicted"],
                    "lower_80": pd.NA,
                    "upper_80": pd.NA,
                    "lower_95": pd.NA,
                    "upper_95": pd.NA,
                    "segment_id": seg_id,
                })
            # Statistical
            try:
                st = _stat_fn(series, n_days)
                for i, r in st.iterrows():
                    row = {
                        "date": dates[int(r["step"]) - 1],
                        "step": int(r["step"]),
                        "model": statistical_method,
                        "predicted": r["predicted"],
                        "segment_id": seg_id,
                    }
                    row["lower_80"] = r.get("lower_80", pd.NA)
                    row["upper_80"] = r.get("upper_80", pd.NA)
                    row["lower_95"] = r.get("lower_95", pd.NA)
                    row["upper_95"] = r.get("upper_95", pd.NA)
                    rows.append(row)
            except Exception:
                pass  # skip segment if ETS/ARIMA fails
        if reconcile == "bottom_up":
            # Stub: no change
            pass
    else:
        series = hist[TARGET_COLUMN].reset_index(drop=True)
        # Baseline
        bl = _baseline_fn(series, n_days)
        for i, r in bl.iterrows():
            rows.append({
                "date": dates[int(r["step"]) - 1],
                "step": int(r["step"]),
                "model": "baseline",
                "predicted": r["predicted"],
                "lower_80": pd.NA,
                "upper_80": pd.NA,
                "lower_95": pd.NA,
                "upper_95": pd.NA,
                "segment_id": SEGMENT_ALL,
            })
        # Statistical
        try:
            st = _stat_fn(series, n_days)
            for i, r in st.iterrows():
                rows.append({
                    "date": dates[int(r["step"]) - 1],
                    "step": int(r["step"]),
                    "model": statistical_method,
                    "predicted": r["predicted"],
                    "lower_80": r.get("lower_80", pd.NA),
                    "upper_80": r.get("upper_80", pd.NA),
                    "lower_95": r.get("lower_95", pd.NA),
                    "upper_95": r.get("upper_95", pd.NA),
                    "segment_id": SEGMENT_ALL,
                })
        except Exception as e:
            raise RuntimeError(f"Statistical forecast failed: {e}") from e

    out = pd.DataFrame(rows)
    return out

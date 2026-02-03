"""Extended forecast utilities: e.g. next N days from a given "today"."""

from __future__ import annotations

import pandas as pd

from waimea_forecast.config import TARGET_COLUMN


def predict_next_n_days(
    estimator: "WaveHeightEstimator",
    df: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    n_days: int = 60,
    threshold_m: float | None = None,
) -> pd.DataFrame:
    """
    Produce daily predictions for the next n_days from "today" (as_of_date).

    Uses recursive 1-step-ahead prediction: each day's forecast is used as
    input (e.g. in lags) for the next. Requires a 1-day horizon model; for
    7d/30d models only the first horizon step is meaningful for "day 1" and
    the rest are recursive approximations.

    Parameters
    ----------
    estimator : WaveHeightEstimator
        Fitted estimator (ideally horizon_days=1 for true daily recursion).
    df : pd.DataFrame
        Full history with 'date' and target; will be truncated to as_of_date.
    as_of_date : datetime or str
        "Today"; predictions are for as_of_date+1, as_of_date+2, ..., as_of_date+n_days.
    n_days : int, optional
        Number of days to forecast (default 60).
    threshold_m : float, optional
        Threshold (m) for P(contest-ready); default: estimator's calibrator threshold.

    Returns
    -------
    pd.DataFrame
        Columns: date, predicted, p_contest_ready (one row per forecast day).
    """
    as_of = pd.Timestamp(as_of_date)
    df = df.sort_values("date").reset_index(drop=True)
    mask = df["date"] <= as_of
    if not mask.any():
        raise ValueError(f"No rows on or before as_of_date={as_of}")
    df_today = df.loc[mask].copy()

    threshold = threshold_m if threshold_m is not None else getattr(
        estimator, "_calibrator_threshold_m", 3.0
    )

    dates = []
    preds = []
    p_contest_list = []

    for _ in range(n_days):
        point_pred, p_above = estimator.predict_with_proba(df_today, threshold_m=threshold)
        # Prediction is for the next horizon_days; take the last row's forecast
        pred_val = float(point_pred[-1])
        p_val = float(p_above[-1])
        next_date = df_today["date"].iloc[-1] + pd.Timedelta(days=estimator._horizon_days)
        dates.append(next_date)
        preds.append(pred_val)
        p_contest_list.append(p_val)
        # Append one row so next iteration has "today" = next_date
        new_row = df_today.iloc[-1].copy()
        new_row["date"] = next_date
        new_row[TARGET_COLUMN] = pred_val
        df_today = pd.concat([df_today, new_row.to_frame().T], ignore_index=True)

    return pd.DataFrame({
        "date": dates,
        "predicted": preds,
        "p_contest_ready": p_contest_list,
    })

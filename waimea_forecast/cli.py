"""CLI entry points for train and predict."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from waimea_forecast.config import (
    CONTEST_THRESHOLD_M,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_DATA_PATH,
    FORECAST_HORIZON_DAYS,
    TARGET_COLUMN,
)
from waimea_forecast.data.loader import load_wide, validate_schema
from waimea_forecast.metrics import format_validation_report
from waimea_forecast.models.estimator import WaveHeightEstimator


def main_train() -> None:
    """Load data, train estimator, save artifact; print train/validation metrics."""
    parser = argparse.ArgumentParser(description="Train Waimea wave height estimator")
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("WAIMEA_DATA_PATH", DEFAULT_DATA_PATH),
        help="Path to wide.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("WAIMEA_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
        help="Path to save artifact",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FORECAST_HORIZON_DAYS,
        choices=[1, 7, 30],
        help="Forecast horizon in days (1, 7, or 30)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    est = WaveHeightEstimator(horizon_days=args.horizon)
    est.fit(df)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    est.save(model_path)
    print(f"Saved artifact to {model_path}", file=sys.stderr)

    from waimea_forecast.features.engineering import build_features, prepare_supervised

    featurized, _ = build_features(df, impute=False)
    _, _, X_val, y_val, _, _ = prepare_supervised(
        featurized,
        est._feature_state,
        validation_fraction=0.2,
        drop_na_rows=False,
    )
    X_val_imp = est._imputer.transform(X_val)
    val_ok = ~np.isnan(X_val_imp).any(axis=1)
    y_val_clean = y_val.iloc[np.where(val_ok)[0]]
    preds = est.predict(df)
    val_preds = np.array([preds[i] for i in y_val_clean.index if i < len(preds)])
    if len(val_preds) == len(y_val_clean):
        report = format_validation_report(
            y_val_clean.values, val_preds, threshold=CONTEST_THRESHOLD_M
        )
        print(report, file=sys.stderr)


def main_predict() -> None:
    """Load artifact and data; output predictions to stdout or file."""
    parser = argparse.ArgumentParser(description="Run Waimea wave height predictions")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to wide.csv (or input data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("WAIMEA_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
        help="Path to artifact",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for predictions (default: stdout)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    est = WaveHeightEstimator.load(model_path)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    preds = est.predict(df)
    h = getattr(est, "_horizon_days", 1)
    n = len(df)
    pred_dates = df["date"].iloc[h:n].reset_index(drop=True)
    pred_values = np.asarray(preds[: n - h])
    actual_values = df[TARGET_COLUMN].iloc[h:n].reset_index(drop=True)
    # Drop rows where actual is missing
    valid = actual_values.notna()
    pred_dates = pred_dates[valid]
    pred_values = pred_values[valid.values]
    actual_values = actual_values[valid].values
    abs_delta = np.abs(pred_values - actual_values)
    correct_3m = (
        (np.asarray(pred_values >= CONTEST_THRESHOLD_M))
        == (np.asarray(actual_values >= CONTEST_THRESHOLD_M))
    ).astype(int)

    out_df = pd.DataFrame(
        {
            "date": pred_dates,
            "predicted": pred_values,
            "actual": actual_values,
            "abs_delta": abs_delta,
            "correct_3m": correct_3m,
        }
    )

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Wrote predictions to {out_path}", file=sys.stderr)
    else:
        print(out_df.to_csv(index=False, date_format="%Y-%m-%d"))


def main_rolling_60() -> None:
    """Rolling N-day forecast: baseline + ETS/ARIMA + optional segments + intervals."""
    from waimea_forecast.rolling_60 import run_rolling_60

    parser = argparse.ArgumentParser(
        description="Rolling 60-day daily forecast: baseline + statistical model + intervals"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("WAIMEA_DATA_PATH", DEFAULT_DATA_PATH),
        help="Path to wide.csv",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        required=True,
        metavar="DATE",
        help="'Today' date (YYYY-MM-DD); forecast starts the next day",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        metavar="N",
        help="Number of days to forecast (default: 60)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="persistence",
        choices=["persistence", "seasonal_naive", "rolling_mean"],
        help="Baseline method (default: persistence)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ets",
        choices=["ets", "arima"],
        help="Statistical model for point + intervals (default: ets)",
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Run per-segment forecasts (by wave_height column)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune ETS/ARIMA config by holdout MAE before forecasting",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: stdout)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    result = run_rolling_60(
        df,
        args.as_of,
        n_days=args.days,
        baseline_method=args.baseline,
        statistical_method=args.model,
        use_segments=args.segments,
        tune_statistical=args.tune,
    )

    out_path = Path(args.out) if args.out else None
    if out_path:
        result.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Wrote rolling forecast to {out_path}", file=sys.stderr)
    else:
        print(result.to_csv(index=False, date_format="%Y-%m-%d"))
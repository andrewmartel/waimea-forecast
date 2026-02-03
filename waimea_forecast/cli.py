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
from waimea_forecast.explain import export_weights_and_shap
from waimea_forecast.metrics import format_validation_report
from waimea_forecast.forecast import predict_next_n_days
from waimea_forecast.models.estimator import WaveHeightEstimator
from waimea_forecast.tuning import select_max_features_cv


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
    parser.add_argument(
        "--max-features",
        type=str,
        default=None,
        metavar="K or cv",
        help="Top K features by |Ridge coefficient|, or 'cv' to choose K by validation MAE; default: use all",
    )
    parser.add_argument(
        "--max-features-candidates",
        type=str,
        default=None,
        metavar="K1 K2 ...",
        help="When --max-features cv, try these K values (default: 10 20 30 40 50 60 80 100 all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONTEST_THRESHOLD_M,
        metavar="M",
        help="Wave height threshold (m) for contest-ready; used for Platt scaling P(>= M) (default: 3.0)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    max_features: int | None = None
    if args.max_features is not None:
        if args.max_features.strip().lower() == "cv":
            candidates_raw = (
                args.max_features_candidates or "10 20 30 40 50 60 80 100"
            ).split()
            candidates: list[int | None] = []
            for s in candidates_raw:
                s = s.strip().lower()
                if s == "all" or s == "":
                    candidates.append(None)
                else:
                    try:
                        candidates.append(int(s))
                    except ValueError:
                        pass
            if None not in candidates:
                candidates.append(None)
            print("Selecting max_features by validation MAE...", file=sys.stderr)
            best_k, results = select_max_features_cv(
                df, args.horizon, candidates=candidates
            )
            print("max_features  validation_MAE", file=sys.stderr)
            for k, mae in results:
                label = "all" if k is None else str(k)
                print(f"  {label:>6}  {mae:.4f}", file=sys.stderr)
            print(
                f"Best: max_features={best_k if best_k is not None else 'all'}",
                file=sys.stderr,
            )
            max_features = best_k
        else:
            try:
                max_features = int(args.max_features)
            except ValueError:
                print(
                    f"Invalid --max-features: use an integer or 'cv'. Got: {args.max_features!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

    est = WaveHeightEstimator(
        horizon_days=args.horizon,
        max_features=max_features,
    )
    est.fit(df)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    est.save(model_path)
    print(f"Saved artifact to {model_path}", file=sys.stderr)

    from waimea_forecast.features.engineering import build_features, prepare_supervised

    featurized, _ = build_features(
        df, horizon_days=est._horizon_days, impute=False
    )
    _, _, X_val, y_val, _, _ = prepare_supervised(
        featurized,
        est._feature_state,
        validation_fraction=0.2,
        drop_na_rows=False,
        horizon_days=est._horizon_days,
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

    # Model weights and SHAP plots (same validation features)
    X_val_imp = est._imputer.transform(X_val)
    X_val_imp = np.where(np.isnan(X_val_imp), 0.0, X_val_imp)
    X_val_scaled = est._scaler.transform(X_val_imp)
    weights_path, shap_path = export_weights_and_shap(
        est,
        X_val_imp,
        X_val_scaled,
        est._feature_columns,
        model_path.parent,
        model_path.stem,
    )
    print(f"Saved weights plot to {weights_path}", file=sys.stderr)
    print(f"Saved SHAP plot to {shap_path}", file=sys.stderr)


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="M",
        help="Wave height threshold (m) for P(contest-ready); default: use model's calibrator threshold",
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

    threshold = args.threshold if args.threshold is not None else getattr(
        est, "_calibrator_threshold_m", CONTEST_THRESHOLD_M
    )
    pred_values, p_contest = est.predict_with_proba(df, threshold_m=threshold)
    h = getattr(est, "_horizon_days", 1)
    n = len(df)
    pred_dates = df["date"].iloc[h:n].reset_index(drop=True)
    pred_values = np.asarray(pred_values[: n - h])
    p_contest = np.asarray(p_contest[: n - h])
    actual_values = df[TARGET_COLUMN].iloc[h:n].reset_index(drop=True)
    # Drop rows where actual is missing
    valid = actual_values.notna()
    pred_dates = pred_dates[valid]
    pred_values = pred_values[valid.values]
    p_contest = p_contest[valid.values]
    actual_values = actual_values[valid].values
    abs_delta = np.abs(pred_values - actual_values)
    correct_contest_ready = (
        (np.asarray(pred_values >= threshold))
        == (np.asarray(actual_values >= threshold))
    ).astype(int)

    out_df = pd.DataFrame(
        {
            "date": pred_dates,
            "predicted": pred_values,
            "p_contest_ready": p_contest,
            "actual": actual_values,
            "abs_delta": abs_delta,
            "correct_contest_ready": correct_contest_ready,
        }
    )

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Wrote predictions to {out_path}", file=sys.stderr)
    else:
        print(out_df.to_csv(index=False, date_format="%Y-%m-%d"))


def main_explain() -> None:
    """Load artifact and data; save model-weights and SHAP plots."""
    parser = argparse.ArgumentParser(
        description="Generate model weights and SHAP plots from a saved artifact"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("WAIMEA_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
        help="Path to artifact",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("WAIMEA_DATA_PATH", DEFAULT_DATA_PATH),
        help="Path to wide.csv (used to compute SHAP on a sample)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for plots (default: same as model)",
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

    from waimea_forecast.features.engineering import build_features, prepare_supervised

    featurized, _ = build_features(
        df, horizon_days=est._horizon_days, impute=False
    )
    _, _, X_val, _, _, _ = prepare_supervised(
        featurized,
        est._feature_state,
        validation_fraction=0.2,
        drop_na_rows=False,
        horizon_days=est._horizon_days,
    )
    X_val_imp = est._imputer.transform(X_val)
    X_val_imp = np.where(np.isnan(X_val_imp), 0.0, X_val_imp)
    X_val_scaled = est._scaler.transform(X_val_imp)

    out_dir = Path(args.out_dir) if args.out_dir else model_path.parent
    weights_path, shap_path = export_weights_and_shap(
        est,
        X_val_imp,
        X_val_scaled,
        est._feature_columns,
        out_dir,
        model_path.stem,
    )
    print(f"Saved weights plot to {weights_path}", file=sys.stderr)
    print(f"Saved SHAP plot to {shap_path}", file=sys.stderr)


def main_forecast_60() -> None:
    """Daily predictions for the next N days from 'today'; optional validation on final 60 days."""
    parser = argparse.ArgumentParser(
        description="Forecast next N days from a given 'today'; or validate on final 60 days"
    )
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
        help="Path to artifact",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        metavar="DATE",
        help="'Today' date (YYYY-MM-DD). Required unless --validate-60",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days to forecast (default: 60)",
    )
    parser.add_argument(
        "--validate-60",
        action="store_true",
        help="Use 61st-from-end row as 'today' and compare to actuals for final 60 days",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="M",
        help="Wave height threshold (m) for P(contest-ready); default: model's calibrator threshold",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: stdout)",
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

    df = df.sort_values("date").reset_index(drop=True)
    threshold = args.threshold or getattr(est, "_calibrator_threshold_m", CONTEST_THRESHOLD_M)

    if args.validate_60:
        if len(df) < args.days + 1:
            print(
                f"Need at least {args.days + 1} rows for --validate-60",
                file=sys.stderr,
            )
            sys.exit(1)
        as_of_idx = len(df) - args.days - 1
        as_of_date = df["date"].iloc[as_of_idx]
        print(f"Validating: as_of_date={as_of_date} (61st from end)", file=sys.stderr)
    else:
        if not args.as_of:
            print("Provide --as-of DATE or use --validate-60", file=sys.stderr)
            sys.exit(1)
        as_of_date = args.as_of

    fc = predict_next_n_days(est, df, as_of_date, n_days=args.days, threshold_m=threshold)

    if args.validate_60:
        actual_dates = df["date"].iloc[-args.days:].reset_index(drop=True)
        actual_values = df[TARGET_COLUMN].iloc[-args.days:].reset_index(drop=True)
        fc = fc.merge(
            pd.DataFrame({"date": actual_dates, "actual": actual_values}),
            on="date",
            how="left",
        )
        fc["abs_delta"] = (fc["predicted"] - fc["actual"]).abs()

    out_path = Path(args.out) if args.out else None
    if out_path:
        fc.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        print(f"Wrote to {out_path}", file=sys.stderr)
    else:
        print(fc.to_csv(index=False, date_format="%Y-%m-%d"))
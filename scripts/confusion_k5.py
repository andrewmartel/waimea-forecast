#!/usr/bin/env python3
"""Train 30d model with max_features=5 and print confusion matrices (>=3m) on validation and prediction."""

import sys
from pathlib import Path

import numpy as np

# Add project root for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waimea_forecast.config import CONTEST_THRESHOLD_M, TARGET_COLUMN
from waimea_forecast.data.loader import load_wide, validate_schema
from waimea_forecast.features.engineering import build_features, prepare_supervised
from waimea_forecast.metrics import classification_metrics
from waimea_forecast.models.estimator import WaveHeightEstimator


def main():
    data_path = Path("data/wide.csv")
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    est = WaveHeightEstimator(horizon_days=30, max_features=5)
    est.fit(df)

    # Validation set (same 80/20 split as training)
    featurized, _ = build_features(df, horizon_days=30, impute=False)
    _, _, X_val, y_val, _, _ = prepare_supervised(
        featurized, est._feature_state,
        validation_fraction=0.2, drop_na_rows=False, horizon_days=30,
    )
    X_val_imp = est._imputer.transform(X_val)
    val_ok = ~np.isnan(X_val_imp).any(axis=1)
    y_val_clean = y_val.iloc[np.where(val_ok)[0]]
    preds = est.predict(df)
    val_preds = np.array([preds[i] for i in y_val_clean.index if i < len(preds)])

    if len(val_preds) == len(y_val_clean):
        clf_val = classification_metrics(y_val_clean.values, val_preds, threshold=CONTEST_THRESHOLD_M)
        print("=== Validation (30d, max_features=5) — contest-ready >= 3 m ===")
        print("  Rows: true, Cols: predicted. Labels: 0 = <3m, 1 = >=3m")
        print("  Confusion matrix:")
        print("    ", clf_val["confusion_matrix"])
        print("  Accuracy:", clf_val["accuracy"])
        print("  Any predicted >= 3m?", (np.asarray(val_preds) >= CONTEST_THRESHOLD_M).any())
        print()

    # Full prediction set (all rows with actuals, like predict CLI)
    h = 30
    n = len(df)
    pred_values = np.asarray(preds[: n - h])
    actual_values = df[TARGET_COLUMN].iloc[h:n].reset_index(drop=True)
    valid = actual_values.notna()
    pred_values = pred_values[valid.values]
    actual_values = actual_values[valid].values

    clf_pred = classification_metrics(actual_values, pred_values, threshold=CONTEST_THRESHOLD_M)
    print("=== Prediction (full out-of-sample, 30d, max_features=5) — contest-ready >= 3 m ===")
    print("  Rows: true, Cols: predicted. Labels: 0 = <3m, 1 = >=3m")
    print("  Confusion matrix:")
    print("    ", clf_pred["confusion_matrix"])
    print("  Accuracy:", clf_pred["accuracy"])
    print("  Any predicted >= 3m?", (pred_values >= CONTEST_THRESHOLD_M).any())
    print("  Max predicted (m):", float(np.max(pred_values)))


if __name__ == "__main__":
    main()

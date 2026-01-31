"""CLI entry points for train and predict."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from waimea_forecast.config import DEFAULT_ARTIFACT_PATH, DEFAULT_DATA_PATH
from waimea_forecast.data.loader import load_wide, validate_schema
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
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    df = load_wide(data_path)
    validate_schema(df)

    est = WaveHeightEstimator()
    est.fit(df)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    est.save(model_path)
    print(f"Saved artifact to {model_path}", file=sys.stderr)

    from waimea_forecast.features.engineering import build_features, prepare_supervised

    featurized, _ = build_features(df)
    _, _, X_val, y_val, _, _ = prepare_supervised(
        featurized, est._feature_state, validation_fraction=0.2
    )
    preds = est.predict(df)
    val_preds = np.array([preds[i] for i in y_val.index if i < len(preds)])
    if len(val_preds) == len(y_val):
        mae = np.abs(val_preds - y_val.values).mean()
        print(f"Validation MAE: {mae:.4f} m", file=sys.stderr)


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
    out_path = Path(args.out) if args.out else None
    if out_path:
        np.savetxt(out_path, preds, fmt="%.6f")
        print(f"Wrote predictions to {out_path}", file=sys.stderr)
    else:
        for p in preds:
            print(p)

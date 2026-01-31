"""Regression and classification metrics for wave height forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from waimea_forecast.config import CONTEST_THRESHOLD_M


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, R², and bias (mean error)."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "bias": np.nan}

    err = y_pred - y_true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    # MAPE: avoid div by zero; use small epsilon or skip zeros
    denom = np.abs(y_true)
    denom[denom < 1e-10] = np.nan
    pct = np.abs(err) / denom
    mape = np.nanmean(pct) * 100.0 if np.any(np.isfinite(pct)) else np.nan
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    bias = err.mean()

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "bias": float(bias),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = CONTEST_THRESHOLD_M,
) -> dict[str, Any]:
    """
    Binary classification: 1 = >= threshold, 0 = < threshold.
    Returns confusion matrix, accuracy, precision, recall, F1, ROC AUC.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {
            "confusion_matrix": None,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
        }

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true_bin, y_pred)  # use continuous pred for AUC
    except ValueError:
        roc_auc = np.nan

    return {
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def format_validation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = CONTEST_THRESHOLD_M,
) -> str:
    """Format regression and classification metrics for printing."""
    reg = regression_metrics(y_true, y_pred)
    clf = classification_metrics(y_true, y_pred, threshold=threshold)

    lines = [
        "--- Validation (regression) ---",
        f"  MAE:   {reg['MAE']:.4f} m",
        f"  RMSE:  {reg['RMSE']:.4f} m",
        f"  MAPE:  {reg['MAPE']:.2f}%",
        f"  R2:    {reg['R2']:.4f}",
        f"  bias:  {reg['bias']:.4f} m",
        "",
        f"--- Validation (classification, >= {threshold}m = class 1) ---",
        "  Confusion matrix (rows=true, cols=pred):",
        f"    {clf['confusion_matrix']}",
        f"  accuracy:  {clf['accuracy']:.4f}",
        f"  precision: {clf['precision']:.4f}",
        f"  recall:    {clf['recall']:.4f}",
        f"  F1:        {clf['f1']:.4f}",
        f"  ROC AUC:   {clf['roc_auc']:.4f}",
    ]
    return "\n".join(lines)

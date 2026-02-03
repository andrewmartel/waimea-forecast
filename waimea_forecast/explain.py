"""Model interpretability: plot Ridge weights and SHAP values."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from waimea_forecast.models.estimator import WaveHeightEstimator

# Cap sample size for SHAP to keep runtime reasonable
SHAP_SAMPLE_SIZE = 500


def plot_model_weights(
    estimator: "WaveHeightEstimator",
    path: str | Path,
) -> None:
    """
    Plot Ridge regression coefficients (model weights) as a horizontal bar chart.

    Parameters
    ----------
    estimator : WaveHeightEstimator
        Fitted estimator (must have _model and _feature_columns).
    path : str or Path
        Output path for the figure (e.g. model_weights.png).
    """
    import matplotlib.pyplot as plt

    if estimator._model is None:
        raise ValueError("Estimator not fitted; call fit() first.")

    coef = np.asarray(estimator._model.coef_).ravel()
    names = estimator._feature_columns or [f"f{i}" for i in range(len(coef))]

    # Sort by absolute value for readability
    order = np.argsort(np.abs(coef))
    coef = coef[order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(6, len(names) * 0.25)))
    y_pos = np.arange(len(names))
    colors = np.where(coef >= 0, "steelblue", "coral")
    ax.barh(y_pos, coef, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Coefficient (weight)")
    ax.set_title("Model weights (Ridge coefficients)")
    ax.axvline(0, color="gray", linewidth=0.8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap(
    estimator: "WaveHeightEstimator",
    X_scaled: np.ndarray,
    feature_names: list[str],
    path: str | Path,
    X_display: np.ndarray | None = None,
) -> None:
    """
    Compute SHAP values for the Ridge model and save a summary bar plot.

    Uses a subsample of X_scaled if large. LinearExplainer is exact for Ridge.

    Parameters
    ----------
    estimator : WaveHeightEstimator
        Fitted estimator (must have _model).
    X_scaled : np.ndarray
        Scaled feature matrix (same scale as model input), shape (n_samples, n_features).
    feature_names : list[str]
        Names for each feature.
    path : str or Path
        Output path for the figure (e.g. shap_summary.png).
    X_display : np.ndarray, optional
        Same shape as X_scaled; used for feature value display in the plot (e.g. unscaled).
        If None, X_scaled is used.
    """
    import shap
    import matplotlib.pyplot as plt

    if estimator._model is None:
        raise ValueError("Estimator not fitted; call fit() first.")

    n = X_scaled.shape[0]
    if n > SHAP_SAMPLE_SIZE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SHAP_SAMPLE_SIZE, replace=False)
        X_bg = X_scaled[idx]
        X_explain = X_scaled[idx]
        if X_display is not None:
            X_display = X_display[idx]
    else:
        X_bg = X_scaled
        X_explain = X_scaled

    if X_display is None:
        X_display = X_explain

    explainer = shap.LinearExplainer(estimator._model, X_bg)
    shap_values = explainer.shap_values(X_explain)

    # Mean absolute SHAP for bar plot (global importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]
    mean_abs_shap = mean_abs_shap[order]
    names_ordered = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(6, len(names_ordered) * 0.25)))
    y_pos = np.arange(len(names_ordered))
    ax.barh(y_pos, mean_abs_shap, color="steelblue", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_ordered, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP feature importance (mean absolute impact on prediction)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_weights_and_shap(
    estimator: "WaveHeightEstimator",
    X_val_imputed: np.ndarray,
    X_val_scaled: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    model_stem: str,
) -> tuple[Path, Path]:
    """
    Save model-weights and SHAP summary plots next to the artifact.

    Parameters
    ----------
    estimator : WaveHeightEstimator
        Fitted estimator.
    X_val_imputed : np.ndarray
        Validation features after imputation (original scale), for display.
    X_val_scaled : np.ndarray
        Validation features after scaling (model input).
    feature_names : list[str]
        Feature column names.
    output_dir : str or Path
        Directory for output files (e.g. models/).
    model_stem : str
        Base name without extension (e.g. artifact_30d).

    Returns
    -------
    weights_path, shap_path : Path, Path
        Paths to the saved figures.
    """
    output_dir = Path(output_dir)
    weights_path = output_dir / f"{model_stem}_weights.png"
    shap_path = output_dir / f"{model_stem}_shap.png"

    plot_model_weights(estimator, weights_path)
    plot_shap(
        estimator,
        X_val_scaled,
        feature_names,
        shap_path,
        X_display=X_val_imputed,
    )
    return weights_path, shap_path

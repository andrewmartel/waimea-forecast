"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from waimea_forecast.features.engineering import build_features, prepare_supervised
from waimea_forecast.config import TARGET_COLUMN


@pytest.fixture
def sample_df():
    n = 30
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "air_temp_51000h": 22 + np.random.randn(n) * 2,
        "wave_height_51000h": 3 + np.random.rand(n) * 2,
        "wave_height_51101h": 2 + np.random.rand(n) * 1.5,
        TARGET_COLUMN: 2 + np.random.rand(n) * 1.5,
    })


def test_build_features_returns_no_nan_in_feature_columns(sample_df):
    featurized, state = build_features(sample_df)
    feature_cols = [c for c in featurized.columns if c not in ("date", TARGET_COLUMN) and pd.api.types.is_numeric_dtype(featurized[c])]
    X = featurized[feature_cols]
    assert X.notna().all().all(), "Feature matrix should have no NaNs after imputation"


def test_build_features_adds_lags_and_calendar(sample_df):
    featurized, _ = build_features(sample_df)
    assert any("lag" in c for c in featurized.columns)
    assert "day_of_year" in featurized.columns or "month_sin" in featurized.columns


def test_prepare_supervised_shapes(sample_df):
    featurized, feat_state = build_features(sample_df)
    X_train, y_train, X_val, y_val, feature_cols, state = prepare_supervised(featurized, feat_state, validation_fraction=0.2)
    assert len(X_train) + len(X_val) <= len(featurized)
    assert X_train.shape[1] == len(feature_cols)
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert state["feature_column_order"] == feature_cols


def test_prepare_supervised_no_nan_in_X(sample_df):
    featurized, feat_state = build_features(sample_df)
    X_train, y_train, X_val, y_val, _, _ = prepare_supervised(featurized, feat_state)
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()

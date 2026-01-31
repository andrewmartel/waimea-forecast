"""Tests for WaveHeightEstimator."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from waimea_forecast.config import TARGET_COLUMN
from waimea_forecast.models.estimator import WaveHeightEstimator


@pytest.fixture
def tiny_df():
    n = 50
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    np.random.seed(123)
    return pd.DataFrame({
        "date": dates,
        "air_temp_51000h": 22 + np.random.randn(n) * 2,
        "wave_height_51000h": 3 + np.random.rand(n) * 2,
        "wave_height_51101h": 2 + np.random.rand(n) * 1.5,
        TARGET_COLUMN: 2 + np.random.rand(n) * 1.5,
    })


def test_estimator_fit_predict(tiny_df):
    est = WaveHeightEstimator(validation_fraction=0.2)
    est.fit(tiny_df)
    preds = est.predict(tiny_df)
    assert preds.shape == (len(tiny_df),)
    assert np.isfinite(preds).all()


def test_estimator_save_load_roundtrip(tiny_df):
    est = WaveHeightEstimator(validation_fraction=0.2)
    est.fit(tiny_df)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "artifact.joblib"
        est.save(path)
        assert path.exists()
        loaded = WaveHeightEstimator.load(path)
        pred_orig = est.predict(tiny_df)
        pred_loaded = loaded.predict(tiny_df)
        np.testing.assert_array_almost_equal(pred_orig, pred_loaded)


def test_estimator_predict_before_fit_raises():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-01-01"]), TARGET_COLUMN: [2.5]})
    est = WaveHeightEstimator()
    with pytest.raises(ValueError, match="not fitted"):
        est.predict(df)


def test_estimator_load_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        WaveHeightEstimator.load("/nonexistent/artifact.joblib")

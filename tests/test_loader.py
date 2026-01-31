"""Tests for data loader."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from waimea_forecast.data.loader import load_wide, validate_schema
from waimea_forecast.config import TARGET_COLUMN


def test_load_wide_parses_date_and_na():
    csv = """date,air_temp_51000h,wave_height_51201h
2010-01-01,23.5,2.2
2010-01-02,22.1,NA
2010-01-03,24.0,3.1"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv)
        path = f.name
    try:
        df = load_wide(path)
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["wave_height_51201h"].iloc[1] is pd.NA or (isinstance(df["wave_height_51201h"].iloc[1], float) and pd.isna(df["wave_height_51201h"].iloc[1]))
        assert len(df) == 3
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_wide_missing_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_wide("/nonexistent/wide.csv")


def test_validate_schema_ok():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-01-01"]), TARGET_COLUMN: [2.5]})
    validate_schema(df)


def test_validate_schema_missing_target():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-01-01"]), "other": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_missing_date():
    df = pd.DataFrame({TARGET_COLUMN: [2.5], "other": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)

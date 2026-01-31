"""Load and validate the wide-format buoy CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from waimea_forecast.config import TARGET_COLUMN

# Required columns for validation (target must be present)
REQUIRED_COLUMNS = [
    "date",
    TARGET_COLUMN,
]


def load_wide(data_path: str | Path) -> pd.DataFrame:
    """
    Load the wide-format buoy CSV.

    Parses 'date' as datetime and treats 'NA' as missing.

    Parameters
    ----------
    data_path : str or Path
        Path to wide.csv.

    Returns
    -------
    pd.DataFrame
        Loaded data with date column as datetime64.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        na_values=["NA", "na", "N/A", ""],
        keep_default_na=True,
    )
    df["date"] = pd.to_datetime(df["date"], utc=False)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame has required columns including the target.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in [TARGET_COLUMN, "date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

"""Central configuration for the Waimea forecast package."""

# Target variable (Waimea Bay / North Shore buoy)
TARGET_COLUMN = "wave_height_51201h"

# Default paths (relative to repo root or cwd)
DEFAULT_DATA_PATH = "data/wide.csv"
DEFAULT_ARTIFACT_PATH = "models/artifact.joblib"

# Forecast horizon: next-day prediction (1 day ahead)
FORECAST_HORIZON_DAYS = 1

# Train/validation split: last 20% by time for validation
VALIDATION_FRACTION = 0.2

# Artifact version for reproducibility
ARTIFACT_VERSION = "1.0"

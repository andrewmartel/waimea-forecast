# Waimea Bay Wave Forecast

Proof-of-concept package to forecast wave heights at Waimea Bay (North Shore, Oahu) for the World Surf League, so they can schedule a big wave contest when conditions reach at least 3 m.

**Target variable:** `wave_height_51201h` (North Shore buoy).  
**Data:** Daily buoy metrics in wide format (`wide.csv`).

---

## Part 1: Open Questions

### 1. Framing the Problem

Before building, I would ask the WSL on a kick-off call:

- **Definition of “large enough”:** Confirm the 3 m threshold (or different) and whether it refers to significant wave height, max height, or another metric.
- **Lead time:** How far ahead do they need the forecast (1 day, 3 days, 1 week)? This drives model choice and feature design.
- **Decision use:** Is the output a go/no-go date, a probability of contest-ready conditions, or a full time series of predicted heights?
- **Update frequency:** Daily retrains vs. on-demand vs. weekly.
- **Risk tolerance:** How to balance false positives (schedule and cancel) vs. false negatives (miss a good window).
- **Historical context:** Any past contest dates or preferred windows (e.g., winter swell season) to use for validation or targeting.

### 2. Enrichment

Additional data sources to complement the buoys:

- **NOAA / NWS marine forecasts:** Text or gridded wave/wind forecasts and swell direction.
- **Satellite and model data:** Wind fields, swell propagation (e.g., from CDIP, NOAA NDBC, or Copernicus).
- **Other Pacific buoys:** More NDBC buoys for swell direction and period upstream of Hawaii.
- **Seasonal/climate indices:** ENSO, PNA, etc., for seasonal conditioning.

**Ongoing use:** Ingest via APIs or scheduled batch jobs; store in a small data lake or DB; refresh daily and feed the same feature pipeline used in this PoC.

### 3. Data Integrity

- **Missingness:** Buoy outages and gaps are likely MAR (missing at random) or related to instrument/transmission issues. The target `wave_height_51201h` and some features have NAs.
- **Target:** For supervised learning, we use only rows where the target (next-day wave height) is observed; we drop rows with missing target when building labels.
- **Features:** We avoid look-ahead. We impute feature columns with forward fill then fallback to median (median fitted on training data only); no future information is used. For a production system, we could add model-based imputation or separate “missing” indicators and tune with cross-validation.

### 4. Approaches

Options to present to the CEO:

- **Baseline:** Persistence (tomorrow = today) or simple lagged regression — interpretable and quick to deploy.
- **Classical time series:** ARIMA/ETS — good for univariate next-day forecasts but less flexible for multiple buoys and covariates.
- **ML (this PoC):** Ridge regression on lags, other-buoy metrics, and calendar features — interpretable, stable, and a clear starting point.
- **More advanced:** Tree models (e.g. Random Forest), Prophet, or light neural models for multi-step or probabilistic forecasts.

**Recommendation:** Start with this interpretable, reproducible baseline (Ridge + lags + other buoys + calendar). Then iterate with feature selection, other models, and probabilistic outputs (e.g. probability that height ≥ 3 m) as needed.

### 5. Communication

- **Dashboard:** e.g. Streamlit or internal tool showing point forecast and “contest-ready” probability (≥ 3 m) for the next 1–7 days.
- **API:** Endpoint for WSL systems to pull forecasts and thresholds on demand.
- **Scheduled reports:** Daily email or PDF with summary and recommended windows.
- **Uncertainty:** Always show intervals or P(height ≥ 3 m) so the WSL can weigh risk when scheduling.

---
## Clone
# Clone the repository
git clone https://github.com/yourusername/waimea-forecast
cd waimea-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install

Requires Python 3.10+.

```bash
cd waimea-forecast-1
pip install -e .
```

Optional (tests):

```bash
pip install -e ".[dev]"
```

---

## Data

Place the wide-format buoy CSV as `data/wide.csv` (or set the path via `--data` or `WAIMEA_DATA_PATH`). The file must include a `date` column and `wave_height_51201h`. See `data/README.md` for how to obtain the data. The original dataset is provided already, but if you want to put in your own data you can do so as well.

---

## Train

Train the estimator and save the artifact (default: `models/artifact.joblib`):

```bash
python scripts/train.py
```

With custom paths:

```bash
python scripts/train.py --data data/wide.csv --model models/artifact.joblib
```

Or after `pip install -e .`:

```bash
waimea-train --data data/wide.csv --model models/artifact.joblib
```

The script prints the artifact path and an approximate validation MAE.

---

## Predict

Load the artifact and run predictions on a CSV with the same schema as training:

```bash
python scripts/predict.py --data data/wide.csv --model models/artifact.joblib
```

Predictions are written to stdout (one per line). To save to a file:

```bash
python scripts/predict.py --data data/wide.csv --model models/artifact.joblib --out predictions.txt
```

Or:

```bash
waimea-predict --data data/wide.csv --model models/artifact.joblib --out predictions.txt
```

---

## Design Assumptions

- **Target:** `wave_height_51201h` (next-day value).
- **Horizon:** 1 day ahead.
- **Train/validation split:** Last 20% of time for validation; no shuffling.
- **Missing data:** Rows with missing target are dropped for labels; features are imputed with forward fill then training median (no look-ahead).
- **Model:** Ridge regression on lags (1–7 days), other-buoy wave heights (1-day lags), 7-day rolling mean/std of target, and calendar (day-of-year, month sin/cos). Features are standardized before fitting.

---

## Next Steps (with more time)

- **Imputation:** Try model-based or iterative imputation; add missingness indicators.
- **Horizons:** Multi-step (3- and 7-day) and separate models or direct multi-output.
- **Probabilistic forecasts:** Quantile regression or bootstrap to output P(height ≥ 3 m) and intervals.
- **Feature selection:** Regularization path or SHAP to reduce and explain predictors.
- **External data:** Integrate NOAA/NDBC or other APIs and refresh in a pipeline.
- **Retraining:** Scheduled retrains (e.g. weekly) and versioned artifacts.
- **Deployment:** REST API and/or Streamlit dashboard for the WSL.

---

## Package layout

```
waimea_forecast/
├── config.py         # Target, paths, horizon, split
├── data/loader.py    # load_wide(), validate_schema()
├── features/engineering.py  # build_features(), prepare_supervised()
├── models/estimator.py      # WaveHeightEstimator (fit/predict/save/load)
└── cli.py            # main_train(), main_predict()
scripts/
├── train.py          # CLI: train and save artifact
└── predict.py        # CLI: load artifact and predict
tests/
├── test_loader.py
├── test_features.py
└── test_estimator.py
```

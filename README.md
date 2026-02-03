# Waimea Bay Wave Forecast

Proof-of-concept package to forecast wave heights at Waimea Bay (North Shore, Oahu) for the World Surf League, so they can schedule a big wave contest when conditions reach at least 3 m.

**Forecast horizons:** 1-, 7-, and 30-day-ahead. The **30-day-ahead** horizon supports WSL’s potential need for about a month of lead time to organize the contest and for contestants to arrange travel and lodging.

**Target variable:** `wave_height_51201h` (North Shore buoy).  
**Data:** Daily buoy metrics in wide format (`wide.csv`).

---

## Part 1: Open Questions

### 1. Framing the Problem

Before building, I would ask the WSL on a kick-off call:

- **Definition of “large enough”:** Confirm the 3 m threshold (or different) and whether it refers to significant wave height, max height, or another metric.
- **Sustained vs. peak:** Does the contest require waves to be **at or above threshold for a sustained period** (e.g. several hours or a full day), or is it enough if the **maximum** (e.g. peak in a day) reaches that height? If sustained, we may need sub-daily data or a different target (e.g. proportion of readings ≥ 3 m).
- **Timing of readings:** When does the buoy report (e.g. once per day at a fixed hour)? The PoC uses **daily** aggregates; if there are **multiple readings per day**, we could define “contest-ready” from max, mean, or fraction of readings ≥ threshold, and align forecast timing with when the WSL needs to decide.
- **Contest and decision timing:** When does the contest window open/close (e.g. single day vs. multi-day window)? When does the WSL need to commit (e.g. 24 h before)? This drives horizon and how we present P(contest-ready) (e.g. P(max ≥ 3 m) on day D vs. P(any day in next 7 ≥ 3 m)).
- **Lead time:** How far ahead do they need the forecast (1 day, 1 week, 1 month)? This drives model choice and feature design.
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
- **Weather data:** For shorter term forecasts (e.g., 1- or 7-day) to give a better update on what might be expected and allow the team to notify participants of potential changes to the event.

**Ongoing use:** Ingest via APIs or scheduled batch jobs; store in a small data lake or DB; refresh daily and feed the same feature pipeline used in this PoC.

### 3. Data Integrity

- **Missingness:** Buoy outages and gaps mostly appear in chunks (consecutive periods of missing data), which are likely related to instrument/transmission issues. The target `wave_height_51201h` and some features have NAs.
- **Target:** For supervised learning, we use only rows where the target (n-days-ahead wave height) is observed; we drop rows with missing target when building labels.
- **Features:** We avoid look-ahead. We impute feature columns with **MICE** (sklearn `IterativeImputer`) fitted on training data only; optionally use **KNN** (`imputation="knn"` on the estimator). No future information is used. The column `wave_height_21418t` is excluded from features (too much missing data). For production, we could add missingness indicators or tune imputation.

### 4. Approaches

Options to present to the CEO:

- **Baseline:** Persistence (tomorrow = today) or simple lagged regression — interpretable and quick to deploy.
- **Classical time series:** ARIMA/ETS — good for univariate next-day forecasts but less flexible for multiple buoys and covariates.
- **ML (this PoC):** Ridge regression on lags, other-buoy metrics, and calendar features — interpretable, stable, and a clear starting point.
- **More advanced:** Tree models (e.g. Random Forest), Prophet, or light neural models for multi-step or probabilistic forecasts.

**Recommendation:** Start with this interpretable, reproducible baseline (Ridge + lags + other buoys + calendar/seasonality). Then iterate with feature selection, other models, and probabilistic outputs (e.g. probability that height ≥ 3 m) as needed.

### 5. Communication

- **Dashboard:** e.g. Streamlit or internal tool showing point forecast and “contest-ready” probability (≥ 3 m) for the next 30 days.
- **API:** Endpoint for WSL systems to pull forecasts and thresholds on demand.
- **Scheduled reports:** Daily email or PDF with summary and recommended windows.
- **Uncertainty:** Always show intervals or P(height ≥ 3 m) so the WSL can weigh risk when scheduling.

---
## Clone and setup

Navigate to the directory in which you want to clone the repo and run your predictions.

# Clone the repository
```
git clone https://github.com/andrewmartel/waimea-forecast
cd waimea-forecast
```

# Create virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install

Requires Python 3.10+.

```bash
cd waimea-forecast
pip install -e .
```

Optional (tests). On Windows, this also installs `pywin32` and `pyreadline3` so pytest runs without deprecation warnings:

```bash
pip install -e ".[dev]"
```

---

## Data

Place the wide-format buoy CSV as `data/wide.csv` (or set the path via `--data` or `WAIMEA_DATA_PATH`). The file must include a `date` column and `wave_height_51201h`. See `data/README.md` for how to obtain the data. The original dataset is provided already, but if you want to put in your own data you can do so as well.

---

## Train

Train the estimator and save the artifact (default: `models/artifact.joblib`). Use **`--horizon`** to train 1-, 7-, or 30-day-ahead models (default: 1):

```bash
python scripts/train.py
```

Train a 30-day-ahead model (for month-ahead planning) and save to a separate artifact:

```bash
python scripts/train.py --data data/wide.csv --model models/artifact_30d.joblib --horizon 30
```

Train 7-day-ahead:

```bash
python scripts/train.py --data data/wide.csv --model models/artifact_7d.joblib --horizon 7
```

With custom paths (1-day default):

```bash
python scripts/train.py --data data/wide.csv --model models/artifact.joblib
```

Or after `pip install -e .`:

```bash
waimea-train --data data/wide.csv --model models/artifact.joblib
waimea-train --data data/wide.csv --model models/artifact_30d.joblib --horizon 30
```

The script prints the artifact path and a full validation report: **regression** (MAE, RMSE, MAPE, R2, bias) and **classification** (confusion matrix, accuracy, precision, recall, F1, ROC AUC) for the binary “contest-ready” threshold (≥ 3 m). Use **`--max-features K`** to keep only the top K features by absolute Ridge coefficient (drops lower-value inputs); can reduce overfitting—compare validation metrics with and without it. Use **`--max-features cv`** to choose K by validation MAE: the script tries several K values (default: 10, 20, …, 100, all), prints a table of validation MAE, picks the best K, then trains and saves with that K. **`--threshold M`** sets the contest-ready threshold and fits Platt scaling (default 3.0). **`--max-features-candidates "20 40 60 80"`** tries custom K when using `--max-features cv`. Training also saves **weights** and **SHAP** plots next to the artifact.

---

## Predict

Load the artifact and run predictions on a CSV with the same schema as training:

```bash
python scripts/predict.py --data data/wide.csv --model models/artifact.joblib
```

The **horizon** is determined by the artifact (the model was trained with that horizon). Output is a **CSV** with columns: **date**, **predicted** (wave height, m), **p_contest_ready** (Platt-calibrated P(actual ≥ threshold), or NaN if no calibrator), **actual** (observed height when available), **abs_delta**, and **correct_contest_ready** (1 if predicted and actual agree on the threshold, 0 otherwise). Rows with missing actuals are omitted. Use **`--threshold M`** to set the contest-ready threshold for the probability and correct_contest_ready (default: model’s calibrator threshold, usually 3.0). Without `--out`, the CSV is printed to stdout. To save to a file:

```bash
python scripts/predict.py --data data/wide.csv --model models/artifact.joblib --out predictions.csv
```

Or:

```bash
waimea-predict --data data/wide.csv --model models/artifact.joblib --out predictions.csv
```

---

## 60-day forecast from "today"

To get **daily predictions for the next N days** from a given "today" (e.g. for a dashboard or planning tool):

```bash
python scripts/forecast_60.py --data data/wide.csv --model models/artifact.joblib --as-of 2017-11-01 --days 60 --out forecast_60d.csv
```

Or (if installed): `waimea-forecast-60 --data data/wide.csv --model models/artifact.joblib --as-of 2017-11-01 --days 60 --out forecast_60d.csv`

Output columns: **date**, **predicted**, **p_contest_ready**. The model is run recursively (each day’s forecast is used as input for the next); a **1-day horizon** model is recommended for true daily recursion.

**Validate on the final 60 days:** Use the 61st-from-last row as "today" and compare predictions to actuals:

```bash
waimea-forecast-60 --data data/wide.csv --model models/artifact.joblib --validate-60 --out forecast_60d_validate.csv
```

The output will include **actual** and **abs_delta** for the last 60 days of the dataset.

---

## Explain (weights and SHAP)

Generate **model-weights** and **SHAP** plots from a saved artifact (e.g. after loading a model trained elsewhere):

```bash
waimea-explain --model models/artifact_30d.joblib --data data/wide.csv
```

Saves `artifact_30d_weights.png` and `artifact_30d_shap.png` in the model directory (or **`--out-dir`**).

---

## Design Assumptions

- **Target:** `wave_height_51201h` (next-day value).
- **Horizon:** N days ahead (must be 1, 7, or 30 as of now; default if not speified is 1).
- **Train/validation split:** Last 20% of time for validation; no shuffling.
- **Missing data:** Rows with missing target are dropped for labels; features are imputed with MICE (or KNN) on training data only; no look-ahead.
- **Seasonality:** Annual cycle is encoded with **day-of-year sin/cos** (smooth cyclic so Dec 31 is close to Jan 1); **month sin/cos** and raw **day_of_year** are also included. This helps capture seasonal swell patterns (e.g. North Pacific winter swell).
- **Model:** Ridge regression on lags (1–7 or 7/14/30/365 for 30d horizon), other-buoy lags, rolling mean/std, and calendar/seasonality. Features are standardized; optional **Platt scaling** outputs P(actual ≥ threshold) for a configurable contest-ready threshold (default 3 m).

---

## Next Steps (with more time)

- **Imputation:** MICE is in place; add missingness indicators or compare other strategies.
- **Horizons:** 1/7/30-day are supported; 30-day uses 7/14/30/365 lags and 30-day rolling; consider direct multi-output.
- **Probabilistic forecasts:** **Platt scaling** is in place for P(contest-ready); add quantile regression or bootstrap for full intervals.
- **Feature selection:** **`--max-features`** and **`--max-features cv`**; SHAP and weights plots saved at train time; consider more thresholds.
- **60-day forecast:** **`waimea-forecast-60`** / `scripts/forecast_60.py` with **`--validate-60`** to test on final 60 days; recursive 1d recommended.
- **Timing and granularity:** Align with WSL on **when** readings are taken and **sustained vs. max** (see Open Questions); support multiple readings per day if needed.
- **External data:** Integrate NOAA/NDBC or other APIs and refresh in a pipeline.
- **Retraining:** Scheduled retrains (e.g. weekly) and versioned artifacts.
- **Deployment:** REST API and/or Streamlit dashboard for the WSL.

---

## Package layout

```
waimea_forecast/
├── config.py           # Target, paths, horizons (1/7/30d), split, threshold
├── data/loader.py      # load_wide(), validate_schema()
├── features/engineering.py  # build_features(), prepare_supervised()
├── forecast.py         # predict_next_n_days() for 60-day-from-today
├── explain.py          # plot_model_weights(), plot_shap(), export_weights_and_shap()
├── metrics.py          # regression_metrics, classification_metrics, format_validation_report
├── models/estimator.py # WaveHeightEstimator (fit, predict, predict_with_proba, Platt scaling)
├── tuning.py           # select_max_features_cv()
└── cli.py              # main_train, main_predict, main_explain, main_forecast_60
scripts/
├── train.py            # Train and save artifact (weights + SHAP)
├── predict.py          # Predict with point forecast and p_contest_ready
├── forecast_60.py      # Next N days from --as-of or --validate-60
└── confusion_k5.py    # Confusion matrices for max_features=5 (example)
tests/
├── test_loader.py
├── test_features.py
└── test_estimator.py
```

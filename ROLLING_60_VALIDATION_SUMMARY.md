# Rolling 60-day validation summary

**Setup:** As-of date **2017-11-01**; forecast next 60 days (2017-11-02 through 2018-01-01). Actuals from last 60 days of `wide.csv`. Data path: `waimea-forecast-1/data/wide.csv`.

## Point forecast accuracy (MAE / RMSE)

### Without tuning (default ETS/ARIMA configs)

| Model      | MAE (m) | RMSE (m) |
|-----------|---------|----------|
| **Persistence** (baseline) | 1.14 | 1.38 |
| **ETS** (default)         | 1.18 | 1.42 |
| **ARIMA** (1,0,1)         | 1.90 | 2.13 |

### With tuning (`--tune`): holdout MAE over config grid

| Model      | MAE (m) | RMSE (m) |
|-----------|---------|----------|
| **Persistence** | 1.14 | 1.38 |
| **ETS** (tuned) | 1.18 | 1.42 |
| **ARIMA** (tuned) | **0.74** | **0.97** |

- **Tuned ARIMA** is now best: MAE 0.74 m, RMSE 0.97 m (grid over order and seasonal_order; best chosen by 60-day holdout MAE).
- **Tuned ETS** stays similar to default (best in grid ≈ default).
- **Persistence** remains a strong baseline but is beaten by tuned ARIMA.

## Uncertainty intervals

- **ETS (default/tuned):** 80% coverage ~30%, 95% ~45% (intervals too narrow; use in-sample stderr).
- **ARIMA (tuned):** 80% coverage ~60%, 95% ~83% (native multi-step intervals; closer to nominal).

## Output files

- **rolling_60_validation.csv** — Baseline + ETS (no tune).
- **rolling_60_validation_tuned_ets.csv** — Baseline + ETS with `--tune`.
- **rolling_60_validation_tuned_arima.csv** — Baseline + ARIMA with `--tune` (best performer).

## How to reproduce

```bash
# Untuned
waimea-rolling-60 --data /path/to/data/wide.csv --as-of 2017-11-01 --days 60 --out rolling_60_validation.csv

# Tuned (recommended for ARIMA)
waimea-rolling-60 --data /path/to/data/wide.csv --as-of 2017-11-01 --days 60 --tune --model arima --out rolling_60_validation_tuned_arima.csv
```

# CLAUDE.md — Building Energy ML Project

## Project overview
Building energy prediction + HVAC control simulation targeting a co-op application
at BrainBox AI. Three-layer architecture: (1) energy load prediction, (2) rule-based
vs ML control policy comparison, (3) architectural feature analysis.

## Stack
- Python 3.12, Poetry for dependency management
- pandas, numpy, scikit-learn, xgboost, pytorch (phase 3)
- MLflow for experiment tracking
- FastAPI for serving (phase 4)
- Streamlit for dashboard (phase 4)
- Docker for containerization (phase 4)

## Commands
```bash
# run any script
poetry run python src/data/ingest_bdg2.py

# jupyter
poetry run jupyter notebook notebooks/

# mlflow ui
poetry run mlflow ui

# tests (when added)
poetry run pytest

# linting
poetry run ruff check .
poetry run ruff format .
```

## Project structure
building-energy-ml/
├── data/
│   ├── raw/                    # original downloads, never modified
│   └── processed/              # outputs of pipeline scripts
├── notebooks/                  # EDA and exploration only, not production code
├── src/
│   ├── data/
│   │   ├── ingest_bdg2.py      # loads electricity_cleaned.csv, filters NA buildings
│   │   ├── ingest_weather.py   # loads weather.csv from BDG2 (not external API)
│   │   └── merge.py            # joins energy + weather per site_id, adds time/degree features
│   ├── features/
│   │   └── build_features.py   # lag features, rolling stats, building metadata encoding
│   ├── models/
│   │   ├── train_baseline.py   # linear regression, random forest, xgboost with MLflow
│   │   └── train_lstm.py       # PyTorch LSTM, 24h multi-step (phase 3)
│   ├── simulation/
│   │   └── hvac_sim.py         # rule-based vs weather-adaptive HVAC policies (phase 4)
│   ├── visualization/
│   │   └── plot_results.py     # simulation + XGBoost figures → reports/figures/ (phase 4)
│   └── analysis/
│       └── shap_analysis.py    # TreeSHAP + per-building error vs building traits (phase 5)
├── reports/figures/            # generated PNGs (sim_*, xgb_*, shap_*, arch_*)
├── mlruns/                     # MLflow tracking data, gitignored
├── .env                        # local env vars, gitignored
├── .env.example                # committed template
├── pyproject.toml
└── README.md

## Data
- **Source:** Building Data Genome Project 2 (Kaggle)
- **Meter data:** `data/raw/electricity_cleaned.csv` — wide format, columns are building IDs
- **Weather:** `data/raw/weather.csv` — site-matched, already aligned to BDG2 date range
- **Metadata:** `data/raw/metadata.csv` — building characteristics
- **Key columns in metadata:** `building_id`, `site_id`, `primaryspaceusage`, `sqm`,
  `yearbuilt`, `timezone`, `numberoffloors`
- **Processed output:** `data/processed/merged.parquet` — 7.6M rows, 435 buildings,
  2016–2017 hourly
- **Features output:** `data/processed/features.parquet` — 7.4M rows, 25 columns,
  modeling-ready

## Data schema decisions
- Filtered to `US/Eastern` and `US/Central` timezones only
- Filtered to `Office` and `Education` primary use only
- All timestamps stored as UTC timezone-aware
- Buildings with <85% coverage dropped
- `quality_flag = suspect_zero` marks zero readings during business hours in winter
- Weather joined per `site_id` using `merge_asof` with 1h tolerance

## Feature set
```python
FEATURE_COLS = [
    # time
    "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
    # weather
    "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
    # lags (most predictive features)
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    # rolling
    "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
    # building
    "use_type_encoded", "log_sqm", "building_age",
]
TARGET = "meter_reading"  # kWh, electricity only
```

## Train/val/test split
Time-based, never random — avoids data leakage:
- **Train:** 2016-01-01 → 2017-08-30
- **Val:** 2017-08-30 → 2017-09-30 (early stopping signal for XGBoost)
- **Test:** 2017-09-30 → 2017-12-31 (held out, touched once at evaluation)

## Current model results (test set, Oct–Dec 2017)
| Model             | RMSE   | MAE   | R²     | MAPE   | Task                  |
|-------------------|--------|-------|--------|--------|-----------------------|
| Linear Regression | 85.41  | 19.21 | 0.9274 | —      | 1-step ahead          |
| Random Forest     | 67.05  | 7.88  | 0.9552 | —      | 1-step ahead          |
| XGBoost           | 70.38  | 9.90  | 0.9507 | 7.18%  | 1-step ahead          |
| LSTM              | 101.35 | 20.20 | —      | 12.45% | 24-step ahead (multi) |

MLflow experiment: `building-energy-baseline`
XGBoost best iteration: 1126 trees, val RMSE 48.58
LSTM: 2-layer (hidden=128), 30 epochs, stride=24, SEQ_LEN=168, HORIZON=24
- Val loss still dropping at epoch 30 — not fully converged
- RMSE gap vs XGBoost is partially task-driven (multi-step is harder)
- XGBoost's explicit lag features give it a structural advantage on tabular data
- Best checkpoint: `data/lstm_best.pt`

## Phase 5 — SHAP & architectural findings (test set, Oct–Dec 2017)
Script `src/analysis/shap_analysis.py`. SHAP via XGBoost native TreeSHAP
(`pred_contribs`) on a 50k-row test sample; per-building error via point predictions
on all 435 buildings. Outputs: `reports/figures/{shap_*,arch_*}.png`,
`data/processed/building_error_analysis.parquet`, MLflow run `shap_architectural_analysis`.

"Hardness" metric is **CV(RMSE) = RMSE / mean load**, not raw RMSE — raw RMSE just
tracks meter size, which would make "big buildings are harder" a trivial magnitude artifact.

Headline numbers:
- Top feature overall: `load_lag_1h` (mean-abs SHAP ≈ 137 kWh) — model is ~autoregressive.
- **Architectural features = only 1.7%** of total attribution (size + age + use type).
- Top architectural feature: `log_sqm` (mean-abs SHAP 2.74 kWh).
- Size vs CV(RMSE): Spearman ρ = **−0.23** (p=1.6e-6) — significant.
- Age vs CV(RMSE): ρ = +0.09 (p=0.33, n=131 real year-built) — null.
- Median CV(RMSE) / MAPE: Education 0.091 / 5.3%, Office 0.099 / 6.6%.

Domain reasoning ("characteristic X is harder because Y"):
- **Smaller buildings are harder.** A large building's meter sums many independent
  zones/occupants, so idiosyncratic swings average out (law of large numbers) into a
  smooth, highly autocorrelated curve the lags predict almost perfectly. Small buildings
  are driven by a few discrete events (one RTU cycling, one tenant) → spikier, less
  self-similar hour-to-hour → worse *relative* error. Size changes predictability, not
  what drives the prediction.
- **Offices modestly harder than Education.** Schools/universities run rigid academic +
  bell schedules (strong daily/weekly periodicity the 24h/168h lags capture); offices
  carry more aperiodic plug-load and variable occupancy.
- **Age: no significant effect**, and `yearbuilt` is missing for 70% of buildings.
  Likely underpowered and confounded — `yearbuilt` ignores retrofits, so a renovated
  1960s building behaves new. Reported as a null, not spun. `numberoffloors` (4% coverage)
  was too sparse to use.
- **Weather degree-days (`hdd`/`cdd`) have near-zero SHAP** — the recent-load lags already
  encode the weather-driven component, so explicit degree-days add little once lags exist.

Key reframing: architectural features barely move the *point forecast* (1.7%), but they
predict *which buildings the model struggles with* (size→error is significant). The
architectural value is in error stratification, not the forecast itself — actionable for
control: trust forecasts on large, regular-schedule buildings; widen control margins on
small / office-type ones.

## MLflow notes
- Tracking URI: `mlruns/` (local)
- Experiment: `building-energy-baseline`
- XGBoost model saved as `xgboost_model.json` via `model.get_booster().save_model()`
  (not `mlflow.xgboost.log_model` — broken with current xgboost/mlflow versions)
- sklearn models logged with `mlflow.sklearn.log_model`
- Phase 5 run `shap_architectural_analysis` logs per-feature mean-abs SHAP, the
  architectural SHAP share, size/age error correlations, and the figures as artifacts

## Known issues and workarounds
- `mlflow.xgboost.log_model` throws `TypeError: _estimator_type undefined` —
  use `model.get_booster().save_model("xgboost_model.json")` +
  `mlflow.log_artifact()` instead
- Environment Canada weather API now returns HTML instead of CSV —
  use `data/raw/weather.csv` from BDG2 directly
- BDG2 metadata uses `primaryspaceusage` not `primary_use`,
  and `site_id` is a string name (e.g. "Moose") not a numeric ID
- Pandas `pd.Timedelta("1H")` raises FutureWarning — use `"1h"` lowercase
- `to_period("M")` fails on timezone-aware timestamps —
  use `dt.strftime("%Y-%m")` instead
- `shap` package won't install: shap ≥0.52 requires numpy≥2 (project pins numpy<2),
  and older shap pulls `numba`→`llvmlite`, which has no wheel for this macOS/Python and
  fails to compile. Workaround: XGBoost native TreeSHAP
  (`booster.predict(dm, pred_contribs=True)`) — same TreeSHAP algorithm, zero extra deps.
  Booster json doesn't store feature names, so set `booster.feature_names = FEATURE_COLS`.

## Next steps
- [x] Phase 3: PyTorch LSTM for multi-step forecasting
- [x] Phase 3: Compare LSTM vs XGBoost on same test set
- [x] Phase 4: Rule-based vs ML HVAC control simulation
- [x] Phase 5: SHAP analysis of architectural features
- [ ] Phase 6: FastAPI endpoint + Streamlit dashboard + Docker

## Architecture domain context
The architectural feature analysis (phase 5) is the project's unique angle —
correlating model error with building characteristics (age, size, use type) using
SHAP values. This is where the owner's architecture background produces insights
a pure CS approach would miss. Findings should be framed as: "buildings with
characteristic X are harder to model because Y" with domain reasoning.

## Style conventions
- Type hints on all function signatures
- Docstrings on public functions
- No hardcoded paths — all via `Path` + `.env`
- Parquet for all processed data, never CSV
- Commits follow conventional commits: `feat:`, `fix:`, `chore:`, `docs:`
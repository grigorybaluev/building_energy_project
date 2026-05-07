import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
    "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
    "use_type_encoded", "log_sqm", "building_age",
]
TARGET = "meter_reading"


def time_based_split(df: pd.DataFrame,
                     test_months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time — last N months as test set.
    Never split randomly for time series — you'd leak future data into training.
    """
    split_date = df["timestamp"].max() - pd.DateOffset(months=test_months)
    train = df[df["timestamp"] <= split_date].copy()
    test  = df[df["timestamp"] >  split_date].copy()
    print(f"  train: {len(train):,} rows ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"  test:  {len(test):,} rows  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
    return train, test


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # exclude readings below 1st percentile — near-zeros blow up MAPE
    threshold = np.percentile(y_true, 1)
    mask = y_true > max(threshold, 1.0)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def log_metrics(metrics: dict[str, float], prefix: str = "test") -> None:
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}_{k}", v)


def train_linear(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model",   RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train, X_val_ext=None, y_val_ext=None) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=3000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
    )

    # use externally passed val set if provided, else fall back to last 10%
    if X_val_ext is not None:
        eval_set = [(X_val_ext, y_val_ext)]
    else:
        val_size = int(len(X_train) * 0.1)
        eval_set = [(X_train[-val_size:], y_train[-val_size:])]

    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)
    print(f"  best iteration: {model.best_iteration}")
    print(f"  best val RMSE:  {model.best_score:.4f}")
    return model


if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("building-energy-baseline")

    print("Loading features...")
    df = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"  {len(df):,} rows, {df['building_id'].nunique()} buildings")

    print("Splitting...")
    # test = last 3 months (Oct–Dec 2017)
    # val  = month before test (Sep 2017) — same seasonal window
    # train = everything before val
    test_start = df["timestamp"].max() - pd.DateOffset(months=3)
    val_start  = test_start - pd.DateOffset(months=1)

    train_df = df[df["timestamp"] <  val_start].copy()
    val_df   = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)].copy()
    test_df  = df[df["timestamp"] >= test_start].copy()

    print(f"  train: {len(train_df):,} rows ({train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()})")
    print(f"  val:   {len(val_df):,} rows  ({val_df['timestamp'].min().date()} → {val_df['timestamp'].max().date()})")
    print(f"  test:  {len(test_df):,} rows  ({test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()})")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[TARGET].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET].values

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # models = {
    #     "linear_regression": (train_linear,        mlflow.sklearn.log_model),
    #     "random_forest":     (train_random_forest,  mlflow.sklearn.log_model),
    #     "xgboost":           (train_xgboost,        mlflow.xgboost.log_model),
    # }
    # replace the models dict
    models = {
        "linear_regression": (train_linear,       mlflow.sklearn.log_model),
        "random_forest":     (train_random_forest, mlflow.sklearn.log_model),
        "xgboost":           (train_xgboost,       None),  # handled separately
    }

    results = {}

    for name, (train_fn, log_fn) in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("features", FEATURE_COLS)

            if name == "xgboost":
                model = train_fn(X_train, y_train, X_val, y_val)
            else:
                model = train_fn(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            log_metrics(metrics)
            results[name] = metrics

            if name == "xgboost":
                model.get_booster().save_model("xgboost_model.json")
                mlflow.log_artifact("xgboost_model.json", artifact_path=name)
            else:
                log_fn(model, artifact_path=name)

            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE:  {metrics['mae']:.2f}")
            print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")



    # summary table
    print("\n" + "="*55)
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
    print("-"*55)
    for name, m in results.items():
        print(f"{name:<22} {m['rmse']:>8.2f} {m['mae']:>8.2f} {m['r2']:>8.4f} {m['mape']:>7.2f}%")
    print("="*55)

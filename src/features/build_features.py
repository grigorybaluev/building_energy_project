import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import os

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))


def add_lag_features(df: pd.DataFrame) -> None:
    """Adds lag columns in-place. Assumes df is sorted by (building_id, timestamp)."""
    grp = df.groupby("building_id", observed=True)["meter_reading"]
    for lag in [1, 24, 168]:
        df[f"load_lag_{lag}h"] = grp.shift(lag)


def add_rolling_features(df: pd.DataFrame) -> None:
    """Adds rolling columns in-place via pre-allocated arrays — no full-frame copies."""
    n = len(df)
    arr_mean_24  = np.full(n, np.nan, dtype="float32")
    arr_std_24   = np.full(n, np.nan, dtype="float32")
    arr_mean_168 = np.full(n, np.nan, dtype="float32")

    # one shared read of a single column — avoids copying the full frame
    readings = df["meter_reading"].values

    for _, idx in df.groupby("building_id", sort=False, observed=True).groups.items():
        pos = idx.values  # integer positions — valid because df has RangeIndex after reset_index
        s = pd.Series(readings[pos])
        shifted = s.shift(1)
        arr_mean_24[pos]  = shifted.rolling(24,  min_periods=12).mean().values
        arr_std_24[pos]   = shifted.rolling(24,  min_periods=12).std().values
        arr_mean_168[pos] = shifted.rolling(168, min_periods=84).mean().values

    df["load_rolling_mean_24h"]  = arr_mean_24
    df["load_rolling_std_24h"]   = arr_std_24
    df["load_rolling_mean_168h"] = arr_mean_168


def add_building_features(df: pd.DataFrame) -> None:
    """Adds building metadata columns in-place."""
    le = LabelEncoder()
    df["use_type_encoded"] = le.fit_transform(df["primaryspaceusage"].astype(str))
    df["log_sqm"] = np.log1p(df["sqm"].fillna(df["sqm"].median()))
    df["building_age"] = df["timestamp"].dt.year - df["yearbuilt"].fillna(df["yearbuilt"].median())


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "meter_reading",
        "building_id", "timestamp", "site_id",
        "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
        "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
        "load_lag_1h", "load_lag_24h", "load_lag_168h",
        "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
        "use_type_encoded", "log_sqm", "building_age",
        "primaryspaceusage", "quality_flag",
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"  warning: missing columns {missing}")

    df = df[available].copy()

    before = len(df)
    df = df[df["meter_reading"].notna()]
    df = df[df["quality_flag"] == "ok"]
    after = len(df)
    print(f"  dropped {before - after:,} rows (null target or quality flag)")

    lag_cols = ["load_lag_1h", "load_lag_24h", "load_lag_168h"]
    df = df.dropna(subset=lag_cols, how="all")
    print(f"  dropped {after - len(df):,} rows with all lags null")

    return df


LOAD_COLS = [
    "timestamp", "building_id", "site_id",
    "meter_reading", "quality_flag",
    "primaryspaceusage", "sqm", "yearbuilt",
    "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
    "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
]

if __name__ == "__main__":
    print("Loading merged data...")
    df = pd.read_parquet(PROCESSED / "merged.parquet", columns=LOAD_COLS)
    print(f"  {len(df):,} rows, {df['building_id'].nunique()} buildings")

    # string columns → category (435/few unique values each)
    for col in ("building_id", "site_id", "primaryspaceusage", "quality_flag"):
        df[col] = df[col].astype("category")

    # downcast before any feature computation — halves memory for float columns
    df["meter_reading"] = df["meter_reading"].astype("float32")

    print("Sorting by building + timestamp...")
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    print("Adding lag features...")
    add_lag_features(df)

    print("Adding rolling features...")
    add_rolling_features(df)

    print("Adding building features...")
    add_building_features(df)

    print("Selecting and cleaning...")
    df = select_and_clean(df)

    print(f"\nFinal shape: {df.shape}")
    print(f"Buildings: {df['building_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nNull counts per feature:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])

    out = PROCESSED / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved → {out}")

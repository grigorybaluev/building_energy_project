import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import os

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sort within each building so lags are computed correctly
    df = df.sort_values(["building_id", "timestamp"])

    for lag in [1, 24, 168]:
        df[f"load_lag_{lag}h"] = (
            df.groupby("building_id")["meter_reading"]
            .shift(lag)
        )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["building_id", "timestamp"])

    grp = df.groupby("building_id")["meter_reading"]

    # min_periods avoids NaN for the first window — use at least 12hrs of data
    df["load_rolling_mean_24h"] = grp.transform(
        lambda x: x.shift(1).rolling(24, min_periods=12).mean()
    )
    df["load_rolling_std_24h"] = grp.transform(
        lambda x: x.shift(1).rolling(24, min_periods=12).std()
    )
    df["load_rolling_mean_168h"] = grp.transform(
        lambda x: x.shift(1).rolling(168, min_periods=84).mean()
    )

    return df


def add_building_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # label encode building use type
    le = LabelEncoder()
    df["use_type_encoded"] = le.fit_transform(df["primaryspaceusage"].fillna("Unknown"))

    # log-transform sqm — building sizes are right-skewed
    df["log_sqm"] = np.log1p(df["sqm"].fillna(df["sqm"].median()))

    # building age at time of reading
    df["building_age"] = df["timestamp"].dt.year - df["yearbuilt"].fillna(
        df["yearbuilt"].median()
    )

    return df


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        # target
        "meter_reading",
        # identifiers (not used as features but needed for splits)
        "building_id", "timestamp", "site_id",
        # time
        "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
        # weather
        "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
        # lag
        "load_lag_1h", "load_lag_24h", "load_lag_168h",
        # rolling
        "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
        # building
        "use_type_encoded", "log_sqm", "building_age",
        # keep for analysis later
        "primaryspaceusage", "quality_flag",
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"  warning: missing columns {missing}")

    df = df[available].copy()

    # drop rows where target is null or flagged
    before = len(df)
    df = df[df["meter_reading"].notna()]
    df = df[df["quality_flag"] == "ok"]
    after = len(df)
    print(f"  dropped {before - after:,} rows (null target or quality flag)")

    # drop rows where ALL lag features are null
    # (first 168 hours per building will have some nulls — that's expected)
    lag_cols = ["load_lag_1h", "load_lag_24h", "load_lag_168h"]
    df = df.dropna(subset=lag_cols, how="all")
    print(f"  dropped {after - len(df):,} rows with all lags null")

    return df


if __name__ == "__main__":
    print("Loading merged data...")
    df = pd.read_parquet(PROCESSED / "merged.parquet")
    print(f"  {len(df):,} rows, {df['building_id'].nunique()} buildings")

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding building features...")
    df = add_building_features(df)

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

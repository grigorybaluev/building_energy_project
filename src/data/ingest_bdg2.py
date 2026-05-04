import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

RAW = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

NA_TIMEZONES = {"US/Eastern", "US/Central"}
TARGET_USES = {"Office", "Education"}
MIN_COVERAGE = 0.85


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(RAW / "metadata.csv")
    mask = (
        meta["timezone"].isin(NA_TIMEZONES) &
        meta["primaryspaceusage"].isin(TARGET_USES)
    )
    return meta[mask].copy()


def load_meter_wide(filename: str, building_ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW / filename, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    # keep only buildings we care about that exist in this file
    available = [b for b in building_ids if b in df.columns]
    missing = set(building_ids) - set(available)
    if missing:
        print(f"  {len(missing)} buildings not in {filename}")
    return df[available]


def wide_to_long(df: pd.DataFrame, meter_type: str) -> pd.DataFrame:
    long = df.stack(future_stack=True).reset_index()
    long.columns = ["timestamp", "building_id", "meter_reading"]
    long["meter_type"] = meter_type
    return long


def filter_coverage(df: pd.DataFrame) -> pd.DataFrame:
    total_hours = df["timestamp"].nunique()
    coverage = (
        df.groupby("building_id")["meter_reading"]
        .apply(lambda x: x.notna().sum() / total_hours)
    )
    good = coverage[coverage >= MIN_COVERAGE].index
    dropped = coverage[coverage < MIN_COVERAGE].index.tolist()
    if dropped:
        print(f"  dropping {len(dropped)} buildings below {MIN_COVERAGE:.0%} coverage")
    return df[df["building_id"].isin(good)].copy()


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quality_flag"] = "ok"
    is_winter = df["timestamp"].dt.month.isin([12, 1, 2])
    is_business = df["timestamp"].dt.hour.between(8, 18)
    is_zero = df["meter_reading"] == 0
    df.loc[is_zero & is_winter & is_business, "quality_flag"] = "suspect_zero"
    return df


if __name__ == "__main__":
    print("Loading metadata...")
    meta = load_metadata()
    print(f"  {len(meta)} buildings selected")
    building_ids = meta["building_id"].tolist()

    print("Loading electricity meter...")
    elec_wide = load_meter_wide("electricity_cleaned.csv", building_ids)
    df = wide_to_long(elec_wide, "electricity")

    print("Filtering by coverage...")
    df = filter_coverage(df)

    print("Flagging quality issues...")
    df = flag_quality(df)

    # attach metadata columns we'll need later
    meta_slim = meta[["building_id", "primaryspaceusage", "sqm",
                       "yearbuilt", "timezone", "numberoffloors"]].copy()
    df = df.merge(meta_slim, on="building_id", how="left")

    out = PROCESSED / "bdg2_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")
    print(f"Buildings: {df['building_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

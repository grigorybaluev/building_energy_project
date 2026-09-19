import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
RAW = Path(os.getenv("DATA_RAW_DIR", "data/raw"))


def check_date_overlap(energy: pd.DataFrame, weather: pd.DataFrame) -> None:
    e_min, e_max = energy["timestamp"].min(), energy["timestamp"].max()
    w_min, w_max = weather["timestamp"].min(), weather["timestamp"].max()
    print(f"  energy:  {e_min} → {e_max}")
    print(f"  weather: {w_min} → {w_max}")
    overlap_start = max(e_min, w_min)
    overlap_end = min(e_max, w_max)
    if overlap_start >= overlap_end:
        raise ValueError("No date overlap between energy and weather data — check timezones")
    print(f"  overlap: {overlap_start} → {overlap_end}")


def add_time_features(df: pd.DataFrame) -> None:
    """Adds time columns in-place."""
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int32")
    df["day_of_week"] = ts.dt.dayofweek.astype("int32")
    df["month"] = ts.dt.month.astype("int32")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_business_hours"] = (
        df["hour"].between(8, 18) & ~df["is_weekend"].astype(bool)
    ).astype("int8")


def add_degree_days(df: pd.DataFrame,
                    heating_base: float = 18.0,
                    cooling_base: float = 18.0) -> None:
    """Adds HDD/CDD columns in-place."""
    df["hdd"] = (heating_base - df["temp_c"]).clip(lower=0).astype("float32")
    df["cdd"] = (df["temp_c"] - cooling_base).clip(lower=0).astype("float32")


if __name__ == "__main__":
    print("Loading energy data...")
    energy = pd.read_parquet(PROCESSED / "bdg2_clean.parquet")
    for col in ("building_id", "meter_type", "quality_flag", "primaryspaceusage", "timezone"):
        energy[col] = energy[col].astype("category")
    energy["meter_reading"] = energy["meter_reading"].astype("float32")
    print(f"  {len(energy):,} rows, {energy['building_id'].nunique()} buildings")

    print("Loading weather data...")
    weather = pd.read_parquet(PROCESSED / "weather_clean.parquet")
    weather["site_id"] = weather["site_id"].astype("category")
    print(f"  {len(weather):,} rows, {weather['site_id'].nunique()} sites")

    print("Attaching site_id to energy data...")
    meta = pd.read_csv(RAW / "metadata.csv")[["building_id", "site_id"]].drop_duplicates("building_id")
    energy = energy.merge(meta, on="building_id", how="left")
    energy["site_id"] = energy["site_id"].astype("category")

    missing_site = energy["site_id"].isna().sum()
    if missing_site > 0:
        print(f"  warning: {missing_site} rows have no site_id — dropping")
        energy = energy.dropna(subset=["site_id"])

    print("Checking date overlap...")
    check_date_overlap(energy, weather)

    print("Merging weather per site...")
    sites = energy["site_id"].cat.categories
    frames = []

    for site in sites:
        e = energy[energy["site_id"] == site].sort_values("timestamp")
        w = weather[weather["site_id"] == site].sort_values("timestamp")

        if len(w) == 0:
            print(f"  no weather for site {site} — skipping {e['building_id'].nunique()} buildings")
            continue

        merged_site = pd.merge_asof(
            e, w.rename(columns={"site_id": "_site_id_weather"}),
            on="timestamp",
            tolerance=pd.Timedelta("1h"),
            direction="nearest",
        ).drop(columns=["_site_id_weather"])
        frames.append(merged_site)
        join_rate = merged_site["temp_c"].notna().mean()
        print(f"  {site}: {e['building_id'].nunique()} buildings, {join_rate:.1%} join rate")

    merged = pd.concat(frames, ignore_index=True)
    del frames, energy, weather

    overall_join = merged["temp_c"].notna().mean()
    print(f"\nOverall weather join rate: {overall_join:.1%}")
    if overall_join < 0.90:
        raise ValueError(f"Join rate {overall_join:.1%} too low — check site_id mapping")

    site_counts = merged.groupby("building_id", observed=True)["site_id"].nunique()
    bad_buildings = site_counts[site_counts > 1]
    if len(bad_buildings) > 0:
        print(f"  warning: {len(bad_buildings)} buildings still have multiple sites")
        print(bad_buildings)
    else:
        print("  all buildings have exactly 1 site")

    print("Adding time features...")
    add_time_features(merged)

    print("Adding degree days...")
    add_degree_days(merged)

    out = PROCESSED / "merged.parquet"
    merged.to_parquet(out, index=False)

    print(f"\nSaved {len(merged):,} rows → {out}")
    print(f"Buildings: {merged['building_id'].nunique()}")
    print(f"Columns: {merged.columns.tolist()}")
    print("\nSample row:")
    print(merged.iloc[0])

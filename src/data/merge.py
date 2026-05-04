import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek      # 0=Monday, 6=Sunday
    df["month"] = ts.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = (
        (df["hour"].between(8, 18)) & (~df["is_weekend"].astype(bool))
    ).astype(int)
    return df


def add_degree_days(df: pd.DataFrame,
                    heating_base: float = 18.0,
                    cooling_base: float = 18.0) -> pd.DataFrame:
    df = df.copy()
    df["hdd"] = (heating_base - df["temp_c"]).clip(lower=0)  # heating degree hours
    df["cdd"] = (df["temp_c"] - cooling_base).clip(lower=0)  # cooling degree hours
    return df


if __name__ == "__main__":
    print("Loading energy data...")
    energy = pd.read_parquet(PROCESSED / "bdg2_clean.parquet")
    print(f"  {len(energy):,} rows, {energy['building_id'].nunique()} buildings")

    print("Loading weather data...")
    weather = pd.read_parquet(PROCESSED / "weather_clean.parquet")
    print(f"  {len(weather):,} rows")

    print("Checking date overlap...")
    check_date_overlap(energy, weather)

    print("Merging...")
    energy = energy.sort_values("timestamp")
    weather = weather.sort_values("timestamp")

    merged = pd.merge_asof(
        energy,
        weather,
        on="timestamp",
        tolerance=pd.Timedelta("1h"),
        direction="nearest",
    )

    join_rate = merged["temp_c"].notna().mean()
    print(f"  overall weather join rate: {join_rate:.1%}")
    if join_rate < 0.90:
        raise ValueError(
            f"Weather join rate {join_rate:.1%} is too low — check timezone alignment"
        )

    # per-building join rate — flag any outliers
    per_building = (
        merged.groupby("building_id")["temp_c"]
        .apply(lambda x: x.notna().mean())
        .sort_values()
    )
    bad = per_building[per_building < 0.85]
    if len(bad) > 0:
        print(f"  warning: {len(bad)} buildings have <85% weather join rate:")
        print(bad)

    print("Adding time features...")
    merged = add_time_features(merged)

    print("Adding degree days...")
    merged = add_degree_days(merged)

    out = PROCESSED / "merged.parquet"
    merged.to_parquet(out, index=False)

    print(f"\nSaved {len(merged):,} rows → {out}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"\nSample row:")
    print(merged.iloc[0])

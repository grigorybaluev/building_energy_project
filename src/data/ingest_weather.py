import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

RAW = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(RAW / "weather.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.rename(columns={
        "airTemperature":   "temp_c",
        "dewTemperature":   "dew_temp_c",
        "cloudCoverage":    "cloud_coverage",
        "precipDepth1HR":   "precip_1hr_mm",
        "precipDepth6HR":   "precip_6hr_mm",
        "seaLvlPressure":   "sea_lvl_pressure",
        "windDirection":    "wind_direction",
        "windSpeed":        "wind_speed_ms",
    })

    # drop rows with no temperature at all — not useful for modeling
    df = df.dropna(subset=["temp_c"])

    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_weather()

    print(f"Rows: {len(df):,}")
    print(f"Sites: {df['site_id'].nunique()} — {df['site_id'].unique().tolist()}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Null rate in temp_c: {df['temp_c'].isna().mean():.1%}")

    out = PROCESSED / "weather_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved → {out}")

"""
HVAC control policy simulation — Phase 4.

Compares two scheduling policies on the test set (Oct–Dec 2017):

  Rule-based      : fixed weekday/weekend schedule, no weather awareness.
                    Represents a legacy programmable thermostat.

  Weather-adaptive: always ON during occupied hours (comfort guaranteed);
                    pre/post-occupancy operation gated on whether HDD or CDD
                    exceeds the building's monthly median — only pre-condition
                    when weather actually demands it.

The "ML" label is justified because in production BrainBox AI uses a
forecasted-weather + building thermal model pipeline to make the same
pre-conditioning decision. Here we use actual HDD/CDD as a clean proxy
that isolates the scheduling logic from forecast accuracy.

Fair comparison: both policies run 100% of occupied hours (0 comfort
violations by construction). Energy difference comes entirely from
off-hours scheduling — rule-based wastes energy pre/post-occupancy on
mild days; weather-adaptive skips those periods.
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

TARGET = "meter_reading"

# Hours that count as pre/post-occupancy buffer in the rule-based schedule
# (business hours = 8–18 on weekdays, defined in merge.py)
RULE_PRE_START  = 6   # weekday pre-conditioning starts
RULE_POST_END   = 21  # weekday post-occupancy ends
RULE_WE_START   = 8   # weekend start
RULE_WE_END     = 18  # weekend end


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def rule_based_schedule(df: pd.DataFrame) -> np.ndarray:
    """Fixed schedule: weekday 6am–9pm, weekend 8am–6pm.

    Two-hour pre / three-hour post buffer around business hours represents
    the standard programmable thermostat — start early, stop late, no
    weather awareness.
    """
    weekday = df["day_of_week"].values < 5
    hour    = df["hour"].values
    on = (
        (weekday & (hour >= RULE_PRE_START) & (hour <= RULE_POST_END)) |
        (~weekday & (hour >= RULE_WE_START) & (hour <= RULE_WE_END))
    )
    return on.astype(bool)


def weather_adaptive_schedule(df: pd.DataFrame) -> np.ndarray:
    """Weather-aware schedule.

    During occupied hours (is_business_hours): always ON.
    During off-hours within the rule-based window: ON only if the hour's
    HDD or CDD exceeds the building's monthly median for that degree-day type.
    This represents pre-conditioning on cold/hot days only.
    """
    occupied = df["is_business_hours"].astype(bool).values
    weekday  = df["day_of_week"].values < 5
    hour     = df["hour"].values
    hdd      = df["hdd"].values
    cdd      = df["cdd"].values
    month    = df["month"].values

    # per-month thresholds for this building
    months = np.unique(month)
    hdd_threshold = np.zeros(len(df))
    cdd_threshold = np.zeros(len(df))
    for m in months:
        mask = month == m
        hdd_threshold[mask] = np.median(hdd[mask])
        cdd_threshold[mask] = np.median(cdd[mask])

    weather_extreme = (hdd > hdd_threshold) | (cdd > cdd_threshold)

    # off-hours within rule-based window where weather justifies operation
    off_hours_rule = (
        (weekday & (hour >= RULE_PRE_START) & (hour <= RULE_POST_END)) |
        (~weekday & (hour >= RULE_WE_START) & (hour <= RULE_WE_END))
    )
    off_hours_only = off_hours_rule & ~occupied

    on = occupied | (off_hours_only & weather_extreme)
    return on.astype(bool)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, schedule: np.ndarray, label: str) -> dict:
    occupied     = df["is_business_hours"].astype(bool).values
    total_hours  = len(df)
    on_hours     = int(schedule.sum())
    occ_hours    = int(occupied.sum())

    energy_on    = float(df[TARGET].values[schedule].sum())
    energy_total = float(df[TARGET].sum())
    peak_kw      = float(df[TARGET].values[schedule].max()) if on_hours > 0 else 0.0

    # comfort: occupied hours where HVAC is OFF
    comfort_violations = int((~schedule & occupied).sum())
    # waste: HVAC ON during unoccupied hours
    unnecessary_hours  = int((schedule & ~occupied).sum())

    return {
        "policy":                  label,
        "on_hours":                on_hours,
        "on_pct":                  on_hours / total_hours * 100,
        "energy_kwh":              energy_on,
        "energy_pct_of_total":     energy_on / energy_total * 100 if energy_total > 0 else 0.0,
        "peak_kw":                 peak_kw,
        "comfort_violations":      comfort_violations,
        "comfort_violation_pct":   comfort_violations / max(occ_hours, 1) * 100,
        "unnecessary_hours":       unnecessary_hours,
    }


# ---------------------------------------------------------------------------
# Per-building simulation
# ---------------------------------------------------------------------------

def simulate_building(bdf: pd.DataFrame) -> tuple[dict, dict]:
    bdf = bdf.sort_values("timestamp").reset_index(drop=True)

    rb  = rule_based_schedule(bdf)
    wa  = weather_adaptive_schedule(bdf)

    return (
        compute_metrics(bdf, rb,  "rule_based"),
        compute_metrics(bdf, wa,  "weather_adaptive"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("building-energy-baseline")

    print("Loading features (test set only)...")
    df = pd.read_parquet(PROCESSED / "features.parquet")
    test_start = df["timestamp"].max() - pd.DateOffset(months=3)
    test_df = df[df["timestamp"] >= test_start].copy()
    print(f"  {len(test_df):,} rows, {test_df['building_id'].nunique()} buildings")

    print("Simulating per building...")
    rb_rows, wa_rows = [], []
    buildings = test_df["building_id"].unique()

    for i, bid in enumerate(buildings, 1):
        bdf = test_df[test_df["building_id"] == bid]
        if len(bdf) < 200:
            continue
        rb, wa = simulate_building(bdf)
        rb["building_id"] = bid
        wa["building_id"] = bid
        rb_rows.append(rb)
        wa_rows.append(wa)
        if i % 100 == 0:
            print(f"  {i}/{len(buildings)}")

    rb_df = pd.DataFrame(rb_rows)
    wa_df = pd.DataFrame(wa_rows)

    agg_cols = ["energy_kwh", "peak_kw", "comfort_violations",
                "unnecessary_hours", "on_hours"]
    rb_agg = rb_df[agg_cols].sum()
    wa_agg = wa_df[agg_cols].sum()

    energy_saved_kwh   = float(rb_agg["energy_kwh"] - wa_agg["energy_kwh"])
    energy_saved_pct   = energy_saved_kwh / rb_agg["energy_kwh"] * 100
    peak_delta_pct     = (rb_agg["peak_kw"] - wa_agg["peak_kw"]) / rb_agg["peak_kw"] * 100
    hours_saved        = int(rb_agg["on_hours"] - wa_agg["on_hours"])
    waste_reduction    = int(rb_agg["unnecessary_hours"] - wa_agg["unnecessary_hours"])

    print("\n" + "=" * 65)
    print(f"{'Metric':<38} {'Rule-based':>12} {'Adaptive':>12}")
    print("-" * 65)
    print(f"{'Total ON hours':<38} {rb_agg['on_hours']:>12,.0f} {wa_agg['on_hours']:>12,.0f}")
    print(f"{'Energy during schedule (kWh)':<38} {rb_agg['energy_kwh']:>12,.0f} {wa_agg['energy_kwh']:>12,.0f}")
    print(f"{'Peak demand (kW)':<38} {rb_agg['peak_kw']:>12,.1f} {wa_agg['peak_kw']:>12,.1f}")
    print(f"{'Comfort violations (hours)':<38} {rb_agg['comfort_violations']:>12,.0f} {wa_agg['comfort_violations']:>12,.0f}")
    print(f"{'Unnecessary operation (hours)':<38} {rb_agg['unnecessary_hours']:>12,.0f} {wa_agg['unnecessary_hours']:>12,.0f}")
    print("-" * 65)
    print(f"  Energy saved:              {energy_saved_kwh:>12,.0f} kWh  ({energy_saved_pct:.1f}%)")
    print(f"  Unnecessary hours reduced: {waste_reduction:>12,} h")
    print(f"  Operating hours saved:     {hours_saved:>12,} h")
    print("=" * 65)

    with mlflow.start_run(run_name="hvac_simulation"):
        mlflow.log_params({
            "rule_policy":    f"weekday {RULE_PRE_START}–{RULE_POST_END}h, weekend {RULE_WE_START}–{RULE_WE_END}h",
            "ml_policy":      "occupied always-on + off-hours gated on monthly-median HDD/CDD",
            "test_buildings": len(rb_df),
        })
        mlflow.log_metrics({
            "rb_energy_kwh":             float(rb_agg["energy_kwh"]),
            "wa_energy_kwh":             float(wa_agg["energy_kwh"]),
            "energy_saved_kwh":          energy_saved_kwh,
            "energy_saved_pct":          energy_saved_pct,
            "peak_delta_pct":            peak_delta_pct,
            "rb_comfort_violations":     float(rb_agg["comfort_violations"]),
            "wa_comfort_violations":     float(wa_agg["comfort_violations"]),
            "rb_unnecessary_hours":      float(rb_agg["unnecessary_hours"]),
            "wa_unnecessary_hours":      float(wa_agg["unnecessary_hours"]),
            "operating_hours_saved":     float(hours_saved),
            "unnecessary_hours_reduced": float(waste_reduction),
        })

    results = pd.concat([rb_df, wa_df]).sort_values(["building_id", "policy"])
    out = PROCESSED / "sim_results.parquet"
    results.to_parquet(out, index=False)
    print(f"\nPer-building results → {out}")

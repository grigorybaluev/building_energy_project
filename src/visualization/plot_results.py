"""
Generate comparison plots for Phase 4 (HVAC simulation) and
XGBoost model predictions vs actuals.

Outputs to reports/figures/:
  1. sim_summary_bar.png        — energy / hours headline comparison
  2. sim_savings_distribution.png — per-building % energy savings
  3. sim_monthly_savings.png    — monthly savings breakdown
  4. sim_by_building_type.png   — Office vs Education comparison
  5. xgb_actual_vs_pred.png     — XGBoost scatter on test set
  6. xgb_residuals_by_hour.png  — residuals by hour of day

Run:
    poetry run python src/visualization/plot_results.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
FIGURES   = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_business_hours",
    "temp_c", "dew_temp_c", "wind_speed_ms", "hdd", "cdd",
    "load_lag_1h", "load_lag_24h", "load_lag_168h",
    "load_rolling_mean_24h", "load_rolling_std_24h", "load_rolling_mean_168h",
    "use_type_encoded", "log_sqm", "building_age",
]

PALETTE = {"rule_based": "#4C72B0", "weather_adaptive": "#55A868"}
LABELS  = {"rule_based": "Rule-based", "weather_adaptive": "Weather-adaptive (ML)"}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_sim() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim = pd.read_parquet(PROCESSED / "sim_results.parquet")
    rb  = sim[sim["policy"] == "rule_based"].set_index("building_id")
    wa  = sim[sim["policy"] == "weather_adaptive"].set_index("building_id")
    return sim, rb, wa


def load_test_with_preds() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    test_start = df["timestamp"].max() - pd.DateOffset(months=3)
    test_df = df[df["timestamp"] >= test_start].copy()

    booster = xgb.Booster()
    booster.load_model("xgboost_model.json")
    X = test_df[FEATURE_COLS].values.astype("float32")
    test_df["predicted"] = np.clip(booster.predict(xgb.DMatrix(X)), 0, None)
    return test_df


# ---------------------------------------------------------------------------
# Plot 1 — simulation headline bar chart
# ---------------------------------------------------------------------------

def plot_sim_summary(rb: pd.DataFrame, wa: pd.DataFrame) -> None:
    metrics = {
        "Energy consumed\n(M kWh)":      (rb["energy_kwh"].sum() / 1e6,
                                           wa["energy_kwh"].sum() / 1e6),
        "Total ON hours\n(thousands)":    (rb["on_hours"].sum() / 1e3,
                                           wa["on_hours"].sum() / 1e3),
        "Unnecessary hours\n(thousands)": (rb["unnecessary_hours"].sum() / 1e3,
                                           wa["unnecessary_hours"].sum() / 1e3),
    }

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2,
           [v[0] for v in metrics.values()],
           width, label="Rule-based",
           color=PALETTE["rule_based"], alpha=0.9)
    ax.bar(x + width / 2,
           [v[1] for v in metrics.values()],
           width, label="Weather-adaptive (ML)",
           color=PALETTE["weather_adaptive"], alpha=0.9)

    # annotate savings
    for i, (rb_val, wa_val) in enumerate([v for v in metrics.values()]):
        saving_pct = (rb_val - wa_val) / rb_val * 100
        ax.annotate(f"−{saving_pct:.1f}%",
                    xy=(x[i] + width / 2, wa_val),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, color="#2d6a3f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylabel("Value")
    ax.set_title("HVAC Simulation: Rule-based vs Weather-adaptive (ML)\nTest Set Oct–Dec 2017, 435 Buildings",
                 fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    out = FIGURES / "sim_summary_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — per-building savings distribution
# ---------------------------------------------------------------------------

def plot_savings_distribution(rb: pd.DataFrame, wa: pd.DataFrame) -> None:
    common = rb.index.intersection(wa.index)
    savings_pct = (
        (rb.loc[common, "energy_kwh"] - wa.loc[common, "energy_kwh"])
        / rb.loc[common, "energy_kwh"] * 100
    ).dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(savings_pct, bins=40, color=PALETTE["weather_adaptive"],
            edgecolor="white", alpha=0.85)
    ax.axvline(savings_pct.median(), color="#c0392b", lw=1.8,
               linestyle="--", label=f"Median {savings_pct.median():.1f}%")
    ax.axvline(0, color="black", lw=1, linestyle=":")

    ax.set_xlabel("Energy saved vs rule-based (%)")
    ax.set_ylabel("Number of buildings")
    ax.set_title("Per-building Energy Savings Distribution\nWeather-adaptive vs Rule-based",
                 fontweight="bold")
    ax.legend()

    fig.tight_layout()
    out = FIGURES / "sim_savings_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — monthly savings
# ---------------------------------------------------------------------------

def plot_monthly_savings(test_df: pd.DataFrame,
                         rb: pd.DataFrame, wa: pd.DataFrame) -> None:
    # join building-level sim metrics with test data to get monthly breakdown
    meta = (
        test_df.groupby(["building_id", "month"])["meter_reading"]
        .sum()
        .reset_index()
        .rename(columns={"meter_reading": "actual_kwh"})
    )
    # monthly fraction per building
    bld_total = meta.groupby("building_id")["actual_kwh"].transform("sum")
    meta["monthly_frac"] = meta["actual_kwh"] / bld_total.replace(0, np.nan)

    rb_reset = rb["energy_kwh"].reset_index().rename(columns={"energy_kwh": "rb_kwh"})
    wa_reset = wa["energy_kwh"].reset_index().rename(columns={"energy_kwh": "wa_kwh"})
    meta = meta.merge(rb_reset, on="building_id").merge(wa_reset, on="building_id")

    meta["rb_monthly"]  = meta["rb_kwh"] * meta["monthly_frac"]
    meta["wa_monthly"]  = meta["wa_kwh"] * meta["monthly_frac"]
    monthly = meta.groupby("month")[["rb_monthly", "wa_monthly"]].sum() / 1e6

    month_labels = {10: "October", 11: "November", 12: "December"}
    monthly.index = [month_labels.get(m, str(m)) for m in monthly.index]

    x = np.arange(len(monthly))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(x - width / 2, monthly["rb_monthly"], width,
           label="Rule-based", color=PALETTE["rule_based"], alpha=0.9)
    ax.bar(x + width / 2, monthly["wa_monthly"], width,
           label="Weather-adaptive (ML)", color=PALETTE["weather_adaptive"], alpha=0.9)

    for i, (rb_v, wa_v) in enumerate(zip(monthly["rb_monthly"], monthly["wa_monthly"])):
        pct = (rb_v - wa_v) / rb_v * 100
        ax.annotate(f"−{pct:.1f}%",
                    xy=(x[i] + width / 2, wa_v),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, color="#2d6a3f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(monthly.index)
    ax.set_ylabel("Energy (M kWh)")
    ax.set_title("Monthly Energy Consumption by Policy\n(Oct = mild, Dec = cold → savings vary by month)",
                 fontweight="bold")
    ax.legend()

    fig.tight_layout()
    out = FIGURES / "sim_monthly_savings.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 4 — savings by building type
# ---------------------------------------------------------------------------

def plot_by_building_type(test_df: pd.DataFrame,
                          rb: pd.DataFrame, wa: pd.DataFrame) -> None:
    use_map = (
        test_df.groupby("building_id")["primaryspaceusage"]
        .first()
        .astype(str)
    )
    common = rb.index.intersection(wa.index).intersection(use_map.index)

    data = pd.DataFrame({
        "use_type":   use_map.loc[common],
        "rb_kwh":     rb.loc[common, "energy_kwh"],
        "wa_kwh":     wa.loc[common, "energy_kwh"],
        "savings_pct": (
            (rb.loc[common, "energy_kwh"] - wa.loc[common, "energy_kwh"])
            / rb.loc[common, "energy_kwh"] * 100
        ),
    }).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: total energy by type
    agg = data.groupby("use_type")[["rb_kwh", "wa_kwh"]].sum() / 1e6
    agg.plot(kind="bar", ax=axes[0], color=[PALETTE["rule_based"], PALETTE["weather_adaptive"]],
             alpha=0.9, rot=0)
    axes[0].set_title("Total Energy by Building Type (M kWh)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Energy (M kWh)")
    axes[0].legend(["Rule-based", "Weather-adaptive"])

    # right: savings % distribution by type
    sns.boxplot(data=data, x="use_type", y="savings_pct", ax=axes[1],
                palette={"Office": "#4C72B0", "Education": "#DD8452"},
                width=0.5)
    axes[1].axhline(0, color="black", lw=0.8, linestyle=":")
    axes[1].set_title("Per-building Savings % by Building Type", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Energy saved vs rule-based (%)")

    fig.tight_layout()
    out = FIGURES / "sim_by_building_type.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 5 — XGBoost actual vs predicted
# ---------------------------------------------------------------------------

def plot_xgb_actual_vs_pred(test_df: pd.DataFrame) -> None:
    sample = test_df.sample(n=min(20_000, len(test_df)), random_state=42)
    actual = sample["meter_reading"].values
    pred   = sample["predicted"].values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, pred, alpha=0.15, s=4, color="#4C72B0", rasterized=True)
    lim = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.2, label="Perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual meter reading (kWh)")
    ax.set_ylabel("XGBoost predicted (kWh)")
    ax.set_title("XGBoost: Actual vs Predicted\nTest Set Oct–Dec 2017 (20k sample)",
                 fontweight="bold")

    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    r2   = float(1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2))
    ax.text(0.05, 0.93, f"RMSE = {rmse:.1f}\nR² = {r2:.4f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend()

    fig.tight_layout()
    out = FIGURES / "xgb_actual_vs_pred.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 6 — XGBoost residuals by hour
# ---------------------------------------------------------------------------

def plot_xgb_residuals_by_hour(test_df: pd.DataFrame) -> None:
    test_df = test_df.copy()
    test_df["residual"] = test_df["predicted"] - test_df["meter_reading"]

    hourly = (
        test_df.groupby("hour")["residual"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly["hour"], hourly["mean"],
           color=["#c0392b" if v > 0 else "#2980b9" for v in hourly["mean"]],
           alpha=0.85, label="Mean residual")
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.2, color="gray", label="±1 std")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Residual (predicted − actual, kWh)")
    ax.set_title("XGBoost Residuals by Hour of Day\n(red = over-predict, blue = under-predict)",
                 fontweight="bold")
    ax.set_xticks(range(0, 24))
    ax.legend()

    fig.tight_layout()
    out = FIGURES / "xgb_residuals_by_hour.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading simulation results...")
    sim, rb, wa = load_sim()

    print("Loading test set + XGBoost predictions...")
    test_df = load_test_with_preds()

    print("Generating plots...")
    plot_sim_summary(rb, wa)
    plot_savings_distribution(rb, wa)
    plot_monthly_savings(test_df, rb, wa)
    plot_by_building_type(test_df, rb, wa)
    plot_xgb_actual_vs_pred(test_df)
    plot_xgb_residuals_by_hour(test_df)

    print(f"\nAll plots saved to {FIGURES.resolve()}/")

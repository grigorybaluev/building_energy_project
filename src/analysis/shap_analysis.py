"""
Phase 5 — SHAP attribution + architectural error analysis.

Two questions, framed for the project's architecture angle:

  1. What drives the XGBoost load predictions? (global TreeSHAP attribution)
  2. Which BUILDING CHARACTERISTICS make a building harder to model?
     (per-building error vs size / age / use type)

SHAP values come from XGBoost's native TreeSHAP — `booster.predict(..., pred_contribs=True)`
— not the `shap` package. Same TreeSHAP algorithm (Lundberg et al.), but no numba/llvmlite
build and no numpy>=2 requirement, so it stays inside the project's numpy<2 pin. TreeSHAP
satisfies additivity: per-row contributions + bias term sum exactly to the raw prediction.

Why CV(RMSE) instead of RMSE for the "hardness" metric:
    Raw RMSE scales with building size — a large building's kWh swings are bigger, so its
    RMSE is bigger no matter how well it's modelled. That would make "large buildings are
    harder" a trivial artifact of magnitude. CV(RMSE) = RMSE / mean(actual) is scale-free,
    so differences reflect genuine modelling difficulty rather than meter size.

Age caveat:
    `yearbuilt` is present for only ~30% of buildings (131/435); the model's `building_age`
    feature is median-imputed for the rest. The age-vs-error analysis is therefore restricted
    to buildings with a real `yearbuilt`. `numberoffloors` (~4% coverage) is too sparse to use.

Outputs:
    reports/figures/shap_global_importance.png
    reports/figures/shap_beeswarm.png
    reports/figures/shap_arch_dependence.png
    reports/figures/arch_error_by_use_type.png
    reports/figures/arch_error_by_size_age.png
    data/processed/building_error_analysis.parquet
    MLflow run "shap_architectural_analysis" (experiment building-energy-baseline)

Run:
    poetry run python src/analysis/shap_analysis.py
"""

import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from dotenv import load_dotenv
from matplotlib.patches import Patch
from scipy.stats import spearmanr

load_dotenv()

PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
RAW = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "xgboost_model.json"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_business_hours",
    "temp_c",
    "dew_temp_c",
    "wind_speed_ms",
    "hdd",
    "cdd",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "load_rolling_mean_24h",
    "load_rolling_std_24h",
    "load_rolling_mean_168h",
    "use_type_encoded",
    "log_sqm",
    "building_age",
]
ARCH_FEATURES = ["log_sqm", "building_age", "use_type_encoded"]
TARGET = "meter_reading"

TEST_MONTHS = 3  # test set = last 3 months (Oct–Dec 2017)
REF_YEAR = 2017  # test set year → age = REF_YEAR - yearbuilt
SHAP_COMPUTE_SAMPLE = (
    50000  # rows TreeSHAP runs on (exact TreeSHAP is O(samples·trees·leaves·depth²))
)
SHAP_PLOT_SAMPLE = 8000  # points drawn in scatter / beeswarm plots
MIN_TEST_ROWS = 200  # skip buildings with too little test data (matches hvac_sim)

ARCH_COLOR = "#C44E52"
OTHER_COLOR = "#4C72B0"
USE_PALETTE = {"Office": "#4C72B0", "Education": "#DD8452"}

sns.set_theme(style="whitegrid", font_scale=1.05)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_test_set() -> pd.DataFrame:
    """Load the held-out test slice (last TEST_MONTHS) with features + metadata."""
    cols = FEATURE_COLS + [TARGET, "building_id", "primaryspaceusage", "timestamp"]
    df = pd.read_parquet(PROCESSED / "features.parquet", columns=cols)
    cutoff = df["timestamp"].max() - pd.DateOffset(months=TEST_MONTHS)
    test = df[df["timestamp"] >= cutoff].copy()
    test["building_id"] = test["building_id"].astype(str)
    return test


def load_booster() -> xgb.Booster:
    """Load the saved booster and attach feature names (not stored in the json)."""
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    booster.feature_names = FEATURE_COLS
    return booster


def _dmatrix(X: np.ndarray) -> xgb.DMatrix:
    return xgb.DMatrix(X.astype("float32"), feature_names=FEATURE_COLS)


def compute_shap(booster: xgb.Booster, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Native TreeSHAP. Returns (shap_values [n, n_feat], expected_value)."""
    contribs = booster.predict(_dmatrix(X), pred_contribs=True)
    shap_values = contribs[:, :-1]  # last column is the bias / base value
    expected = float(contribs[:, -1].mean())
    return shap_values, expected


def predict(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Non-negative point predictions (loads can't be negative)."""
    return np.clip(booster.predict(_dmatrix(X)), 0, None)


# ---------------------------------------------------------------------------
# SHAP attribution plots
# ---------------------------------------------------------------------------


def plot_global_importance(shap_values: np.ndarray, names: list[str]) -> pd.Series:
    """Mean |SHAP| per feature; architectural features highlighted. Returns the series."""
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=names).sort_values()
    colors = [ARCH_COLOR if n in ARCH_FEATURES else OTHER_COLOR for n in mean_abs.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(mean_abs.index, mean_abs.values, color=colors, alpha=0.9)
    ax.set_xlabel("Mean |SHAP value|  (avg absolute impact on prediction, kWh)")
    ax.set_title(
        "Global Feature Importance (TreeSHAP)\nTest set Oct–Dec 2017", fontweight="bold"
    )
    ax.legend(
        handles=[
            Patch(color=ARCH_COLOR, label="Architectural feature"),
            Patch(color=OTHER_COLOR, label="Time / weather / autoregressive"),
        ],
        loc="lower right",
    )

    fig.tight_layout()
    out = FIGURES / "shap_global_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return mean_abs


def plot_beeswarm(
    shap_values: np.ndarray, X: np.ndarray, names: list[str], top_n: int = 15
) -> None:
    """Beeswarm-style summary: SHAP value (x) coloured by feature value, per feature row."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n][
        ::-1
    ]  # least → most important (bottom→top)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(10, 0.45 * top_n + 1.5))
    for row, fi in enumerate(order):
        sv = shap_values[:, fi]
        fv = X[:, fi].astype(float)
        lo, hi = np.nanpercentile(fv, 1), np.nanpercentile(fv, 99)
        color = np.clip((fv - lo) / (hi - lo + 1e-9), 0, 1)
        jitter = (rng.random(len(sv)) - 0.5) * 0.7
        ax.scatter(
            sv,
            row + jitter,
            c=color,
            cmap="coolwarm",
            s=7,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("SHAP value  (impact on predicted load, kWh)")
    ax.set_title(
        f"SHAP Summary — top {top_n} features\nColour = feature value",
        fontweight="bold",
    )

    sm = cm.ScalarMappable(cmap="coolwarm", norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("Feature value")

    fig.tight_layout()
    out = FIGURES / "shap_beeswarm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_arch_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    names: list[str],
    use_label: dict[int, str],
    building_ids: np.ndarray,
    known_year_bids: set[str],
) -> None:
    """SHAP dependence for the three architectural features."""
    idx = {n: i for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- log_sqm (continuous, all buildings) ---
    fi = idx["log_sqm"]
    axes[0].scatter(
        X[:, fi],
        shap_values[:, fi],
        s=8,
        alpha=0.3,
        color="#4C72B0",
        edgecolors="none",
        rasterized=True,
    )
    _binned_trend(axes[0], X[:, fi], shap_values[:, fi])
    axes[0].axhline(0, color="black", lw=0.8, ls=":")
    axes[0].set_xlabel("log_sqm  (log floor area)")
    axes[0].set_ylabel("SHAP value (kWh)")
    axes[0].set_title("Building size → predicted load", fontweight="bold")

    # --- building_age (continuous, real-yearbuilt buildings only) ---
    fi = idx["building_age"]
    keep = np.array([b in known_year_bids for b in building_ids])
    axes[1].scatter(
        X[keep, fi],
        shap_values[keep, fi],
        s=8,
        alpha=0.3,
        color="#55A868",
        edgecolors="none",
        rasterized=True,
    )
    if keep.sum() > 20:
        _binned_trend(axes[1], X[keep, fi], shap_values[keep, fi])
    axes[1].axhline(0, color="black", lw=0.8, ls=":")
    axes[1].set_xlabel("building_age (years)")
    axes[1].set_ylabel("SHAP value (kWh)")
    axes[1].set_title(
        f"Building age → predicted load\n(real year-built only, "
        f"{keep.sum():,} rows)",
        fontweight="bold",
    )

    # --- use_type_encoded (categorical) ---
    fi = idx["use_type_encoded"]
    dep = pd.DataFrame(
        {
            "use_type": [use_label.get(int(v), str(int(v))) for v in X[:, fi]],
            "shap": shap_values[:, fi],
        }
    )
    sns.boxplot(
        data=dep,
        x="use_type",
        y="shap",
        hue="use_type",
        legend=False,
        ax=axes[2],
        palette=USE_PALETTE,
        width=0.5,
        fliersize=1,
    )
    axes[2].axhline(0, color="black", lw=0.8, ls=":")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("SHAP value (kWh)")
    axes[2].set_title("Use type → predicted load", fontweight="bold")

    fig.suptitle("SHAP Dependence — Architectural Features", fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "shap_arch_dependence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _binned_trend(ax, x: np.ndarray, y: np.ndarray, bins: int = 10) -> None:
    """Overlay a median trend line over quantile bins of x."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < bins * 2:
        return
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    centers, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 5:
            centers.append((lo + hi) / 2)
            meds.append(np.median(y[m]))
    ax.plot(
        centers, meds, color="#c0392b", lw=2.2, marker="o", ms=4, label="binned median"
    )
    ax.legend(loc="best", fontsize=9)


# ---------------------------------------------------------------------------
# Per-building error analysis
# ---------------------------------------------------------------------------


def building_metrics(bid: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Error metrics for one building. CV(RMSE) is the scale-free 'hardness' metric."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mean_actual = float(np.mean(y_true))
    cv_rmse = rmse / mean_actual if mean_actual > 0 else np.nan

    # MAPE excluding near-zero actuals (consistent with train_baseline.compute_metrics)
    thr = max(1.0, float(np.percentile(y_true, 1)))
    mask = y_true > thr
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else np.nan
    )

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - mean_actual) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "building_id": bid,
        "n": int(len(y_true)),
        "mean_actual_kwh": mean_actual,
        "rmse": rmse,
        "mae": mae,
        "cv_rmse": cv_rmse,
        "mape": mape,
        "r2": r2,
    }


def per_building_error(test_df: pd.DataFrame, booster: xgb.Booster) -> pd.DataFrame:
    """Predict on every test row, then compute error metrics per building."""
    y_pred = predict(booster, test_df[FEATURE_COLS].values)
    work = pd.DataFrame(
        {
            "building_id": test_df["building_id"].values,
            "y": test_df[TARGET].values.astype(float),
            "yhat": y_pred,
        }
    )
    rows = [
        building_metrics(bid, sub["y"].values, sub["yhat"].values)
        for bid, sub in work.groupby("building_id", sort=False)
        if len(sub) >= MIN_TEST_ROWS
    ]
    return pd.DataFrame(rows)


def enrich_with_metadata(err_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Join raw building metadata: real sqm, yearbuilt → age, use type."""
    meta = pd.read_csv(
        RAW / "metadata.csv",
        usecols=["building_id", "sqm", "yearbuilt", "numberoffloors"],
    )
    meta["building_id"] = meta["building_id"].astype(str)

    use = (
        test_df.groupby("building_id", observed=True)["primaryspaceusage"]
        .first()
        .astype(str)
        .rename("use_type")
        .reset_index()
    )

    out = err_df.merge(meta, on="building_id", how="left").merge(
        use, on="building_id", how="left"
    )
    out["yearbuilt_known"] = out["yearbuilt"].notna()
    out["age"] = REF_YEAR - out["yearbuilt"]
    return out


# ---------------------------------------------------------------------------
# Architectural error plots
# ---------------------------------------------------------------------------


def plot_error_by_use_type(err: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in [
        (axes[0], "cv_rmse", "CV(RMSE)  = RMSE / mean load"),
        (axes[1], "mape", "MAPE (%)"),
    ]:
        sns.boxplot(
            data=err,
            x="use_type",
            y=metric,
            hue="use_type",
            legend=False,
            ax=ax,
            palette=USE_PALETTE,
            width=0.5,
            fliersize=1.5,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        for i, ut in enumerate(sorted(err["use_type"].dropna().unique())):
            med = err.loc[err["use_type"] == ut, metric].median()
            ax.annotate(
                f"median {med:.3f}" if metric == "cv_rmse" else f"median {med:.1f}%",
                xy=(i, med),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#2d2d2d",
            )
    axes[0].set_title("Prediction difficulty by use type", fontweight="bold")
    axes[1].set_title("MAPE by use type", fontweight="bold")
    fig.suptitle("Per-building Error by Building Use Type", fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES / "arch_error_by_use_type.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_error_by_size_age(err: pd.DataFrame) -> tuple[float, float, float, float]:
    """Scatter CV(RMSE) vs size and vs age. Returns spearman (rho_sqm, p_sqm, rho_age, p_age)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- size: all buildings ---
    sz = err.dropna(subset=["sqm", "cv_rmse"])
    rho_sqm, p_sqm = spearmanr(sz["sqm"], sz["cv_rmse"])
    for ut, sub in sz.groupby("use_type"):
        axes[0].scatter(
            sub["sqm"],
            sub["cv_rmse"],
            s=28,
            alpha=0.6,
            color=USE_PALETTE.get(ut, "#888"),
            label=ut,
            edgecolors="none",
        )
    # trend on raw m² (not log10) so bin centres land correctly on the log axis
    _binned_trend(axes[0], sz["sqm"].values, sz["cv_rmse"].values)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Floor area (m², log scale)")
    axes[0].set_ylabel("CV(RMSE)")
    # cap y so the bulk is readable; flag the off-scale outliers
    ycap = float(sz["cv_rmse"].quantile(0.99))
    n_off = int((sz["cv_rmse"] > ycap).sum())
    axes[0].set_ylim(0, ycap * 1.05)
    if n_off:
        axes[0].text(
            0.98,
            0.97,
            f"{n_off} building(s) off-scale (max {sz['cv_rmse'].max():.1f})",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#666",
        )
    axes[0].set_title(
        f"Size vs difficulty\nSpearman ρ = {rho_sqm:.2f} ({_fmt_p(p_sqm)})",
        fontweight="bold",
    )
    axes[0].legend(title="Use type", loc="upper left")

    # --- age: real yearbuilt only ---
    ag = err[err["yearbuilt_known"]].dropna(subset=["age", "cv_rmse"])
    rho_age, p_age = (
        spearmanr(ag["age"], ag["cv_rmse"]) if len(ag) > 5 else (np.nan, np.nan)
    )
    for ut, sub in ag.groupby("use_type"):
        axes[1].scatter(
            sub["age"],
            sub["cv_rmse"],
            s=28,
            alpha=0.6,
            color=USE_PALETTE.get(ut, "#888"),
            label=ut,
            edgecolors="none",
        )
    _binned_trend(axes[1], ag["age"].values, ag["cv_rmse"].values, bins=6)
    axes[1].set_xlabel("Building age (years)")
    axes[1].set_ylabel("CV(RMSE)")
    axes[1].set_title(
        f"Age vs difficulty (n={len(ag)} real year-built)\n"
        f"Spearman ρ = {rho_age:.2f} ({_fmt_p(p_age)})",
        fontweight="bold",
    )
    axes[1].legend(title="Use type")

    fig.suptitle(
        "Per-building Error vs Building Characteristics", fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out = FIGURES / "arch_error_by_size_age.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return float(rho_sqm), float(p_sqm), float(rho_age), float(p_age)


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "p=n/a"
    return f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("building-energy-baseline")

    print("Loading test set...")
    test_df = load_test_set()
    print(f"  {len(test_df):,} rows, {test_df['building_id'].nunique()} buildings")

    booster = load_booster()

    # Per-building error uses cheap point predictions over the FULL test set.
    print("Per-building error (full test set)...")
    err = per_building_error(test_df, booster)
    err = enrich_with_metadata(err, test_df)
    known_year_bids = set(err.loc[err["yearbuilt_known"], "building_id"])
    print(f"  {len(err)} buildings scored; {len(known_year_bids)} with real year-built")

    # TreeSHAP is far costlier than prediction, so run it on a random sample of the
    # test set — global mean|SHAP| is stable at this size and the plots only draw a
    # few thousand points anyway.
    rng = np.random.default_rng(42)
    X_all = test_df[FEATURE_COLS].values
    n_shap = min(SHAP_COMPUTE_SAMPLE, len(X_all))
    shap_idx = rng.choice(len(X_all), size=n_shap, replace=False)
    X = X_all[shap_idx]
    shap_bids = test_df["building_id"].values[shap_idx]
    print(f"Computing TreeSHAP on {n_shap:,}-row sample (native pred_contribs)...")
    shap_values, expected = compute_shap(booster, X)
    print(f"  SHAP matrix {shap_values.shape}, base value {expected:.2f} kWh")

    print("Plotting...")
    mean_abs = plot_global_importance(shap_values, FEATURE_COLS)

    plot_idx = rng.choice(n_shap, size=min(SHAP_PLOT_SAMPLE, n_shap), replace=False)
    use_label = (
        test_df.groupby("use_type_encoded")["primaryspaceusage"]
        .first()
        .astype(str)
        .to_dict()
    )
    plot_beeswarm(shap_values[plot_idx], X[plot_idx], FEATURE_COLS)
    plot_arch_dependence(
        shap_values[plot_idx],
        X[plot_idx],
        FEATURE_COLS,
        use_label,
        shap_bids[plot_idx],
        known_year_bids,
    )
    plot_error_by_use_type(err)
    rho_sqm, p_sqm, rho_age, p_age = plot_error_by_size_age(err)

    # --- summary numbers ---
    arch_share = mean_abs[ARCH_FEATURES].sum() / mean_abs.sum() * 100
    by_type = err.groupby("use_type")[["cv_rmse", "mape"]].median()

    print("\n" + "=" * 64)
    print("PHASE 5 — SHAP + ARCHITECTURAL FINDINGS")
    print("=" * 64)
    print(f"Architectural features' share of total |SHAP|: {arch_share:.1f}%")
    print(
        f"Top architectural feature: {mean_abs[ARCH_FEATURES].idxmax()} "
        f"(mean|SHAP| {mean_abs[ARCH_FEATURES].max():.2f} kWh)"
    )
    print(f"Size  vs CV(RMSE): Spearman ρ = {rho_sqm:+.2f} ({_fmt_p(p_sqm)})")
    print(
        f"Age   vs CV(RMSE): Spearman ρ = {rho_age:+.2f} ({_fmt_p(p_age)})  "
        f"[n={len(known_year_bids)} real year-built]"
    )
    print("Median error by use type:")
    for ut, r in by_type.iterrows():
        print(f"  {ut:<12} CV(RMSE) {r['cv_rmse']:.3f}   MAPE {r['mape']:.1f}%")
    print("=" * 64)

    with mlflow.start_run(run_name="shap_architectural_analysis"):
        mlflow.log_params(
            {
                "shap_method": "xgboost native TreeSHAP (pred_contribs)",
                "test_buildings": len(err),
                "error_metric": "cv_rmse (RMSE / mean load)",
                "age_subset": f"{len(known_year_bids)} buildings with real yearbuilt",
            }
        )
        mlflow.log_metric("shap_base_value_kwh", expected)
        mlflow.log_metric("arch_shap_share_pct", float(arch_share))
        for feat, val in mean_abs.items():
            mlflow.log_metric(f"shap_meanabs__{feat}", float(val))
        mlflow.log_metrics(
            {
                "corr_sqm_cvrmse_spearman": rho_sqm,
                "corr_age_cvrmse_spearman": rho_age,
                "cvrmse_median_office": float(by_type.loc["Office", "cv_rmse"])
                if "Office" in by_type.index
                else float("nan"),
                "cvrmse_median_education": float(by_type.loc["Education", "cv_rmse"])
                if "Education" in by_type.index
                else float("nan"),
                "mape_median_office": float(by_type.loc["Office", "mape"])
                if "Office" in by_type.index
                else float("nan"),
                "mape_median_education": float(by_type.loc["Education", "mape"])
                if "Education" in by_type.index
                else float("nan"),
            }
        )
        for fig_name in [
            "shap_global_importance",
            "shap_beeswarm",
            "shap_arch_dependence",
            "arch_error_by_use_type",
            "arch_error_by_size_age",
        ]:
            mlflow.log_artifact(
                str(FIGURES / f"{fig_name}.png"), artifact_path="figures"
            )

    out = PROCESSED / "building_error_analysis.parquet"
    err.to_parquet(out, index=False)
    print(f"\nPer-building error table → {out}")
    print(f"Figures → {FIGURES.resolve()}/")

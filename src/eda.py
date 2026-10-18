"""Stage 6 EDA (A1): analysis and plotting functions.

Q1. How do AC power and irradiance relate, per inverter?
Q2. Where are the gaps / offline events?

notebooks/01_eda_A1.ipynb calls these functions; figures go to reports/figures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
PROCESSED = ROOT / CONFIG["paths"]["processed"]
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
DAYLIGHT_WM2 = CONFIG["daylight_irradiance_wm2"]
SEED = CONFIG["random_state"]

# Colours (one sequential blue ramp, neutral inks and grid).
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NO_DATA = "#e8e7e2"
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blues", ["#f2f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

COLUMNS = ["inverter_id", "time", "ac_power_w", "n_valid_5min",
           "irradiance_wm2", "capacity_kwp"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.size": 9, "legend.frameon": False,
    })


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    return path


def load(columns=COLUMNS):
    """Load the 15-min table and add specific yield (kW per kWp)."""
    df = pd.read_parquet(PROCESSED / "hkust_15min_v1.parquet", columns=columns)
    df["inverter_id"] = df["inverter_id"].astype(str)
    df["specific_yield"] = df["ac_power_w"] / 1000 / df["capacity_kwp"]
    return df


# ---------------------------------------------------------------- Q1

def daylight_valid(df):
    """Rows usable for the power-irradiance relationship."""
    return df[(df["irradiance_wm2"] >= DAYLIGHT_WM2) & df["specific_yield"].notna()]


def plot_yield_vs_irradiance(df):
    """All inverters: specific yield vs irradiance as a 2D density (hexbin)."""
    d = df[df["irradiance_wm2"].notna() & df["specific_yield"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.hexbin(d["irradiance_wm2"], d["specific_yield"], gridsize=90,
                   extent=(0, 1300, 0, 1.3), bins="log", cmap=SEQ_BLUES, mincnt=1,
                   linewidths=0)
    ax.plot([0, 1300], [0, 1.3], color=INK_2, lw=1, ls="--")
    ax.annotate("1 kW/kWp per 1000 W/m²", xy=(1150, 1.15), xytext=(820, 1.22),
                color=INK_2, fontsize=8)
    ax.set_xlim(0, 1300)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Irradiance (W/m², weather station)")
    ax.set_ylabel("Specific yield (kW per kWp)")
    ax.set_title(f"AC power vs irradiance, all 44 inverters ({len(d):,} 15-min bins)")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, label="Bins per hexagon (log scale)", shrink=0.85)
    return fig


def per_inverter_fit(df):
    """Per inverter (daylight bins only): Pearson r, slope through the origin
    in kW/kWp per 1000 W/m2, and the 99th percentile of specific yield."""
    d = daylight_valid(df)
    rows = []
    for inv, g in d.groupby("inverter_id"):
        x = g["irradiance_wm2"].to_numpy(float) / 1000
        y = g["specific_yield"].to_numpy(float)
        rows.append({
            "inverter_id": inv,
            "n_bins": len(g),
            "pearson_r": np.corrcoef(x, y)[0, 1],
            "slope_kw_per_kwp": (x @ y) / (x @ x),
            "p99_specific_yield": np.quantile(y, 0.99),
            "capacity_kwp": g["capacity_kwp"].iloc[0],
        })
    out = pd.DataFrame(rows).sort_values("slope_kw_per_kwp").reset_index(drop=True)
    return out.round({"pearson_r": 3, "slope_kw_per_kwp": 3, "p99_specific_yield": 3})


def plot_per_inverter_grid(df, fits, sample=1500):
    """Small multiples: one scatter per inverter, ordered by fitted slope."""
    d = daylight_valid(df)
    order = fits["inverter_id"].tolist()
    ncols = 6
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.0 * nrows),
                             sharex=True, sharey=True)
    for ax, inv in zip(axes.flat, order):
        g = d[d["inverter_id"] == inv]
        g = g.sample(min(sample, len(g)), random_state=SEED)
        ax.scatter(g["irradiance_wm2"], g["specific_yield"], s=2, color=BLUE,
                   alpha=0.25, linewidths=0)
        slope = fits.loc[fits["inverter_id"] == inv, "slope_kw_per_kwp"].item()
        ax.plot([0, 1300], [0, 1.3 * slope], color=INK, lw=1)
        ax.set_title(f"{inv.removesuffix('_inverter')}\nslope {slope:.2f}", fontsize=7.5)
        ax.set_xlim(0, 1300)
        ax.set_ylim(0, 1.3)
    for ax in list(axes.flat)[len(order):]:
        ax.set_visible(False)
    fig.supxlabel("Irradiance (W/m²)", color=INK_2)
    fig.supylabel("Specific yield (kW per kWp)", color=INK_2)
    fig.suptitle("Per-inverter power vs irradiance (daylight bins, 1,500-point sample each; "
                 "line = fitted slope)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    return fig


def weekly_yield_ratio(df):
    """Campus-wide weekly median of specific yield / (irradiance / 1000).

    For a healthy system this ratio is roughly constant (it is close to the
    performance ratio), so a sudden step for all inverters at once points to
    the irradiance sensor, not the inverters."""
    d = daylight_valid(df)
    ratio = d["specific_yield"] / (d["irradiance_wm2"] / 1000)
    weekly = ratio.groupby(d["time"].dt.to_period("W").dt.start_time).median()
    return weekly.rename("median_ratio").reset_index()


def plot_weekly_yield_ratio(weekly, change_date="2023-02-14"):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(weekly["time"], weekly["median_ratio"], color=BLUE, lw=2)
    ax.axvline(pd.Timestamp(change_date), color=INK_2, lw=1, ls="--")
    ax.annotate(f"Irradiance readings jump\n{change_date}", xy=(pd.Timestamp(change_date), 1.1),
                xytext=(10, 0), textcoords="offset points", color=INK_2, fontsize=8, va="top")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Median specific yield per\n1000 W/m² irradiance")
    ax.set_xlabel("Week")
    ax.set_title("Output per unit irradiance, all inverters: a step change in Feb 2023 "
                 "(sensor, not inverters)")
    return fig


def low_irradiance_profile(df, max_wm2=150, step=5):
    """Behaviour near dawn/dusk: per irradiance bin, the share of bins with
    any AC output and the median specific yield (valid power readings only)."""
    d = df[df["irradiance_wm2"].notna() & df["specific_yield"].notna()
           & (df["irradiance_wm2"] < max_wm2)].copy()
    d["irr_bin"] = (d["irradiance_wm2"] // step * step).astype(int)
    out = d.groupby("irr_bin").agg(
        n_bins=("specific_yield", "size"),
        share_producing=("ac_power_w", lambda s: (s > 0).mean()),
        median_yield=("specific_yield", "median"),
        q25_yield=("specific_yield", lambda s: s.quantile(0.25)),
        q75_yield=("specific_yield", lambda s: s.quantile(0.75)),
    )
    return out.reset_index()


def plot_low_irradiance(profile):
    """Two panels sharing the x axis (two measures -> two charts, not two y axes)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    x = profile["irr_bin"] + 2.5
    a1.plot(x, 100 * profile["share_producing"], color=BLUE, lw=2)
    a1.set_ylabel("Bins with AC output > 0 (%)")
    a1.set_ylim(0, 102)
    a1.set_title("Low-irradiance behaviour (evidence for the night-time threshold)")
    a2.fill_between(x, profile["q25_yield"], profile["q75_yield"], color=BLUE,
                    alpha=0.2, linewidth=0, label="Interquartile range")
    a2.plot(x, profile["median_yield"], color=BLUE, lw=2, label="Median")
    a2.set_ylabel("Specific yield (kW per kWp)")
    a2.set_xlabel("Irradiance (W/m², 5 W/m² bins)")
    a2.legend(loc="upper left")
    for ax in (a1, a2):
        for t in (20, 50):
            ax.axvline(t, color=MUTED, lw=1, ls=":")
    a1.text(21, 5, "20", color=MUTED, fontsize=8)
    a1.text(51, 5, "50 W/m²", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Q2

def plot_gap_heatmap(gaps=None, coverage=None):
    """Inverter x month heatmap of % daylight bins missing.
    Grey = inverter not yet installed (no rows for that month)."""
    gaps = gaps if gaps is not None else pd.read_csv(TABLES / "gaps_by_inverter_month.csv")
    coverage = coverage if coverage is not None else pd.read_csv(TABLES / "coverage_by_inverter.csv")
    grid = gaps.pivot(index="inverter_id", columns="month", values="missing_pct")
    months = pd.period_range("2021-05", "2023-12", freq="M").astype(str)
    grid = grid.reindex(index=coverage.sort_values("coverage")["inverter_id"],
                        columns=months)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor(NO_DATA)
    im = ax.imshow(grid.to_numpy(float), aspect="auto", cmap=SEQ_BLUES,
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([i.removesuffix("_inverter") for i in grid.index], fontsize=7)
    ticks = [i for i, m in enumerate(months) if m.endswith(("-01", "-07"))]
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i] for i in ticks], fontsize=8)
    ax.grid(False)
    ax.set_title("Missing daylight data per inverter and month "
                 "(rows sorted by overall coverage, lowest first; grey = not yet installed)")
    fig.colorbar(im, ax=ax, label="Daylight bins missing (%)", shrink=0.6)
    return fig


def gap_runs(df):
    """Consecutive runs of missing daylight bins per inverter.

    Only daylight bins (irradiance >= threshold) are used, in time order, so a
    multi-day outage counts as one run and nights do not break it. Each run is
    measured in 15-min daylight bins."""
    d = df[df["irradiance_wm2"] >= DAYLIGHT_WM2].sort_values(["inverter_id", "time"])
    missing = (d["n_valid_5min"] == 0).to_numpy()
    inverter = d["inverter_id"].to_numpy()
    new_run = np.r_[True, (missing[1:] != missing[:-1]) | (inverter[1:] != inverter[:-1])]
    run_id = np.cumsum(new_run)
    runs = pd.DataFrame({"run": run_id, "missing": missing, "inverter_id": inverter,
                         "time": d["time"].to_numpy()})
    runs = runs[runs["missing"]].groupby("run").agg(
        inverter_id=("inverter_id", "first"), start=("time", "first"),
        end=("time", "last"), length_bins=("missing", "size"))
    return runs.reset_index(drop=True)


RUN_BUCKETS = [(1, 1, "15 min"), (2, 4, "30 min-1 h"), (5, 16, "1-4 h"),
               (17, 48, "4-12 h"), (49, 10**9, "> 12 h (multi-day)")]


def gap_run_summary(runs):
    total = runs["length_bins"].sum()
    rows = []
    for lo, hi, label in RUN_BUCKETS:
        r = runs[runs["length_bins"].between(lo, hi)]
        rows.append({"gap_length": label, "n_runs": len(r),
                     "pct_of_runs": round(100 * len(r) / len(runs), 1),
                     "missing_bins": int(r["length_bins"].sum()),
                     "pct_of_missing_bins": round(100 * r["length_bins"].sum() / total, 1)})
    return pd.DataFrame(rows)


def plot_gap_runs(summary):
    """Two panels: how many gaps vs how much missing data, per gap length."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(summary))
    for ax, col, title in [(a1, "pct_of_runs", "Share of gaps (%)"),
                           (a2, "pct_of_missing_bins", "Share of missing daylight bins (%)")]:
        ax.barh(y, summary[col], color=BLUE, height=0.6)
        for yi, v in zip(y, summary[col]):
            ax.text(v + 1, yi, f"{v:.1f}%", va="center", color=INK_2, fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 105)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y)
    a1.set_yticklabels(summary["gap_length"])
    a1.invert_yaxis()
    a1.set_ylabel("Gap length (consecutive daylight bins)")
    fig.suptitle("Gaps are mostly whole days or longer; short gaps (up to 1 h) are rare",
                 color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def missing_by_hour(df):
    d = df.assign(hour=df["time"].dt.hour, missing=df["n_valid_5min"] == 0)
    return (100 * d.groupby("hour")["missing"].mean()).round(2).rename("pct_missing").reset_index()


def plot_missing_by_hour(by_hour):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_hour["hour"], by_hour["pct_missing"], color=BLUE, width=0.6)
    for h, v in zip(by_hour["hour"], by_hour["pct_missing"]):
        ax.text(h, v + 0.6, f"{v:.1f}", ha="center", color=INK_2, fontsize=8)
    ax.set_xticks(by_hour["hour"])
    ax.set_xticklabels([f"{h:02d}:00" for h in by_hour["hour"]], fontsize=8)
    ax.set_xlabel("Hour of day (logger window 06:00-17:59)")
    ax.set_ylabel("15-min bins with no valid power (%)")
    ax.set_title("Missing data by hour of day (all bins, all inverters)")
    ax.grid(axis="x", visible=False)
    return fig

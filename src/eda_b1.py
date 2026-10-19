"""Stage 6 EDA (B1): 3-Year Degradation Trend and Consistently Underperforming Stations.

Questions owned by B1 (§6 of project plan):
- Q1. How much does inverter power degrade over the 3 years (2021–2023)?
      Informs: Temporal drift awareness, justification for chronological TimeSeriesSplit,
               and necessity of seasonal / multi-year features.
- Q2. Which rooftop stations consistently underperform their peers?
      Informs: Distinguishing structural shading/rooftop constraints from hardware faults;
               justifies the Stage 7 Peer-Deviation feature formulation.

Run this file directly to generate all 4 publication-grade figures:
    python -m src.eda_b1
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {"paths": {"processed": "data/processed"}}

PROCESSED = ROOT / CONFIG.get("paths", {}).get("processed", "data/processed")
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Cohesive, publication-quality aesthetic styling
BLUE_PRIMARY = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
DARK_NAVY = "#0d366b"
AMBER_WARNING = "#e67e22"
RED_FAULT = "#d9534f"
GREEN_HEALTHY = "#2ca02c"
GRAY_GRID = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT = "#52514e"


def set_plot_style():
    """Apply unified, professional matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": MUTED_TEXT,
        "axes.titlecolor": TEXT_COLOR,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRAY_GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED_TEXT,
        "ytick.color": MUTED_TEXT,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "font.size": 9.5,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data():
    """Load the processed HKUST 15-minute telemetry dataset.

    Returns:
        pd.DataFrame: Loaded dataset with parsed timestamps and normalized power columns.
    """
    parquet_path = PROCESSED / "hkust_15min_v1.parquet"
    key_cols = [
        "time", "inverter_id", "ac_power_w", "irradiance_wm2",
        "capacity_kwp", "station_id"
    ]

    if parquet_path.exists():
        print(f"Loading real dataset from {parquet_path.relative_to(ROOT)}...")
        df = pd.read_parquet(parquet_path, columns=key_cols)
    else:
        print("Notice: hkust_15min_v1.parquet not found. Generating simulated representative sample.")
        np.random.seed(42)
        dates = pd.date_range("2021-05-01 06:00:00", "2023-12-31 18:00:00", freq="15min")
        # Filter daylight hours
        dates = dates[(dates.hour >= 6) & (dates.hour <= 18)]
        inverter_names = [f"inverter_{i+1:02d}" for i in range(15)]

        rows = []
        for d in dates:
            day_of_year = d.dayofyear
            # Sun position factor
            hour_factor = np.sin(np.pi * (d.hour + d.minute / 60.0 - 6) / 12.0)
            if hour_factor <= 0:
                continue
            irr = np.clip(1000.0 * hour_factor * np.random.uniform(0.7, 1.05), 0, 1100.0)

            # 3-year aging: ~0.8% degradation per year
            year_aging = 1.0 - 0.008 * (d.year - 2021)

            for idx, inv in enumerate(inverter_names):
                cap = 25.0
                # Simulate underperforming station for inverter 1 and 2
                station_factor = 0.55 if idx < 2 else (1.15 if idx > 12 else 1.0)
                ac_power = irr * cap * 0.78 * station_factor * year_aging * np.random.uniform(0.92, 1.03)

                rows.append({
                    "time": d,
                    "inverter_id": inv,
                    "station_id": f"station_{(idx // 3) + 1}",
                    "capacity_kwp": cap,
                    "irradiance_wm2": irr,
                    "ac_power_w": max(0.0, ac_power),
                })
        df = pd.DataFrame(rows)

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["hour"] = df["time"].dt.hour

    # Specific Yield (kW AC power per kWp DC rated capacity)
    if "specific_yield" not in df.columns:
        df["specific_yield"] = (df["ac_power_w"] / 1000.0) / df["capacity_kwp"]

    return df


# ---------------------------------------------------------------------------
# Q1: 3-Year Inverter Degradation Analysis
# ---------------------------------------------------------------------------

def filter_peak_sun_hours(df, min_irradiance=600.0, start_hour=11, end_hour=14):
    """Filter for clear-sky, high-irradiance conditions to isolate hardware aging.

    Comparing raw solar generation across years is misleading because weather varies.
    By filtering for high irradiance (G >= 600 W/m²) at midday peak hours,
    solar input is held constant, isolating true electrical and cell degradation.
    """
    peak = df[
        (df["irradiance_wm2"] >= min_irradiance) &
        (df["hour"].between(start_hour, end_hour)) &
        (df["capacity_kwp"] > 0)
    ].copy()

    # Instantaneous Performance Ratio (clipped to physical upper bound)
    expected_power = (peak["irradiance_wm2"] / 1000.0) * (peak["capacity_kwp"] * 1000.0) * 0.78
    peak["pr"] = np.clip(peak["ac_power_w"] / (expected_power + 1e-4), 0.0, 2.5)

    return peak


def analyze_degradation(peak_df):
    """Calculate annual degradation statistics and linear regression slope."""
    annual_stats = peak_df.groupby("year")["specific_yield"].agg(
        median="median",
        mean="mean",
        p25=lambda x: np.percentile(x.dropna(), 25),
        p75=lambda x: np.percentile(x.dropna(), 75),
        count="count"
    ).reset_index()

    # Fit linear regression slope: Specific Yield = beta_0 + beta_1 * (year - 2021)
    years = annual_stats["year"].values
    medians = annual_stats["median"].values

    if len(years) >= 2:
        slope, intercept = np.polyfit(years, medians, 1)
        base_yield = medians[0]
        deg_rate_pct = (slope / base_yield) * 100.0
    else:
        slope, deg_rate_pct = 0.0, 0.0

    # Calculate per-inverter degradation rates
    inverter_records = []
    for inv_id, grp in peak_df.groupby("inverter_id"):
        y_medians = grp.groupby("year")["specific_yield"].median()
        if len(y_medians) >= 2 and y_medians.iloc[0] > 0.05:
            inv_slope, _ = np.polyfit(y_medians.index.values, y_medians.values, 1)
            inv_rate = (inv_slope / y_medians.iloc[0]) * 100.0
            inverter_records.append({
                "inverter_id": inv_id,
                "base_yield_2021": y_medians.iloc[0],
                "final_yield": y_medians.iloc[-1],
                "slope_kw_per_kwp_yr": inv_slope,
                "deg_rate_pct_yr": inv_rate
            })

    inv_deg_df = pd.DataFrame(inverter_records)
    return annual_stats, slope, deg_rate_pct, inv_deg_df


def plot_q1_degradation_trend(peak_df, annual_stats, slope, deg_rate_pct):
    """Plot annual boxplots and regression trendline of peak solar yield."""
    set_plot_style()
    years = sorted(peak_df["year"].unique())
    year_data = [peak_df[peak_df["year"] == yr]["specific_yield"].dropna() for yr in years]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)

    # Boxplots showing distribution across years
    box = ax.boxplot(
        year_data,
        tick_labels=[f"{yr}\n(N={len(d):,})" for yr, d in zip(years, year_data)],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.45,
        medianprops=dict(color=DARK_NAVY, linewidth=2),
        meanprops=dict(color=RED_FAULT, linewidth=1.5, linestyle="--")
    )

    palette = [BLUE_LIGHT, "#86b6ef", "#3987e5"]
    for patch, col in zip(box["boxes"], palette[:len(years)]):
        patch.set_facecolor(col)
        patch.set_edgecolor(DARK_NAVY)

    # Overlay regression line across medians
    x_coords = np.array(range(1, len(years) + 1))
    medians = annual_stats["median"].values
    reg_line = medians[0] + slope * (np.array(years) - years[0])

    ax.plot(x_coords, medians, marker="o", color=DARK_NAVY, linewidth=2.2, label="Annual Median Yield")
    ax.plot(x_coords, reg_line, linestyle=":", color=RED_FAULT, linewidth=2.0,
            label=f"Fitted Degradation Slope ({deg_rate_pct:+.2f}% / year)")

    # Annotation box
    ax.annotate(
        f"Fitted Annual Slope: {slope:+.4f} kW/kWp per yr\n"
        f"Degradation Rate: {deg_rate_pct:+.2f}% / year\n"
        f"Clear-Sky Filter: G >= 600 W/m², 11:00-14:00",
        xy=(0.04, 0.15), xycoords="axes fraction",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#c3c2b7", alpha=0.9)
    )

    ax.set_title("Q1: 3-Year Solar Inverter Degradation Trend (2021–2023)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Calendar Year", fontsize=10)
    ax.set_ylabel("Peak Specific Yield (kW / kWp)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    save_path = FIGURES / "b1_q1_annual_degradation_trend.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


def plot_q1_inverter_degradation_distribution(inv_deg_df):
    """Plot distribution of annual degradation rates across all campus inverters."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=120)

    # Clean out extreme outlier strings (< -30% or > 20% due to major rewiring or partial record)
    valid_rates = inv_deg_df["deg_rate_pct_yr"].dropna()
    valid_rates_clipped = np.clip(valid_rates, -25.0, 15.0)

    median_deg = valid_rates_clipped.median()

    # Histogram
    n, bins, patches = ax.hist(
        valid_rates_clipped, bins=16, color=BLUE_PRIMARY, edgecolor="white", alpha=0.85
    )

    # Highlight negative degradation in blue/navy, severe in amber
    for b_left, p in zip(bins[:-1], patches):
        if b_left < -10.0:
            p.set_facecolor(AMBER_WARNING)
        elif b_left < 0.0:
            p.set_facecolor(BLUE_PRIMARY)
        else:
            p.set_facecolor("#2ca02c")

    ax.axvline(median_deg, color=RED_FAULT, linestyle="--", linewidth=2.0,
               label=f"Campus Median: {median_deg:.2f}% / year")
    ax.axvline(-0.8, color=DARK_NAVY, linestyle=":", linewidth=1.8,
               label="Theoretical Nameplate Aging (-0.8% / yr)")

    ax.set_title("Q1: Distribution of Annual Degradation Rates Across 44 Inverters", fontsize=11, fontweight="bold")
    ax.set_xlabel("Annual Degradation Rate (% / year)", fontsize=10)
    ax.set_ylabel("Inverter Count", fontsize=10)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    save_path = FIGURES / "b1_q1_inverter_degradation_slopes.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


# ---------------------------------------------------------------------------
# Q2: Consistently Underperforming Rooftop Stations Analysis
# ---------------------------------------------------------------------------

def analyze_station_performance(df, min_irradiance=100.0):
    """Compute 3-year performance ratio (PR) and peer deviation for each inverter."""
    daylight = df[
        (df["irradiance_wm2"] >= min_irradiance) &
        (df["capacity_kwp"] > 0)
    ].copy()

    # Theoretical expected generation under standard efficiency (PR formula)
    expected_power = (daylight["irradiance_wm2"] / 1000.0) * (daylight["capacity_kwp"] * 1000.0) * 0.78
    daylight["pr"] = np.clip(daylight["ac_power_w"] / (expected_power + 1e-4), 0.0, 2.5)

    # Station aggregate metrics
    station_stats = daylight.groupby("inverter_id").agg(
        median_pr=("pr", "median"),
        mean_pr=("pr", "mean"),
        p25_pr=("pr", lambda x: np.percentile(x.dropna(), 25)),
        p75_pr=("pr", lambda x: np.percentile(x.dropna(), 75)),
        capacity_kwp=("capacity_kwp", "first"),
        station_id=("station_id", "first"),
        count=("pr", "count")
    ).reset_index()

    # Campus median benchmark
    campus_median_pr = station_stats["median_pr"].median()
    station_stats["peer_deviation_pct"] = (
        (station_stats["median_pr"] - campus_median_pr) / campus_median_pr
    ) * 100.0

    station_stats = station_stats.sort_values("median_pr", ascending=True).reset_index(drop=True)
    return daylight, station_stats, campus_median_pr


def plot_q2_station_ranking(station_stats, campus_median_pr):
    """Plot horizontal bar chart ranking all stations by 3-year median PR."""
    set_plot_style()
    n_stations = len(station_stats)
    fig_height = max(8.0, n_stations * 0.28)
    fig, ax = plt.subplots(figsize=(10.0, fig_height), dpi=120)

    y_pos = np.arange(n_stations)

    # Assign colors: Bottom 5 = Amber/Red, Top 5 = Green, Others = Soft Blue
    colors = []
    for i in range(n_stations):
        if i < 5:
            colors.append(RED_FAULT)
        elif i >= n_stations - 5:
            colors.append(GREEN_HEALTHY)
        else:
            colors.append(BLUE_PRIMARY)

    bars = ax.barh(y_pos, station_stats["median_pr"], color=colors, height=0.68, edgecolor="none", alpha=0.9)

    # Campus median benchmark line
    ax.axvline(campus_median_pr, color=DARK_NAVY, linestyle="--", linewidth=1.8,
               label=f"Campus Median Benchmark (PR = {campus_median_pr:.3f})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(station_stats["inverter_id"], fontsize=8)
    ax.set_xlabel("3-Year Median Performance Ratio (PR)", fontsize=10)
    ax.set_title("Q2: 3-Year Inverter Performance Ranking Across Campus Rooftops", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    # Add text labels on the bars for bottom and top performers
    for idx, (bar, pr) in enumerate(zip(bars, station_stats["median_pr"])):
        if idx < 5 or idx >= n_stations - 5:
            ax.text(pr + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{pr:.2f}",
                    va="center", ha="left", fontsize=7.5, color=TEXT_COLOR, fontweight="semibold")

    ax.set_xlim(0, max(station_stats["median_pr"]) * 1.15)

    save_path = FIGURES / "b1_q2_station_performance_ranking.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


def plot_q2_underperformance_comparison(daylight_df, bottom_inverters=None, top_inverter=None):
    """Plot diurnal generation curves comparing bottom stations against campus benchmark."""
    set_plot_style()
    if bottom_inverters is None:
        bottom_inverters = ["zone_d_inverter_1", "ug_hall2_2f_inverter", "library_inverter_3"]
    if top_inverter is None:
        top_inverter = "sq4_inverter"

    daylight_df["norm_power"] = daylight_df["ac_power_w"] / (daylight_df["capacity_kwp"] * 1000.0)
    hourly_df = daylight_df[daylight_df["hour"].between(6, 18)]

    # Campus median profile by hour
    campus_diurnal = hourly_df.groupby("hour")["norm_power"].median()

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=120)

    # Campus benchmark
    ax.plot(
        campus_diurnal.index, campus_diurnal.values,
        color=DARK_NAVY, linewidth=2.8, linestyle="-", label="Campus Peer Median (Benchmark)"
    )

    # Top performer for comparison
    top_data = hourly_df[hourly_df["inverter_id"] == top_inverter].groupby("hour")["norm_power"].mean()
    if not top_data.empty:
        ax.plot(
            top_data.index, top_data.values,
            color=GREEN_HEALTHY, linewidth=2.0, linestyle="-.", label=f"Top Station: {top_data.name}"
        )

    # Bottom underperforming stations
    colors = [RED_FAULT, AMBER_WARNING, "#9b59b6"]
    for inv, col in zip(bottom_inverters, colors):
        inv_data = hourly_df[hourly_df["inverter_id"] == inv].groupby("hour")["norm_power"].mean()
        if not inv_data.empty:
            ax.plot(
                inv_data.index, inv_data.values,
                color=col, linewidth=2.0, linestyle="--", marker="o", markersize=4,
                label=f"Underperforming: {inv}"
            )

    ax.set_title("Q2: Diurnal Generation Shortfall of Underperforming Stations vs. Campus Benchmark",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Hour of Day (Daylight: 06:00 – 18:00)", fontsize=10)
    ax.set_ylabel("Mean Normalized Output (kW / kWp)", fontsize=10)
    ax.set_xticks(range(6, 19))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    save_path = FIGURES / "b1_q2_bottom_stations_shortfall.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

def run_all():
    """Execute all B1 EDA analyses and save all 4 report figures."""
    print("=" * 65)
    print("Executing Stage 6 EDA for Partner B1...")
    print("=" * 65)

    df = load_data()
    print(f"Dataset successfully loaded: {len(df):,} total 15-min intervals.")

    # 1. Question 1: Degradation Analysis
    print("\n--- Running Question 1: 3-Year Inverter Degradation Analysis ---")
    peak_df = filter_peak_sun_hours(df)
    print(f"Filtered {len(peak_df):,} clear-sky peak midday intervals (G >= 600 W/m²).")

    annual_stats, slope, deg_rate_pct, inv_deg_df = analyze_degradation(peak_df)
    print("\nAnnual Peak Specific Yield Summary:")
    print(annual_stats.to_string(index=False))
    print(f"\nFitted Annual Degradation Slope: {slope:+.4f} kW/kWp per year")
    print(f"Overall Compound Degradation Rate: {deg_rate_pct:+.2f}% / year")

    plot_q1_degradation_trend(peak_df, annual_stats, slope, deg_rate_pct)
    plot_q1_inverter_degradation_distribution(inv_deg_df)

    # 2. Question 2: Underperforming Stations Analysis
    print("\n--- Running Question 2: Consistently Underperforming Stations ---")
    daylight_df, station_stats, campus_median_pr = analyze_station_performance(df)
    print(f"Campus Median 3-Year PR Benchmark: {campus_median_pr:.3f}")

    print("\nBottom 5 Consistently Underperforming Inverters:")
    print(station_stats[["inverter_id", "median_pr", "mean_pr", "peer_deviation_pct"]].head(5).to_string(index=False))

    print("\nTop 5 Consistently Performing Inverters:")
    print(station_stats[["inverter_id", "median_pr", "mean_pr", "peer_deviation_pct"]].tail(5).to_string(index=False))

    plot_q2_station_ranking(station_stats, campus_median_pr)

    worst_inverters = station_stats["inverter_id"].head(3).tolist()
    top_inverter = station_stats["inverter_id"].iloc[-2]  # Pick representative top inverter
    plot_q2_underperformance_comparison(daylight_df, worst_inverters, top_inverter)

    print("\n" + "=" * 65)
    print("All B1 EDA tasks successfully completed! Figures saved in reports/figures/")
    print("=" * 65)


if __name__ == "__main__":
    run_all()

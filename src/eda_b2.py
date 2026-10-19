"""Stage 6 EDA (B2): Seasonality and Class Distribution Analysis.

Questions owned by B2 (§6 of project plan):
- Q1. How strong is seasonality across 3 years (2021-2023)?
      Informs: Chronological split & seasonal cyclic features.
- Q2. How are the status classes (Healthy, Warning, Faulty) distributed?
      Informs: Data balancing strategy & evaluation metric choice (Faulty Recall).

Run this file to generate all EDA figures:
    python -m src.eda_b2
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


# ---------------------------------------------------------------------------
# Simple Data Loader
# ---------------------------------------------------------------------------

def load_data():
    """Load the processed HKUST 15-minute dataset.

    If data/processed/hkust_15min_v1.parquet is not yet downloaded,
    it generates a realistic sample with 44 inverters across 2021-2023
    so the analysis runs immediately.
    """
    parquet_path = PROCESSED / "hkust_15min_v1.parquet"

    if parquet_path.exists():
        print(f"Loading real dataset from {parquet_path.relative_to(ROOT)}...")
        df = pd.read_parquet(parquet_path)
    else:
        print("Notice: hkust_15min_v1.parquet not found. Generating simulated representative sample.")
        np.random.seed(42)

        # 3 years of dates (daylight intervals)
        dates = pd.date_range("2021-05-01", "2023-12-31", freq="1D")
        inverter_names = [f"inverter_{i+1:02d}" for i in range(12)]  # sample of 12 representative inverters

        rows = []
        for d in dates:
            month = d.month
            day_of_year = d.dayofyear

            # Seasonality: solar output peaks in summer (June-Aug), lowest in winter (Dec-Feb)
            seasonal_factor = 0.6 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            # Add random weather variation
            weather_noise = np.random.uniform(0.6, 1.1)

            for inv in inverter_names:
                capacity_kwp = 25.0  # typical 25 kWp inverter

                # Specific yield in kW per kWp (0 to ~0.8)
                yield_val = np.clip(0.65 * seasonal_factor * weather_noise, 0.05, 0.85)
                ac_power_w = yield_val * capacity_kwp * 1000.0

                # Health score (0 to 100)
                # Most days healthy, some warning, few faulty
                rand_score = np.random.uniform(0, 1)
                if rand_score > 0.12:
                    health_score = np.random.uniform(90, 100)
                    status = "Healthy"
                elif rand_score > 0.03:
                    health_score = np.random.uniform(70, 89)
                    status = "Warning"
                else:
                    health_score = np.random.uniform(20, 69)
                    status = "Faulty"
                    ac_power_w *= 0.3  # faulty inverters produce much less

                rows.append({
                    "time": d,
                    "inverter_id": inv,
                    "capacity_kwp": capacity_kwp,
                    "ac_power_w": ac_power_w,
                    "specific_yield": yield_val,
                    "health_score": health_score,
                    "status": status,
                })

        df = pd.DataFrame(rows)

    # Ensure time and specific yield exist
    df["time"] = pd.to_datetime(df["time"])
    if "specific_yield" not in df.columns:
        df["specific_yield"] = (df["ac_power_w"] / 1000.0) / df["capacity_kwp"]

    # Assign meteorological seasons
    month = df["time"].dt.month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["season"] = month.map(season_map)

    # If status label is not yet created, create rule-based baseline
    if "status" not in df.columns:
        if "health_score" in df.columns:
            conditions = [
                df["health_score"] >= 90,
                (df["health_score"] >= 70) & (df["health_score"] < 90),
                df["health_score"] < 70,
            ]
            choices = ["Healthy", "Warning", "Faulty"]
            df["status"] = np.select(conditions, choices, default="Healthy")
        else:
            # Baseline status based on specific yield
            df["status"] = "Healthy"
            df.loc[df["specific_yield"] < 0.35, "status"] = "Warning"
            df.loc[df["specific_yield"] < 0.15, "status"] = "Faulty"

    return df


# ---------------------------------------------------------------------------
# Q1: Seasonality Analysis
# ---------------------------------------------------------------------------

def plot_monthly_seasonality(df):
    """Plot monthly average solar specific yield across 3 years (2021-2023)."""
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["time"].dt.to_period("M").astype(str)

    monthly_stats = df_copy.groupby("year_month")["specific_yield"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    monthly_stats.plot(kind="bar", color="#2a78d6", ax=ax, edgecolor="#1c5cab", width=0.7)

    ax.set_title("Q1: Monthly Average Solar Specific Yield (2021 - 2023)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year - Month", fontsize=10)
    ax.set_ylabel("Specific Yield (kW / kWp)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    save_path = FIGURES / "b2_q1_monthly_yield_trend.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return monthly_stats


def plot_season_boxplots(df):
    """Compare distributions of daily yield across the 4 seasons."""
    seasons_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_data = [df[df["season"] == s]["specific_yield"].dropna() for s in seasons_order]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    box = ax.boxplot(season_data, tick_labels=seasons_order, patch_artist=True, showmeans=True)


    colors = ["#cde2fb", "#86b6ef", "#3987e5", "#ffaa33"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_title("Q1: Solar Specific Yield Distribution by Season", fontsize=12, fontweight="bold")
    ax.set_xlabel("Season", fontsize=10)
    ax.set_ylabel("Specific Yield (kW / kWp)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    save_path = FIGURES / "b2_q1_seasonality_boxplots.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")


def plot_monthly_comparison_years(df):
    """A monthly average generation plot, comparing all three years (2021, 2022, 2023)."""
    df_copy = df.copy()
    df_copy["year"] = df_copy["time"].dt.year
    df_copy["month"] = df_copy["time"].dt.month

    # Group by year and month
    pivot = df_copy.pivot_table(index="month", columns="year", values="specific_yield", aggfunc="mean")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    colors = {2021: "#1f77b4", 2022: "#ff7f0e", 2023: "#2ca02c"}

    for yr in pivot.columns:
        ax.plot(pivot.index, pivot[yr], marker="o", label=f"Year {yr}", color=colors.get(yr, None), linewidth=2)

    ax.set_title("Q1: Monthly Average Solar Generation Compared Across 3 Years", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Specific Yield (kW / kWp)", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Year", frameon=True)

    save_path = FIGURES / "b2_q1_yearly_comparison.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return pivot


def plot_monthly_boxplots(df):
    """A box plot of generation grouped by month (Jan to Dec)."""
    df_copy = df.copy()
    df_copy["month"] = df_copy["time"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_data = [df_copy[df_copy["month"] == m]["specific_yield"].dropna() for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    box = ax.boxplot(monthly_data, tick_labels=month_names, patch_artist=True, showmeans=True)

    for patch in box["boxes"]:
        patch.set_facecolor("#86b6ef")
        patch.set_edgecolor("#1c5cab")

    ax.set_title("Q1: Distribution of Daily Solar Generation Grouped by Month", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Specific Yield (kW / kWp)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    save_path = FIGURES / "b2_q1_monthly_boxplots.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")


def plot_seasonal_decomposition(df):
    """A seasonal decomposition plot showing trend, seasonality, and residual variation."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    df_copy = df.copy()
    # Campus daily average
    daily = df_copy.groupby(df_copy["time"].dt.date)["specific_yield"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D").interpolate(method="time")

    # Use 365 days period if multi-year, else 30
    period = 365 if len(daily) >= 730 else 30
    res = seasonal_decompose(daily, model="additive", period=period)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), dpi=120, sharex=True)

    ax1.plot(res.observed, color="#2a78d6", lw=1)
    ax1.set_ylabel("Observed", fontsize=9)
    ax1.set_title("Q1: Time Series Seasonal Decomposition (Trend, Seasonality, Residual)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(res.trend, color="#ff7f0e", lw=1.5)
    ax2.set_ylabel("Trend", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.plot(res.seasonal, color="#2ca02c", lw=1)
    ax3.set_ylabel("Seasonality", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4.scatter(res.resid.index, res.resid, color="#7f7f7f", s=5, alpha=0.7)
    ax4.axhline(0, color="black", linestyle="--", lw=0.8)
    ax4.set_ylabel("Residual", fontsize=9)
    ax4.set_xlabel("Date", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)

    save_path = FIGURES / "b2_q1_seasonal_decomposition.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return res



# ---------------------------------------------------------------------------
# Q2: Class Distribution Analysis
# ---------------------------------------------------------------------------

def plot_class_distribution(df):
    """Plot overall distribution of Healthy, Warning, and Faulty classes."""
    counts = df["status"].value_counts()
    percentages = df["status"].value_counts(normalize=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120)

    # 1. Bar Chart
    colors = {"Healthy": "#28a745", "Warning": "#ffc107", "Faulty": "#dc3545"}
    bar_colors = [colors.get(c, "#6c757d") for c in counts.index]

    ax1.bar(counts.index, counts.values, color=bar_colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Operational Status: Class Counts", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of 15-min Intervals", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for i, v in enumerate(counts.values):
        ax1.text(i, v + (max(counts.values) * 0.02), f"{v:,}", ha="center", fontsize=9)

    # 2. Pie Chart
    ax2.pie(
        percentages.values,
        labels=percentages.index,
        autopct="%1.1f%%",
        colors=bar_colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax2.set_title("Class Proportions (% Imbalance)", fontsize=11, fontweight="bold")

    save_path = FIGURES / "b2_q2_class_distribution_overall.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")

    return counts, percentages


def plot_class_by_season(df):
    """Show how status classes vary across Winter, Spring, Summer, and Autumn."""
    crosstab = pd.crosstab(df["season"], df["status"], normalize="index") * 100
    seasons_order = ["Winter", "Spring", "Summer", "Autumn"]
    crosstab = crosstab.reindex([s for s in seasons_order if s in crosstab.index])

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    colors = ["#28a745", "#dc3545", "#ffc107"]  # Healthy, Faulty, Warning
    crosstab.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", alpha=0.85)

    ax.set_title("Q2: Class Proportions Across Seasons", fontsize=12, fontweight="bold")
    ax.set_xlabel("Season", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=0)

    save_path = FIGURES / "b2_q2_class_by_season.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_eda():
    """Run complete EDA pipeline for B2."""
    print("=" * 60)
    print("RUNNING B2 EDA: Seasonality & Class Distribution")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df):,} rows for analysis.")

    print("\n--- 1. Seasonality Analysis (Q1) ---")
    plot_monthly_seasonality(df)
    plot_season_boxplots(df)
    plot_monthly_comparison_years(df)
    plot_monthly_boxplots(df)
    plot_seasonal_decomposition(df)

    print("\n--- 2. Class Distribution Analysis (Q2) ---")
    plot_class_distribution(df)
    plot_class_by_season(df)

    print("\nAll B2 EDA plots successfully created in reports/figures/!")



if __name__ == "__main__":
    run_all_eda()

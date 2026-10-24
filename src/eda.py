"""Stage 6: exploratory data analysis. Two questions per team member (section 6).

A1  Q1 How do AC power and irradiance relate?  (+ sensor change, night threshold)
    Q2 Where are the gaps / offline events?
A2  Q1 How does temperature affect output?
    Q2 Are features redundant?
B1  Q1 Do the inverters degrade over the years?
    Q2 Which inverters underperform consistently?
B2  Q1 How strong is seasonality?
    Q2 How are the classes distributed?

Every function draws one simple figure into reports/figures/.
Run: python -m src.eda
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import CONFIG, PROCESSED, TABLES, save_figure
from src.dataset import CANDIDATE_FEATURES, load

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def specific_yield(df):
    """AC power per kW of installed panels (kW/kWp), so big and small arrays compare."""
    return df["ac_power_w"] / 1000 / df["capacity_kwp"]


# ---------------------------------------------------------------- A1

def power_vs_irradiance(df):
    d = df[df["irradiance_wm2"] >= 50].dropna(subset=["ac_power_w"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hb = ax.hexbin(d["irradiance_wm2"], specific_yield(d), gridsize=70, bins="log",
                   extent=(0, 1100, 0, 1.2), cmap="Blues", mincnt=1)
    ax.plot([0, 1100], [0, 1.1], "k--", label="1 kW/kWp at 1000 W/m2")
    ax.set(title="AC power vs irradiance, all inverters", xlabel="Irradiance (W/m2)",
           ylabel="Specific yield (kW per kWp)")
    ax.legend(loc="upper left")
    fig.colorbar(hb, label="15-min rows (log)")
    save_figure(fig, "eda_a1_power_vs_irradiance")


def sensor_change(df):
    """Output per unit of sunlight, week by week. A sudden drop for every inverter
    at once means the sunlight sensor changed, not the inverters."""
    d = df[(df["irradiance_raw_wm2"] >= 100)].dropna(subset=["ac_power_w"])
    week = d["time"].dt.to_period("W").dt.start_time
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for column, label in [("irradiance_raw_wm2", "raw sensor"), ("irradiance_wm2", "after correction")]:
        ratio = specific_yield(d) / (d[column] / 1000)
        ax.plot(ratio.groupby(week).median(), label=label)
    ax.axvline(pd.Timestamp(CONFIG["irradiance_sensor_change"]), color="red", linestyle="--",
               label="2023-02-14")
    ax.set(title="Output per unit irradiance: step change on 2023-02-14 (sensor)",
           ylabel="kW/kWp per 1000 W/m2", ylim=(0, 1.3))
    ax.legend()
    save_figure(fig, "eda_a1_sensor_change")


def night_threshold(df):
    """Below what light level do inverters stop producing? (justifies 50 W/m2)"""
    d = df[df["irradiance_wm2"] < 150].dropna(subset=["ac_power_w"])
    producing = (d["ac_power_w"] > 0).groupby((d["irradiance_wm2"] // 5) * 5).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(producing.index, producing.values, marker=".")
    ax.axvline(50, color="red", linestyle="--", label="50 W/m2")
    ax.set(title="Share of bins producing power at low light", xlabel="Irradiance (W/m2)",
           ylabel="% of bins with power > 0")
    ax.legend()
    save_figure(fig, "eda_a1_night_threshold")


def gap_heatmap():
    gaps = pd.read_csv(TABLES / "gaps_by_inverter_month.csv")
    grid = gaps.pivot(index="inverter_id", columns="month", values="missing_share") * 100
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_yticks(range(len(grid)), [i.replace("_inverter", "") for i in grid.index], fontsize=7)
    ticks = range(0, len(grid.columns), 6)
    ax.set_xticks(ticks, [grid.columns[i] for i in ticks])
    ax.set_title("% of daylight data missing per inverter and month (white = not installed yet)")
    ax.grid(False)
    fig.colorbar(im, label="% missing")
    save_figure(fig, "eda_a1_gap_heatmap")


def missing_by_hour(df):
    missing = (df["n_valid_5min"] == 0).groupby(df["time"].dt.hour).mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(missing.index, missing.values)
    ax.set(title="Missing data by hour (06:00 = inverters still asleep)", xlabel="Hour of day",
           ylabel="% of bins with no power reading")
    save_figure(fig, "eda_a1_missing_by_hour")


# ---------------------------------------------------------------- A2

def temperature_effect(table):
    """Performance ratio in strong sun, by air temperature: hot panels convert less light.
    (The health score is capped at 100, so the uncapped performance ratio is used here.)"""
    sunny = table[table["irradiance_wm2"] >= 600]
    groups = sunny.groupby((sunny["temp_c"] // 2) * 2)["pr"]
    by_temp = groups.median()[groups.size() >= 500]          # skip bins with few rows
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(by_temp.index, by_temp.values, marker="o")
    ax.set(title="Median performance ratio in strong sun (>= 600 W/m2) by temperature",
           xlabel="Air temperature (C, 2-degree bins)", ylabel="Performance ratio")
    save_figure(fig, "eda_a2_temperature_effect")
    return by_temp


def feature_correlation(table):
    corr = table[table["split"] == "train"][CANDIDATE_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.set_title("Correlation between features (training period)")
    ax.grid(False)
    fig.colorbar(im, label="Pearson r")
    save_figure(fig, "eda_a2_feature_correlation")


# ---------------------------------------------------------------- B1

def peak_sun(table):
    """Clear midday rows (11:00-14:00, >= 600 W/m2): the fairest time to compare."""
    hour = table["time"].dt.hour
    return table[(hour >= 11) & (hour < 14) & (table["irradiance_wm2"] >= 600)]


def degradation(table):
    peak = peak_sun(table)
    yearly = peak.groupby(peak["time"].dt.year)["pr"].median()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(yearly.index.astype(str), yearly.values)
    for x, y in enumerate(yearly.values):
        ax.text(x, y, f"{y:.3f}", ha="center", va="bottom")
    ax.set(title="Median performance ratio at peak sun, per year", ylabel="Performance ratio")
    save_figure(fig, "eda_b1_degradation")
    return yearly


def underperformers(table):
    peak = peak_sun(table)
    by_inverter = (peak.groupby("inverter_id")["pr"].median() / peak["pr"].median() - 1) * 100
    by_inverter = by_inverter.sort_values()
    fig, ax = plt.subplots(figsize=(7, 9))
    colors = ["tab:red" if v < -10 else "tab:blue" for v in by_inverter.values]
    ax.barh([i.replace("_inverter", "") for i in by_inverter.index], by_inverter.values, color=colors)
    ax.axvline(0, color="black")
    ax.set(title="Each inverter vs campus median at peak sun (red = more than 10 % below)",
           xlabel="% difference in performance ratio")
    ax.tick_params(axis="y", labelsize=7)
    save_figure(fig, "eda_b1_underperformers")
    return by_inverter


# ---------------------------------------------------------------- B2

def seasonality(df):
    """Average daylight output per kWp for each month, campus median over inverters.
    An average (not a sum) so that months with outages are not pulled down."""
    d = df[df["irradiance_wm2"] >= 50].dropna(subset=["ac_power_w"])
    monthly = specific_yield(d).groupby([d["inverter_id"], d["time"].dt.year, d["time"].dt.month]).mean()
    monthly = monthly.groupby(level=[1, 2]).median().unstack(level=0)
    monthly.columns.name = "year"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    monthly.plot(ax=ax, marker="o")
    ax.set(title="Average daylight output per kWp by month (campus median)", xlabel="Month",
           ylabel="kW per kWp")
    save_figure(fig, "eda_b2_seasonality")


def class_distribution(table):
    season = table["time"].dt.month.map(SEASONS)
    shares = pd.crosstab(season, table["status_lead"], normalize="index") * 100
    shares = shares.loc[["Winter", "Spring", "Summer", "Autumn"], ["Healthy", "Warning", "Faulty"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    shares.plot.bar(stacked=True, ax=ax, color=["tab:green", "tab:orange", "tab:red"], rot=0)
    ax.set(title="Class shares by season (status one hour ahead)", ylabel="% of rows", xlabel="")
    save_figure(fig, "eda_b2_class_distribution")
    return shares.round(1)


def main():
    df = pd.read_parquet(PROCESSED / "hkust_15min.parquet")
    table = load()
    print("A1"); power_vs_irradiance(df); sensor_change(df); night_threshold(df)
    gap_heatmap(); missing_by_hour(df)
    print("A2"); print(temperature_effect(table).round(3).to_string()); feature_correlation(table)
    print("B1"); print(degradation(table).round(3).to_string()); underperformers(table)
    print("B2"); seasonality(df); print(class_distribution(table).to_string())


if __name__ == "__main__":
    main()

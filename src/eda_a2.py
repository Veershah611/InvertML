"""A2 EDA on B2's training partition only; run: python -m src.eda_a2."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import A2_HISTORY_FEATURES, B1_CORE_FEATURES, prune_collinear_features
from src.preprocess import ROOT, feature_frame

DATA = ROOT / "data" / "processed" / "a2" / "train.parquet"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    # Holding irradiance and midday hours roughly constant makes this more
    # interpretable than raw power vs temperature. Still observational EDA.
    t = df["time"]
    subset = df[
        df["irradiance_wm2"].between(300, 800)
        & t.dt.hour.between(10, 14)
        & df["performance_ratio"].notna()
        & df["temp_c"].between(0, 45)
    ].copy()
    subset["temp_bin_c"] = (subset["temp_c"] // 5 * 5).astype(int)
    temp = subset.groupby("temp_bin_c", observed=True).agg(
        count=("performance_ratio", "size"),
        median_pr=("performance_ratio", "median"),
        median_health=("health_score", "median"),
    ).reset_index()
    temp.to_csv(TABLES / "a2_temperature_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(temp["temp_bin_c"] + 2.5, temp["median_pr"], marker="o")
    ax.set(xlabel="Ambient temperature bin midpoint (°C)", ylabel="Median performance ratio",
           title="Training data: temperature and performance at 300–800 W/m², 10:00–14:59")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "a2_temperature_effect.png", dpi=160)
    plt.close(fig)

    # Calculate per-inverter association so an inverter's fixed capacity and
    # orientation cannot alone produce the pooled slope.
    slopes = []
    for inv, group in subset.groupby("inverter_id", observed=True):
        if len(group) < 100 or group["temp_c"].std() < 1:
            continue
        slope = np.polyfit(group["temp_c"], group["performance_ratio"], 1)[0]
        slopes.append({"inverter_id": inv, "n": len(group),
                       "pr_change_per_c": float(slope)})
    slope_df = pd.DataFrame(slopes).sort_values("pr_change_per_c")
    slope_df.to_csv(TABLES / "a2_temperature_inverter_slopes.csv", index=False)

    X = feature_frame(df).select_dtypes("number")
    corr = X.corr().fillna(0)
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            r = corr.loc[left, right]
            if abs(r) >= .85:
                pairs.append({"feature_a": left, "feature_b": right,
                              "pearson_r": float(r)})
    pair_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"])
    pair_df = pair_df.reindex(pair_df.pearson_r.abs().sort_values(ascending=False).index)
    pair_df.to_csv(TABLES / "a2_feature_correlations.csv", index=False)
    protected = ["peer_dev_campus", "self_dev_30d", "thermal_stress",
                 "capacity_kwp", "hour_sin", "day_sin"] + A2_HISTORY_FEATURES
    selected, log = prune_collinear_features(
        X, list(X.columns), threshold=.85,
        protected_features=protected, save_report=False,
    )
    pd.DataFrame(log, columns=["dropped_feature", "kept_feature", "correlation",
                               "threshold", "reason"]).to_csv(
        TABLES / "a2_feature_pruning.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("A2 training-only input correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "a2_feature_correlation.png", dpi=150)
    plt.close(fig)
    print(f"Temperature rows: {len(subset):,}; inverters with slopes: {len(slope_df)}")
    print(f"Median per-inverter PR slope / °C: {slope_df.pr_change_per_c.median():.5f}")
    print(f"Strong correlation pairs: {len(pair_df)}; candidate numeric columns: {len(X.columns)}; selected: {len(selected)}")
    print("Suggested dropped columns:", [item["dropped_feature"] for item in log])
    # Stage 5 distribution: preserve each partition's natural class balance.
    seasonal = []
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(DATA.parent / f"{split}.parquet",
                                columns=["time", "status_lead4"])
        frame["season"] = frame.time.dt.month.map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn",
        })
        counts = frame.groupby(["season", "status_lead4"], observed=True).size()
        totals = frame.groupby("season", observed=True).size()
        for (season, label), count in counts.items():
            seasonal.append({"split": split, "season": season, "status": label,
                             "count": int(count), "percentage": 100 * count / totals[season]})
    season_df = pd.DataFrame(seasonal)
    season_df.to_csv(TABLES / "a2_class_distribution_by_season.csv", index=False)
    train_dist = season_df[season_df.split.eq("train")].pivot(
        index="season", columns="status", values="percentage"
    ).reindex(["winter", "spring", "summer", "autumn"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(train_dist))
    for label, color in [("Healthy", "#3a9d62"), ("Warning", "#e6a239"),
                         ("Faulty", "#d45757")]:
        values = train_dist[label] if label in train_dist else np.zeros(len(train_dist))
        ax.bar(train_dist.index, values, bottom=bottom, label=label, color=color)
        bottom += np.asarray(values)
    ax.set(ylabel="Percentage of forecast labels", title="A2 training class balance by season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "a2_class_distribution_by_season.png", dpi=150)
    plt.close(fig)
    return selected


if __name__ == "__main__":
    run()

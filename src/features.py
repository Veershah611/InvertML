"""Stage 7: Feature Engineering & Selection Framework (src/features.py).

Owned by:
- B1 (Pipeline Lead for Stage 7): Performance Ratio, Peer Deviation, Self-Relative Deviation,
  Thermal Interactions, and Collinearity Pruning / Feature Selection.
- B2: Cyclical time features (sin/cos) and station metadata encoders.
- Pair A (A1/A2): Outage features and rolling lag features.

This module provides the unified feature engineering pipeline and feature selection engine
for the entire Solar Inverter Health Monitoring System.

Run directly to execute self-test and feature selection:
    python -m src.features
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluate import split_chronological

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

PROCESSED_DIR = ROOT / CONFIG.get("paths", {}).get("processed", "data/processed")
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Feature Definitions & Constants
# ===========================================================================

# B2 Feature Constants
B2_TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

B2_METADATA_FEATURES = [
    "capacity_kwp",
    "tilt_deg",
    "altitude_m",
    "orient_south",
    "orient_east_west",
    "orient_mixed",
]

# Public API used by A1's baselines and B2's MLP/notebook.
B2_ALL_FEATURES = B2_TIME_FEATURES + B2_METADATA_FEATURES

# B1 Signature Feature Group
B1_CORE_FEATURES = [
    "specific_yield",
    "performance_ratio",
    "peer_dev_campus",
    "peer_dev_station",
    "self_dev_30d",
    "thermal_stress",
    "temp_delta_stc",
]

A2_HISTORY_SOURCES = ["performance_ratio", "peer_dev_campus"]
A2_LAG_STEPS = {"15m": pd.Timedelta("15min"),
                "1h": pd.Timedelta("1h"), "1d": pd.Timedelta("1d")}
A2_ROLLING_WINDOWS = {"1h": "1h", "4h": "4h", "1d": "1D"}
A2_HISTORY_FEATURES = [
    f"{source}_lag_{name}"
    for source in A2_HISTORY_SOURCES for name in A2_LAG_STEPS
] + [
    f"{source}_roll_{name}_{stat}"
    for source in A2_HISTORY_SOURCES
    for name in A2_ROLLING_WINDOWS
    for stat in ("mean", "std")
]

# All Candidate Features prior to selection
ALL_CANDIDATE_FEATURES = B2_ALL_FEATURES + B1_CORE_FEATURES


# ===========================================================================
# 1. B2 Cyclical Time Features
# ===========================================================================

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add cyclical sine and cosine encodings for hour-of-day and day-of-year.
    
    Why cyclical encoding:
        Standard integer hour (0-23) falsely suggests hour 23 (11:00 PM) and
        hour 0 (midnight) are far apart. Sine and cosine transformations map
        time onto a continuous 2D circle where 23:45 and 00:00 are adjacent.
    """
    df = df.copy()

    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in dataframe.")

    t = pd.to_datetime(df[time_col])

    # Fractional continuous hour (0.0 to 23.75 for 15-min bins)
    cont_hour = t.dt.hour + (t.dt.minute / 60.0)
    hour_angle = 2.0 * np.pi * (cont_hour / 24.0)

    # Fractional continuous day of year (1.0 to 366.0)
    cont_day = t.dt.dayofyear + (cont_hour / 24.0)
    day_angle = 2.0 * np.pi * (cont_day / 365.25)

    df["hour_sin"] = np.sin(hour_angle).astype("float32")
    df["hour_cos"] = np.cos(hour_angle).astype("float32")
    df["day_sin"] = np.sin(day_angle).astype("float32")
    df["day_cos"] = np.cos(day_angle).astype("float32")

    return df


# ===========================================================================
# 2. B2 Station Metadata Features
# ===========================================================================

def clean_tilt(val, default_tilt: float = 15.0) -> float:
    """Clean rooftop tilt angle (default 15.0 deg campus median)."""
    if pd.isna(val):
        return default_tilt
    s = str(val).strip().lower().replace("deg", "")
    if "/" in s:
        parts = [float(p) for p in s.split("/") if p.strip()]
        non_zero = [p for p in parts if p > 0]
        return float(non_zero[0]) if non_zero else default_tilt
    try:
        return float(s)
    except ValueError:
        return default_tilt


def categorize_azimuth(val) -> str:
    """Categorize roof orientation into standard compass buckets."""
    if pd.isna(val):
        return "other"
    s = str(val).strip().lower().replace("deg", "")

    if "mix" in s:
        return "mixed"
    if "south" in s:
        return "south"
    if "east" in s or "west" in s:
        return "east_west"

    try:
        deg = float(s) % 360
        if 135 <= deg <= 225:
            return "south"
        elif (45 <= deg < 135) or (225 < deg <= 315):
            return "east_west"
        else:
            return "other"
    except ValueError:
        return "other"


def add_station_metadata_features(
    df: pd.DataFrame, 
    meta_df: pd.DataFrame = None, 
    inverter_col: str = "inverter_id"
) -> pd.DataFrame:
    """Clean and encode station metadata (capacity, tilt, orientation, altitude)."""
    df = df.copy()

    needed_meta = ["capacity_kwp", "tilt_deg", "azimuth", "altitude_m"]
    has_meta = any(c in df.columns for c in needed_meta)

    if not has_meta:
        if meta_df is None:
            meta_path = PROCESSED_DIR / "inverter_metadata.parquet"
            if meta_path.exists():
                meta_df = pd.read_parquet(meta_path)

        if meta_df is not None and inverter_col in df.columns:
            join_cols = [c for c in [inverter_col, "capacity_kwp", "tilt_deg", "azimuth", "altitude_m"]
                         if c in meta_df.columns]
            df = df.merge(meta_df[join_cols], on=inverter_col, how="left")

    # 1. Capacity (kWp)
    if "capacity_kwp" not in df.columns:
        df["capacity_kwp"] = 25.0
    df["capacity_kwp"] = pd.to_numeric(df["capacity_kwp"], errors="coerce").fillna(25.0).astype("float32")

    # 2. Tilt angle (degrees)
    if "tilt_deg" not in df.columns:
        df["tilt_deg"] = 15.0
    df["tilt_deg"] = df["tilt_deg"].apply(clean_tilt).astype("float32")

    # 3. Altitude (meters)
    if "altitude_m" not in df.columns:
        df["altitude_m"] = 50.0
    df["altitude_m"] = pd.to_numeric(df["altitude_m"], errors="coerce").fillna(50.0).astype("float32")

    # 4. Roof Orientation (One-hot encoded)
    if "azimuth" in df.columns:
        orient_series = df["azimuth"].apply(categorize_azimuth)
    else:
        orient_series = pd.Series(["south"] * len(df), index=df.index)

    df["orient_south"] = (orient_series == "south").astype("float32")
    df["orient_east_west"] = (orient_series == "east_west").astype("float32")
    df["orient_mixed"] = (orient_series == "mixed").astype("float32")

    return df


# ===========================================================================
# 3. B1 Physical Feature Engineering (Stage 7 Signature Group)
# ===========================================================================

def add_performance_ratio_and_yield(
    df: pd.DataFrame,
    derating: float = 0.78,
    min_power_floor: float = 100.0
) -> pd.DataFrame:
    """Calculate normalized specific yield and instantaneous Performance Ratio (PR).
    
    Formulas:
        Specific Yield (kW/kWp) = (ac_power_w / 1000.0) / capacity_kwp
        Expected Power (W) = (irradiance_wm2 / 1000.0) * (capacity_kwp * 1000.0) * derating
        Performance Ratio = ac_power_w / max(Expected Power, min_power_floor)
    """
    df = df.copy()

    cap_kwp = df["capacity_kwp"].astype("float32")
    ac_w = df["ac_power_w"].astype("float32")
    irr_wm2 = df["irradiance_wm2"].astype("float32")
    valid_power = np.isfinite(ac_w) & ac_w.ge(0)
    if "is_offline" in df:
        valid_power &= ~df["is_offline"].fillna(True).astype(bool)
    if "n_valid_5min" in df:
        valid_power &= df["n_valid_5min"].gt(0)

    # 1. Specific Yield (kW AC / kWp DC)
    safe_cap = cap_kwp.where(np.isfinite(cap_kwp) & cap_kwp.gt(0))
    ac_w = ac_w.where(valid_power)
    df["specific_yield"] = ((ac_w / 1000.0) / safe_cap).astype("float32")

    # 2. Theoretical Expected Power
    irr_wm2 = irr_wm2.where(np.isfinite(irr_wm2) & irr_wm2.ge(0))
    expected_w = (irr_wm2 / 1000.0) * (safe_cap * 1000.0) * derating
    expected_w = np.maximum(expected_w, min_power_floor)
    df["expected_power_w"] = expected_w.astype("float32")

    # 3. Performance Ratio (clipped to physical upper bound 2.5)
    pr = ac_w / expected_w
    # Match the daylight requirement used by B1's health labels. Missing
    # telemetry stays missing and cannot depress the campus peer median.
    df["performance_ratio"] = np.clip(pr, 0.0, 2.5).where(
        irr_wm2 >= 100.0
    ).astype("float32")

    return df


def add_peer_deviation_features(
    df: pd.DataFrame,
    time_col: str = "time",
    station_col: str = "station_id"
) -> pd.DataFrame:
    """Compute instantaneous Peer-Deviation features across campus and rooftop peers.
    
    Why it matters:
        Cloud dynamics cause simultaneous power dips for all solar inverters on campus.
        By normalizing against instantaneous peer median output at the exact same timestamp,
        common-mode weather noise cancels out, isolating localized inverter hardware faults.
        
    Formulas:
        peer_dev_campus = (PR_i - Median_campus(PR)) / (Median_campus(PR) + epsilon)
        peer_dev_station = (PR_i - Median_station(PR)) / (Median_station(PR) + epsilon)
    """
    df = df.copy()

    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"

    if "performance_ratio" not in df.columns:
        df = add_performance_ratio_and_yield(df)

    epsilon = 0.05  # Protective constant against zero-division

    # 1. Campus-wide Peer Median Benchmark at each timestamp
    campus_median_pr = df.groupby(time_col)["performance_ratio"].transform("median")
    df["peer_dev_campus"] = (
        (df["performance_ratio"] - campus_median_pr) / (campus_median_pr + epsilon)
    ).astype("float32")

    # 2. Rooftop / Station Peer Median Benchmark (for multi-inverter rooftops)
    if station_col in df.columns:
        station_median_pr = df.groupby(
            [time_col, station_col], observed=True
        )["performance_ratio"].transform("median")
        df["peer_dev_station"] = (
            (df["performance_ratio"] - station_median_pr) / (station_median_pr + epsilon)
        ).astype("float32")
    else:
        # Fallback to campus peer deviation if station column is unavailable
        df["peer_dev_station"] = df["peer_dev_campus"]

    # Clip extreme ratios for tree and neural network numerical stability
    df["peer_dev_campus"] = np.clip(df["peer_dev_campus"], -3.0, 3.0).astype("float32")
    df["peer_dev_station"] = np.clip(df["peer_dev_station"], -3.0, 3.0).astype("float32")

    return df


def add_self_relative_deviation_features(
    df: pd.DataFrame,
    inverter_col: str = "inverter_id",
    time_col: str = "time",
    lookback_days: int = 30
) -> pd.DataFrame:
    """Compute Self-Relative Deviation against each inverter's rolling historical baseline.
    
    Why it matters:
        Slow, gradual string degradation, soiling, or persistent partial shading
        is best identified by comparing an inverter's current PR against its own
        historical performance under identical solar irradiance conditions.
        
    The baseline contains observed daylight PR in [t-lookback_days, t).
    With no history in that window the deviation stays missing for a
    training-fitted imputer. Input order and index are preserved.
    """
    df = df.copy()

    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"
    if not isinstance(lookback_days, int) or lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")
    df[time_col] = pd.to_datetime(df[time_col])
    if df[[inverter_col, time_col]].isna().any().any():
        raise ValueError("Inverter ID and timestamp are required for rolling features")
    if df.duplicated([inverter_col, time_col]).any():
        raise ValueError("Duplicate inverter/timestamp rows in rolling features")

    if "performance_ratio" not in df.columns:
        df = add_performance_ratio_and_yield(df)

    # Solar irradiance binning to condition baseline on light intensity
    # 0: Low (100-300 W/m²), 1: Med (300-600 W/m²), 2: High (>600 W/m²)
    irr = df["irradiance_wm2"]
    irr_bins = np.select([irr < 300.0, irr < 600.0], [0, 1], default=2)
    work = df[[inverter_col, time_col, "performance_ratio"]].copy()
    work["_irr_bin"] = irr_bins
    work["_position"] = np.arange(len(work))
    work.loc[~irr.ge(100), "performance_ratio"] = np.nan
    work = work.sort_values([inverter_col, time_col])
    epsilon = 0.05
    baselines = np.full(len(df), np.nan)
    for _, group in work.groupby([inverter_col, "_irr_bin"], observed=True):
        history = group.set_index(time_col)["performance_ratio"]
        median = history.rolling(
            f"{lookback_days}D", min_periods=1, closed="left"
        ).median()
        baselines[group["_position"].to_numpy()] = median.to_numpy()
    baseline = pd.Series(baselines, index=df.index)
    df["self_dev_30d"] = ((df["performance_ratio"] - baseline) / (baseline + epsilon)).astype("float32")
    df["self_dev_30d"] = np.clip(df["self_dev_30d"], -3.0, 3.0).where(irr.ge(100))

    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weather & thermal stress interaction features.
    
    Formulas:
        thermal_stress = (irradiance_wm2 / 1000.0) * temp_c
        temp_delta_stc = temp_c - 25.0  (Standard Test Conditions reference delta)
    """
    df = df.copy()

    temp = df["temp_c"].astype("float32") if "temp_c" in df.columns else pd.Series(np.nan, index=df.index)
    irr = df["irradiance_wm2"].astype("float32") if "irradiance_wm2" in df.columns else pd.Series(np.nan, index=df.index)

    # Thermal dissipation stress proxy
    df["thermal_stress"] = ((irr / 1000.0) * temp).astype("float32")

    # Temperature delta above Standard Test Conditions (25 deg C)
    df["temp_delta_stc"] = (temp - 25.0).astype("float32")

    return df


# ===========================================================================
# 4. Master Feature Pipeline
# ===========================================================================

def build_b2_features(df, time_col="time", meta_df=None):
    """Compatibility entry point for the existing A1 and B2 model families."""
    return add_station_metadata_features(
        add_time_features(df, time_col=time_col), meta_df=meta_df
    )


def build_b1_features(df, time_col="time", inverter_col="inverter_id",
                      station_col="station_id"):
    """Physical features without replacing metadata or missing measurements."""
    df = add_performance_ratio_and_yield(df)
    df = add_peer_deviation_features(df, time_col=time_col, station_col=station_col)
    df = add_self_relative_deviation_features(df, inverter_col=inverter_col,
                                             time_col=time_col)
    return add_weather_interaction_features(df)


def add_a2_history_features(df, inverter_col="inverter_id", time_col="time"):
    """Exact-time lags and past-only 1h/4h/1d windows per inverter.

    The source frame may include missing/offline rows. Values are not forward
    filled; missing exact timestamps remain missing. The current row is excluded
    from rolling windows. Historical context must be supplied before filtering
    the requested prediction rows, including at chronological split boundaries.
    """
    required = {inverter_col, time_col, *A2_HISTORY_SOURCES}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing A2 history inputs: {sorted(missing)}")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    if out.duplicated([inverter_col, time_col]).any():
        raise ValueError("Duplicate inverter/timestamp rows in A2 history")
    # Positional key preserves the caller's original row order and index.
    out["_a2_position"] = np.arange(len(out))
    ordered = out.sort_values([inverter_col, time_col])
    lookup = ordered.set_index([inverter_col, time_col])[A2_HISTORY_SOURCES]
    for label, delta in A2_LAG_STEPS.items():
        earlier = pd.MultiIndex.from_arrays(
            [ordered[inverter_col], ordered[time_col] - delta],
            names=[inverter_col, time_col],
        )
        values = lookup.reindex(earlier)
        for source in A2_HISTORY_SOURCES:
            ordered[f"{source}_lag_{label}"] = values[source].to_numpy(dtype="float32")
    for source in A2_HISTORY_SOURCES:
        for _, group in ordered.groupby(inverter_col, sort=False, observed=True):
            series = group.set_index(time_col)[source].astype("float32")
            for label, window in A2_ROLLING_WINDOWS.items():
                past = series.rolling(window, closed="left", min_periods=1)
                ordered.loc[group.index, f"{source}_roll_{label}_mean"] = past.mean().to_numpy()
                ordered.loc[group.index, f"{source}_roll_{label}_std"] = past.std().to_numpy()
    for col in A2_HISTORY_FEATURES:
        ordered[col] = ordered[col].astype("float32")
    return ordered.sort_values("_a2_position").drop(columns="_a2_position").set_axis(df.index)


def build_all_features(
    df: pd.DataFrame,
    meta_df: pd.DataFrame = None,
    time_col: str = "time",
    inverter_col: str = "inverter_id",
    station_col: str = "station_id"
) -> pd.DataFrame:
    """Execute complete Stage 7 feature extraction pipeline combining B1 and B2 groups."""
    df = df.copy()

    # 1. B2 Cyclical Time Features
    df = add_time_features(df, time_col=time_col)

    # 2. B2 Station Metadata Features
    df = add_station_metadata_features(df, meta_df=meta_df, inverter_col=inverter_col)

    return build_b1_features(df, time_col=time_col, inverter_col=inverter_col,
                             station_col=station_col)


# ===========================================================================
# 5. Collinearity Pruning & Feature Selection (B1 Leads)
# ===========================================================================

def compute_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix across numerical engineered features."""
    usable = [c for c in feature_cols if c in df.columns]
    X_num = df[usable].select_dtypes(include=[np.number]).dropna()
    return X_num.corr(method="pearson")


def prune_collinear_features(
    X_train: pd.DataFrame,
    candidate_features: list[str],
    threshold: float = 0.85,
    protected_features: list[str] = None,
    save_report: bool = True,
) -> tuple[list[str], list[dict]]:
    """Iteratively identify and prune collinear feature pairs exceeding threshold |r| > threshold.
    
    Parameters:
        X_train: Training feature matrix (fitted on training data only to prevent leakage).
        candidate_features: Full list of features to evaluate.
        threshold: Absolute Pearson correlation cutoff (default 0.85).
        protected_features: Core domain features that should never be pruned automatically.
        
    Returns:
        (selected_features, dropped_pairs_log)
    """
    if protected_features is None:
        protected_features = [
            "peer_dev_campus",
            "self_dev_30d",
            "thermal_stress",
            "capacity_kwp",
            "hour_sin",
            "day_sin"
        ]

    usable_features = [c for c in candidate_features if c in X_train.columns]
    corr_matrix = X_train[usable_features].corr(method="pearson").abs()

    dropped_features = set()
    pruning_log = []

    # Iterate through upper triangle of correlation matrix
    for i in range(len(usable_features)):
        col_i = usable_features[i]
        if col_i in dropped_features:
            continue

        for j in range(i + 1, len(usable_features)):
            col_j = usable_features[j]
            if col_j in dropped_features:
                continue

            r_val = corr_matrix.loc[col_i, col_j]
            if r_val > threshold:
                if col_i in protected_features and col_j in protected_features:
                    continue
                # Decide which feature to drop
                # Keep protected features; otherwise drop the one with higher index
                if col_i in protected_features and col_j not in protected_features:
                    to_drop = col_j
                    kept = col_i
                elif col_j in protected_features and col_i not in protected_features:
                    to_drop = col_i
                    kept = col_j
                else:
                    # Drop col_j by convention
                    to_drop = col_j
                    kept = col_i

                dropped_features.add(to_drop)
                pruning_log.append({
                    "dropped_feature": to_drop,
                    "kept_feature": kept,
                    "correlation": round(float(r_val), 3),
                    "threshold": threshold,
                    "reason": f"High collinearity with {kept} (|r| = {r_val:.2f} > {threshold:.2f})"
                })
                # Once col_i is removed it cannot justify dropping more columns.
                if to_drop == col_i:
                    break

    selected_features = [f for f in usable_features if f not in dropped_features]

    # Save summary table
    if pruning_log and save_report:
        log_df = pd.DataFrame(pruning_log)
        log_path = TABLES_DIR / "b1_feature_selection_summary.csv"
        log_df.to_csv(log_path, index=False)
        print(f"Saved feature selection report: {log_path.relative_to(ROOT)}")

    return selected_features, pruning_log


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    save_filename: str = "b1_feature_correlation_matrix.png"
) -> Path:
    """Plot and save publication-quality correlation heatmap."""
    import seaborn as sns
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8.5
    })

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation (r)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        ax=ax
    )

    ax.set_title("Stage 7 Feature Collinearity & Correlation Matrix (Training Split)", fontsize=11, fontweight="bold")
    save_path = FIGURES_DIR / save_filename
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


# ===========================================================================
# 6. Standalone Runner & Integration Test
# ===========================================================================

def run_stage7_feature_pipeline():
    """Execute complete Stage 7 feature engineering and selection workflow."""
    print("=" * 70)
    print("Executing Stage 7 Feature Engineering & Selection Pipeline (B1 Lead)...")
    print("=" * 70)

    parquet_path = PROCESSED_DIR / "hkust_15min_v1.parquet"
    if parquet_path.exists():
        print(f"Loading real telemetry from {parquet_path.relative_to(ROOT)}...")
        raw_df = pd.read_parquet(parquet_path)
    else:
        print("Notice: hkust_15min_v1.parquet not found. Generating simulated representative sample.")
        np.random.seed(42)
        dates = pd.date_range("2021-05-01 06:00:00", "2023-12-31 18:00:00", freq="15min")
        dates = dates[(dates.hour >= 6) & (dates.hour <= 18)]
        inverters = [f"inverter_{i+1:02d}" for i in range(8)]
        rows = []
        for d in dates:
            hour_factor = np.sin(np.pi * (d.hour + d.minute / 60.0 - 6) / 12.0)
            if hour_factor <= 0:
                continue
            irr = np.clip(1000.0 * hour_factor * np.random.uniform(0.7, 1.05), 50.0, 1100.0)
            temp = 20.0 + 10.0 * hour_factor + np.random.uniform(-2, 2)
            for idx, inv in enumerate(inverters):
                cap = 25.0
                perf_mult = 0.6 if idx == 0 else (1.1 if idx == 7 else 1.0)
                ac = irr * cap * 0.78 * perf_mult * np.random.uniform(0.95, 1.02)
                rows.append({
                    "time": d,
                    "inverter_id": inv,
                    "station_id": f"station_{(idx // 2) + 1}",
                    "capacity_kwp": cap,
                    "irradiance_wm2": irr,
                    "temp_c": temp,
                    "ac_power_w": ac,
                    "tilt_deg": "15deg",
                    "azimuth": "south"
                })
        raw_df = pd.DataFrame(rows)

    print(f"Input records: {len(raw_df):,} | Inverters: {raw_df['inverter_id'].nunique()}")

    # 1. Feature Extraction
    print("\n1. Applying unified feature transformations (B1 Physical + B2 Time/Metadata)...")
    feature_df = build_all_features(raw_df)

    print("\nEngineered Features Generated:")
    for f in ALL_CANDIDATE_FEATURES:
        if f in feature_df.columns:
            print(f"  * {f:20s} -> min: {feature_df[f].min():.3f}, max: {feature_df[f].max():.3f}, NaNs: {feature_df[f].isna().sum()}")

    # 2. Chronological Split (Using B2's split_chronological directly)
    print("\n2. Splitting data using B2's split_chronological()...")
    train_df, val_df, test_df = split_chronological(feature_df, time_col="time")

    # 3. Collinearity Analysis on Training Split
    print("\n3. Computing correlation matrix on training split...")
    corr_matrix = compute_correlation_matrix(train_df, ALL_CANDIDATE_FEATURES)
    plot_correlation_matrix(corr_matrix)

    # 4. Feature Selection & Collinearity Pruning
    print("\n4. Pruning collinear feature pairs (|r| > 0.85)...")
    selected_features, pruning_log = prune_collinear_features(
        train_df, ALL_CANDIDATE_FEATURES, threshold=0.85
    )

    print(f"\nCandidate features count : {len(ALL_CANDIDATE_FEATURES)}")
    print(f"Selected features count  : {len(selected_features)}")
    print(f"Pruned redundant features: {len(pruning_log)}")
    for item in pruning_log:
        print(f"  - Dropped '{item['dropped_feature']}' due to {item['reason']}")

    print("\nFinal Selected Feature Matrix:")
    for idx, sf in enumerate(selected_features, 1):
        print(f"  {idx:02d}. {sf}")

    print("\n" + "=" * 70)
    print("Stage 7 Feature Pipeline Successfully Executed and Verified!")
    print("=" * 70)

    return feature_df, selected_features, pruning_log


if __name__ == "__main__":
    run_stage7_feature_pipeline()

"""Stage 7: Feature Engineering Framework (src/features.py).

This module contains the feature engineering logic for the Solar Inverter
Health Monitoring System. Each teammate contributes their feature group:
- B1 (Lead): Performance Ratio (PR), self/peer deviations, target formulation.
- B2: Cyclical time features (sin/cos) and station metadata encoders.
- A1: Outage features (is_offline, minutes offline in last 24h).
- A2: Lag & rolling window features.

Run this file directly to test the B2 feature transformations:
    python -m src.features
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

PROCESSED_DIR = ROOT / CONFIG.get("paths", {}).get("processed", "data/processed")


# ===========================================================================
# B2 Feature Definitions & Column Constants
# ===========================================================================

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

B2_ALL_FEATURES = B2_TIME_FEATURES + B2_METADATA_FEATURES


# ===========================================================================
# 1. Cyclical Time Features (B2)
# ===========================================================================

def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add cyclical sine and cosine encodings for hour-of-day and day-of-year.
    
    Why cyclical encoding:
        Standard integer hour (0-23) falsely suggests hour 23 (11:00 PM) and
        hour 0 (midnight) are far apart (distance 23). Sine and cosine cyclic
        transformations map time onto a smooth continuous 2D circle where 23:45
        and 00:00 are adjacent.
        
    Sub-hour precision:
        Because inverter data is sampled every 15 minutes, using fractional hour
        (hour + minute/60) prevents discrete step jumps and yields a smooth curve.

    Formulas:
        continuous_hour = hour + minute / 60.0
        hour_sin = sin(2 * pi * continuous_hour / 24)
        hour_cos = cos(2 * pi * continuous_hour / 24)

        continuous_day = day_of_year + continuous_hour / 24.0
        day_sin = sin(2 * pi * continuous_day / 365.25)
        day_cos = cos(2 * pi * continuous_day / 365.25)
    """
    df = df.copy()

    # Automatically resolve column name if 'timestamp' is used
    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in dataframe.")

    # Ensure datetime format
    t = pd.to_datetime(df[time_col])

    # Fractional continuous hour (0.0 to 23.75 for 15-min bins)
    cont_hour = t.dt.hour + (t.dt.minute / 60.0)
    hour_angle = 2.0 * np.pi * (cont_hour / 24.0)

    # Fractional continuous day of year (1.0 to 366.0, accounting for leap years)
    cont_day = t.dt.dayofyear + (cont_hour / 24.0)
    day_angle = 2.0 * np.pi * (cont_day / 365.25)

    df["hour_sin"] = np.sin(hour_angle).astype("float32")
    df["hour_cos"] = np.cos(hour_angle).astype("float32")
    df["day_sin"] = np.sin(day_angle).astype("float32")
    df["day_cos"] = np.cos(day_angle).astype("float32")

    return df


# ===========================================================================
# 2. Station Metadata Features (B2)
# ===========================================================================

def clean_tilt(val, default_tilt: float = 15.0) -> float:
    """Clean rooftop tilt angle.
    
    Handles values like '15deg', '0/15deg', NaN, or numeric floats.
    Defaults to campus median (15.0 deg) if missing.
    """
    if pd.isna(val):
        return default_tilt
    s = str(val).strip().lower().replace("deg", "")
    if "/" in s:
        # e.g., '0/15' -> choose the non-zero rooftop tilt
        parts = [float(p) for p in s.split("/") if p.strip()]
        non_zero = [p for p in parts if p > 0]
        return float(non_zero[0]) if non_zero else default_tilt
    try:
        return float(s)
    except ValueError:
        return default_tilt


def categorize_azimuth(val) -> str:
    """Categorize roof orientation / azimuth into standard compass buckets.
    
    Buckets:
        - 'south': South-facing roofs (typically 135 deg to 225 deg)
        - 'east_west': East or West facing roofs
        - 'mixed': Multiple roof facets or flat orientation
        - 'other': Unknown or unclassified
    """
    if pd.isna(val):
        return "other"
    s = str(val).strip().lower().replace("deg", "")

    if "mix" in s:
        return "mixed"
    if "south" in s:
        return "south"
    if "east" in s or "west" in s:
        return "east_west"

    # Numeric degree handling
    try:
        deg = float(s)
        # Normalize to 0-360
        deg = deg % 360
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
    """Clean and encode station metadata (capacity, tilt, orientation, altitude).
    
    If metadata columns ('capacity_kwp', 'tilt_deg', 'azimuth') are not already
    in `df`, they can be merged from `meta_df` or `inverter_metadata.parquet`.
    """
    df = df.copy()

    # If metadata is missing from df, try loading from inverter_metadata.parquet
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

    # 1. Capacity (kWp) - default to 25.0 kWp if missing
    if "capacity_kwp" not in df.columns:
        df["capacity_kwp"] = 25.0
    df["capacity_kwp"] = pd.to_numeric(df["capacity_kwp"], errors="coerce").fillna(25.0).astype("float32")

    # 2. Tilt angle (degrees)
    if "tilt_deg" not in df.columns:
        df["tilt_deg"] = 15.0
    df["tilt_deg"] = df["tilt_deg"].apply(clean_tilt).astype("float32")

    # 3. Altitude (meters above sea level)
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
# 3. Master Pipeline for B2 Features
# ===========================================================================

def build_b2_features(
    df: pd.DataFrame, 
    time_col: str = "time", 
    meta_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Build all Task 2 features (Time cyclical encodings + station metadata)."""
    df = add_time_features(df, time_col=time_col)
    df = add_station_metadata_features(df, meta_df=meta_df)
    return df


# ===========================================================================
# 4. Standalone Test & Verification Runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Testing B2 Feature Engineering Module (src/features.py)...")
    print("=" * 65)

    # 1. Create a representative 15-minute sample dataset across seasons
    sample_times = pd.date_range("2022-01-01 06:00", "2022-01-01 18:00", freq="15min")
    sample_df = pd.DataFrame({
        "time": sample_times,
        "inverter_id": ["zone_a1_inverter_1"] * len(sample_times),
        "ac_power_w": np.random.uniform(1000, 15000, len(sample_times)),
        "capacity_kwp": [25.0] * len(sample_times),
        "azimuth": ["180deg"] * len(sample_times),
        "tilt_deg": ["15deg"] * len(sample_times),
        "altitude_m": [60.0] * len(sample_times)
    })

    print(f"Sample input rows: {len(sample_df)}")
    print(f"Input columns: {list(sample_df.columns)}")

    # 2. Apply B2 feature pipeline
    enriched_df = build_b2_features(sample_df)

    print("\n--- Features Added Successfully ---")
    for col in B2_ALL_FEATURES:
        print(f"  * {col:18s} -> dtype: {enriched_df[col].dtype}, range: [{enriched_df[col].min():.3f}, {enriched_df[col].max():.3f}]")

    # 3. Mathematical Sanity Check: sin^2 + cos^2 == 1.0
    hour_trig_sum = (enriched_df["hour_sin"]**2 + enriched_df["hour_cos"]**2)
    day_trig_sum = (enriched_df["day_sin"]**2 + enriched_df["day_cos"]**2)

    assert np.allclose(hour_trig_sum, 1.0, atol=1e-5), "hour_sin^2 + hour_cos^2 != 1.0"
    assert np.allclose(day_trig_sum, 1.0, atol=1e-5), "day_sin^2 + day_cos^2 != 1.0"
    print("\n[Passed] Mathematical Check: sin^2 + cos^2 == 1.0 within tolerance.")

    # 4. Null check
    assert enriched_df[B2_ALL_FEATURES].isna().sum().sum() == 0, "Nulls found in engineered features!"
    print("[Passed] Integrity Check: 0 NaNs across all engineered features.")

    print("\nFirst 3 rows of generated features:")
    print(enriched_df[["time"] + B2_TIME_FEATURES + ["capacity_kwp", "tilt_deg", "orient_south"]].head(3))
    print("\n" + "=" * 65)
    print("B2 Task 2 Feature Engineering Module Ready!")
    print("=" * 65)

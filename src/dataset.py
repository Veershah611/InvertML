"""Stages 4 and 7: cleaning, health-score target, features and splits.

hkust_15min.parquet -> data/processed/model_table.parquet, the one table that
every model reads. One row = one inverter at one 15-minute time t.

Target    health_lead : health score one hour later, Health(t + 1 h)
          status_lead : Healthy (>= 90) / Warning (70-90) / Faulty (< 70)
Features  everything known at time t (see FEATURES below)

Run: python -m src.dataset
"""
import numpy as np
import pandas as pd

from src.common import CONFIG, PROCESSED

DERATE = 0.78            # expected power = irradiance x capacity x 0.78
HEALTH_MIN_IRR = 100     # no health score below 100 W/m2 (too dark to judge)
HEALTHY, FAULTY = 90, 70

CANDIDATE_FEATURES = [
    # current state (A1)
    "health_now", "n_valid_5min", "minutes_offline_24h",
    # weather
    "irradiance_wm2", "temp_c", "rh_pct", "wind_speed_ms",
    # time of day / year (B2)
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    # array metadata from the Brick file (B2)
    "capacity_kwp", "tilt_deg", "altitude_m", "multi_orientation",
    # physics: compare with other inverters and with own past (B1)
    "peer_dev", "self_dev_30d",
    # recent history (A2)
    "health_lag_15m", "health_lag_1d", "health_mean_4h", "peer_dev_mean_4h",
]
# Feature selection (Stage 7): multi_orientation is almost the same as tilt_deg
# (r = 0.97 on train, see reports/tables/study_correlated_features.csv), so it is dropped.
DROPPED = ["multi_orientation"]
FEATURES = [f for f in CANDIDATE_FEATURES if f not in DROPPED]


# ---------------------------------------------------------------- Stage 4

def clean(df):
    """Physical rules. Impossible values become missing; real extremes stay.
    - power below 0 or above 1.1 x capacity is a sensor error
    - irradiance below 0 or above 1500 W/m2 is a sensor error
    - offline = no valid power reading in the bin (kept as an event, not filled in)"""
    df = df.copy()
    bad_power = (df["ac_power_w"] < 0) | (df["ac_power_w"] > 1.1 * df["capacity_kwp"] * 1000)
    df.loc[bad_power, "ac_power_w"] = np.nan
    bad_irr = (df["irradiance_wm2"] < 0) | (df["irradiance_wm2"] > 1500)
    df.loc[bad_irr, "irradiance_wm2"] = np.nan
    df["is_offline"] = (df["ac_power_w"].isna() | (df["n_valid_5min"] == 0)).astype(int)
    df.loc[df["is_offline"] == 1, "ac_power_w"] = np.nan
    return df


# ---------------------------------------------------------------- Stage 7: target

def health_score(ac_power_w, irradiance_wm2, capacity_kwp):
    """Health = actual power / expected power x 100, limited to 0-100.
    Missing when it is too dark (< 100 W/m2) or the power reading is missing."""
    expected = np.maximum(irradiance_wm2 / 1000 * capacity_kwp * 1000 * DERATE, 100)
    health = np.clip(100 * ac_power_w / expected, 0, 100)
    return health.where(irradiance_wm2 >= HEALTH_MIN_IRR)


def status(health):
    """Health score -> Healthy / Warning / Faulty."""
    return np.select([health >= HEALTHY, health >= FAULTY], ["Healthy", "Warning"], "Faulty")


# ---------------------------------------------------------------- Stage 7: features

def past_rolling(df, column, window, how):
    """Per inverter: statistic of `column` over the window BEFORE t (t excluded)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("inverter_id"):
        series = g.set_index("time")[column]
        rolled = series.rolling(window, closed="left", min_periods=1).agg(how)
        out.loc[g.index] = rolled.to_numpy()
    return out


def value_at(df, column, shift):
    """Value of `column` for the same inverter at time t + shift (exact time match).
    shift = -15 min gives the previous bin; +1 h gives the value one hour later.
    If that exact time is missing the result is missing (no jumping over gaps)."""
    other = df[["inverter_id", "time", column]].rename(columns={column: "_value"})
    other["time"] -= shift
    return df[["inverter_id", "time"]].merge(other, on=["inverter_id", "time"], how="left")["_value"].to_numpy()


def add_features(df):
    t = df["time"]
    df["health_now"] = health_score(df["ac_power_w"], df["irradiance_wm2"], df["capacity_kwp"])

    # time of day and day of year as points on a circle (23:45 is next to 00:00)
    hour = t.dt.hour + t.dt.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    day = t.dt.dayofyear
    df["day_sin"], df["day_cos"] = np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25)

    # performance ratio, and how far each inverter is from the campus median at the same time
    df["pr"] = (df["ac_power_w"] / (df["irradiance_wm2"] * df["capacity_kwp"])).where(
        df["irradiance_wm2"] >= HEALTH_MIN_IRR)
    campus = df.groupby("time")["pr"].transform("median")
    df["peer_dev"] = ((df["pr"] - campus) / campus).clip(-1, 2)

    # history of the same inverter (only the past is used)
    df["health_lag_15m"] = value_at(df, "health_now", pd.Timedelta("-15min"))
    df["health_lag_1d"] = value_at(df, "health_now", pd.Timedelta("-1D"))
    df["health_mean_4h"] = past_rolling(df, "health_now", "4h", "mean")
    df["peer_dev_mean_4h"] = past_rolling(df, "peer_dev", "4h", "mean")
    df["self_dev_30d"] = df["health_now"] - past_rolling(df, "health_now", "30D", "median")

    # outage history: minutes offline in daylight during the past 24 hours
    df["offline_daylight"] = df["is_offline"] * (df["irradiance_wm2"] >= CONFIG["daylight_irradiance_wm2"])
    df["minutes_offline_24h"] = 15 * past_rolling(df, "offline_daylight", "24h", "sum").fillna(0)
    return df


def add_target(df):
    """health_lead = health of the same inverter exactly one hour later."""
    df["health_lead"] = value_at(df, "health_now", pd.Timedelta("1h"))
    return df


# ---------------------------------------------------------------- splits

def add_split(df):
    """train <= 2022-06-30 < val <= 2022-12-31 < test (2023).
    The six unseen inverters only appear in 2023 and never in train/val."""
    s = CONFIG["split"]
    df["split"] = np.select([df["time"] < pd.Timestamp(s["train_end"]) + pd.Timedelta("1D"),
                             df["time"] < pd.Timestamp(s["val_end"]) + pd.Timedelta("1D")],
                            ["train", "val"], "test")
    df["unseen"] = df["inverter_id"].isin(CONFIG["unseen_stations"])
    return df[~(df["unseen"] & (df["split"] != "test"))]


def build():
    df = pd.read_parquet(PROCESSED / "hkust_15min.parquet")
    df = df.sort_values(["inverter_id", "time"]).reset_index(drop=True)
    df = add_features(clean(df))
    df = add_target(df)
    rows = df.dropna(subset=["health_now", "health_lead"]).copy()
    rows["status_now"] = status(rows["health_now"])
    rows["status_lead"] = status(rows["health_lead"])
    rows = add_split(rows)

    keep = ["inverter_id", "station_id", "time", "split", "unseen", "ac_power_w", "pr",
            "health_lead", "status_lead", "status_now"] + CANDIDATE_FEATURES
    rows = rows[list(dict.fromkeys(keep))].reset_index(drop=True)
    rows.to_parquet(PROCESSED / "model_table.parquet", index=False)
    print(rows.groupby("split").size().to_string())
    print(f"unseen test rows: {rows['unseen'].sum():,}")
    return rows


def load():
    return pd.read_parquet(PROCESSED / "model_table.parquet")


if __name__ == "__main__":
    build()

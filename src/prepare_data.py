"""Stage 3: data preparation (A1).

Raw HKUST files -> one clean 15-minute table, data/processed/hkust_15min.parquet

1. Metadata: read the Brick .ttl graph with SPARQL -> one row per inverter
2. Inverters: read the 44 inverter CSVs (AC power, DC voltage), 5-min -> 15-min
3. Weather: read irradiance, temperature, humidity, wind, 1-min -> 15-min
4. Build the full 06:00-17:45 grid, merge, correct the irradiance sensor change
5. Gap analysis + 70 % daylight-coverage filter + before/after summary

Times are Hong Kong local time (no daylight saving).
Run: python -m src.prepare_data
"""
import re

import numpy as np
import pandas as pd
import rdflib

from src.common import CONFIG, PROCESSED, RAW, TABLES

DATASET = RAW / "Dataset"
TTL_FILE = DATASET / "Metadata" / "PV generation system metadata.ttl"
INVERTER_DIR = (DATASET / "Time series dataset" / "PV generation dataset"
                / "PV stations with panel level optimizer" / "Inverter level dataset")
WEATHER_DIR = DATASET / "Time series dataset" / "Meteorological dataset"

# weather folder -> (column in the raw file, our column name)
WEATHER = {"Irradiance": ("Irradiance (W/m2)", "irradiance_wm2"),
           "Temperature": ("Temp (Degree Celsius)", "temp_c"),
           "Relative Humidity": ("RH (%)", "rh_pct"),
           "Wind": ("Wind Speed (m/s)", "wind_speed_ms")}

# The raw files mix two timestamp formats; anything else is an error.
TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"]

# One syntax error in the raw .ttl (Zone D block) is fixed in memory only.
TTL_TYPO = ("brick:value 68.62 ] .\n    ext:azimuth", "brick:value 68.62 ] ;\n    ext:azimuth")
# One CSV file name does not match its Brick name.
NAME_FIX = {"indoor_sports_centre_inverter": "indoor_sport_center_inverter"}

SPARQL = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ext:   <https://www.HKUST_PV_System.com/schema/BrickExtension#>
SELECT ?station ?inverter ?altitude ?azimuth ?tilt ?model ?panels ?panel_w WHERE {
  ?station a brick:PV_Generation_System ; brick:hasPart ?inverter .
  ?inverter a brick:Inverter ; brick:hasPart ?panel .
  ?panel ext:number/brick:value ?panels ; ext:ratedPowerOutput/brick:value ?panel_w .
  OPTIONAL { ?station ext:altitude/brick:value ?altitude }
  OPTIONAL { ?station ext:azimuth/brick:value ?azimuth }
  OPTIONAL { ?station ext:tiltAngle/brick:value ?tilt }
  OPTIONAL { ?inverter ext:module/brick:value ?model }
}
"""

FIRST_BIN, LAST_BIN = "06:00", "17:45"   # loggers only record 06:00-17:59
SENSOR_CHANGE = pd.Timestamp(CONFIG["irradiance_sensor_change"])


def clean_name(name):
    """'SQ Apartment1-12_Inverter' -> 'sq_apartment1_12_inverter'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_time(values):
    for fmt in TIME_FORMATS:
        try:
            return pd.to_datetime(values, format=fmt)
        except ValueError:
            pass
    raise ValueError("unknown timestamp format")


# ---------------------------------------------------------------- 1. metadata

def read_metadata():
    """One row per inverter: station, capacity (kWp), tilt, orientation, altitude."""
    text = TTL_FILE.read_text(encoding="utf-8").replace("\r\n", "\n").replace(*TTL_TYPO)
    graph = rdflib.Graph().parse(data=text, format="turtle")
    rows = [{key: (value.toPython() if value is not None else None)
             for key, value in row.asdict().items()} for row in graph.query(SPARQL)]
    panels = pd.DataFrame(rows)
    panels["kwp"] = panels["panels"].astype(float) * panels["panel_w"].astype(float) / 1000

    # an inverter can drive several panel groups: add up their capacity
    meta = panels.groupby("inverter").agg(
        station_id=("station", "first"), capacity_kwp=("kwp", "sum"),
        altitude_m=("altitude", "first"), azimuth=("azimuth", "first"),
        tilt=("tilt", "first"), inverter_model=("model", "first")).reset_index()
    for col in ["inverter", "station_id"]:
        meta[col] = meta[col].astype(str).str.split("#").str[-1]
    meta["altitude_m"] = meta["altitude_m"].astype(float)

    # tilt like "10deg", "0/15deg" (two roof parts: use the average) or "Mixed" (use 5)
    tilt = meta.pop("tilt").str.replace("deg", "")
    meta["tilt_deg"] = [5.0 if t == "Mixed" else np.mean([float(x) for x in t.split("/")])
                        for t in tilt]
    # orientation like "0deg", "90/-90deg" or "Mixed": flag arrays facing several ways
    meta["multi_orientation"] = meta["azimuth"].str.contains("/|Mixed").astype(int)

    # keep the 44 inverters that have a CSV file, named after the file
    files = {NAME_FIX.get(clean_name(f.stem), clean_name(f.stem)): clean_name(f.stem)
             for f in INVERTER_DIR.glob("*.csv")}
    meta["key"] = meta.pop("inverter").map(clean_name)
    missing = set(files) - set(meta["key"])
    if missing:
        raise ValueError(f"CSV files without metadata: {missing}")
    meta = meta[meta["key"].isin(files)]
    meta.insert(0, "inverter_id", meta.pop("key").map(files))
    return meta.sort_values("inverter_id").reset_index(drop=True)


# ---------------------------------------------------------------- 2. inverters

def read_inverters():
    """All 44 inverter files, averaged to 15 minutes.
    n_valid_5min = how many of the three 5-min readings were present (0-3)."""
    frames = []
    for f in sorted(INVERTER_DIR.glob("*.csv")):
        df = pd.read_csv(f, usecols=["Time", "totalActivePower(W)", "dcVoltage(V)"],
                         na_values=["NA"])
        df = df.rename(columns={"Time": "time", "totalActivePower(W)": "ac_power_w",
                                "dcVoltage(V)": "dc_voltage_v"})
        df["time"] = parse_time(df["time"])
        df["inverter_id"] = clean_name(f.stem)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    raw["time"] = raw["time"].dt.floor("15min")          # 10:00, 10:05, 10:10 -> 10:00
    grouped = raw.groupby(["inverter_id", "time"])
    out = grouped[["ac_power_w", "dc_voltage_v"]].mean()
    out["n_valid_5min"] = grouped["ac_power_w"].count()
    return raw, out.reset_index()


# ---------------------------------------------------------------- 3. weather

def read_weather():
    """Four weather variables for 2021-2023, averaged to 15 minutes."""
    columns = []
    for folder, (raw_name, name) in WEATHER.items():
        parts = [pd.read_csv(WEATHER_DIR / folder / f"{folder}_{year}.csv",
                             usecols=["Time", raw_name], na_values=["NA"])
                 for year in (2021, 2022, 2023)]
        df = pd.concat(parts).rename(columns={"Time": "time", raw_name: name})
        df["time"] = parse_time(df["time"]).dt.floor("15min")
        columns.append(df.groupby("time")[name].mean())
    return pd.concat(columns, axis=1).reset_index()


# ---------------------------------------------------------------- 4. grid + merge

def full_grid(inv15):
    """Every 15-min bin from 06:00 to 17:45, from each inverter's first day to the
    end of 2023, so that gaps become visible rows with n_valid_5min = 0."""
    end = pd.Timestamp(CONFIG["split"]["test_end"]) + pd.Timedelta("17:45:00")
    grids = []
    for inverter, start in inv15.groupby("inverter_id")["time"].min().items():
        times = pd.date_range(start, end, freq="15min")
        times = times[(times.strftime("%H:%M") >= FIRST_BIN) & (times.strftime("%H:%M") <= LAST_BIN)]
        grids.append(pd.DataFrame({"inverter_id": inverter, "time": times}))
    grid = pd.concat(grids, ignore_index=True)
    full = grid.merge(inv15, on=["inverter_id", "time"], how="left")
    full["n_valid_5min"] = full["n_valid_5min"].fillna(0).astype(int)
    return full


def correct_sensor(df):
    """The irradiance sensor reads about 1.75x higher from 2023-02-14 (EDA).
    Factor = median over Feb-Dec of (2023 median PR / same month 2022 median PR).
    Irradiance from that day is multiplied by the factor; the raw value is kept."""
    ok = df["ac_power_w"].notna() & (df["irradiance_wm2"] >= 100)
    pr = (df["ac_power_w"] / (df["irradiance_wm2"] * df["capacity_kwp"])).where(ok)
    year, month = df["time"].dt.year, df["time"].dt.month
    rows = []
    for m in range(SENSOR_CHANGE.month, 13):
        before = pr[(year == SENSOR_CHANGE.year - 1) & (month == m)].median()
        after = pr[(year == SENSOR_CHANGE.year) & (month == m) & (df["time"] >= SENSOR_CHANGE)].median()
        rows.append({"month": m, "pr_2022": round(before, 4), "pr_2023": round(after, 4),
                     "ratio": round(after / before, 4)})
    table = pd.DataFrame(rows)
    factor = table["ratio"].median()
    df["irradiance_raw_wm2"] = df["irradiance_wm2"]
    after_change = df["time"] >= SENSOR_CHANGE
    df.loc[after_change, "irradiance_wm2"] *= factor
    return df, table, factor


# ---------------------------------------------------------------- 5. gaps

def gap_tables(df):
    """Missing bin = no valid power reading. Only daylight bins count
    (irradiance >= 50 W/m2), so the inverters' dawn sleep is not a gap."""
    day = df[df["irradiance_wm2"] >= CONFIG["daylight_irradiance_wm2"]]
    missing = day["n_valid_5min"].eq(0)
    by_month = (missing.groupby([day["inverter_id"], day["time"].dt.to_period("M").astype(str)])
                .agg(["size", "mean"]).reset_index())
    by_month.columns = ["inverter_id", "month", "daylight_bins", "missing_share"]
    coverage = (1 - missing.groupby(day["inverter_id"]).mean()).rename("coverage").reset_index()
    coverage["kept"] = coverage["coverage"] >= CONFIG["min_daylight_coverage"]
    return by_month.round(4), coverage.sort_values("coverage").round(4)


def main():
    summary = []
    step = lambda name, d: summary.append({"step": name, "rows": len(d), "columns": d.shape[1],
                                           "inverters": d["inverter_id"].nunique()})

    print("1. metadata")
    meta = read_metadata()
    meta.to_parquet(PROCESSED / "inverter_metadata.parquet", index=False)
    print(f"   {len(meta)} inverters at {meta['station_id'].nunique()} stations")

    print("2. inverters")
    raw, inv15 = read_inverters()
    step("raw 5-min inverter rows", raw)
    step("averaged to 15 min", inv15)

    print("3. weather")
    weather = read_weather()

    print("4. full grid, merge, sensor correction")
    df = full_grid(inv15)
    step("full 06:00-17:45 grid", df)
    df = df.merge(weather, on="time", how="left").merge(meta, on="inverter_id", how="left")
    step("merged with weather + metadata", df)
    df, correction, factor = correct_sensor(df)
    correction.to_csv(TABLES / "irradiance_correction.csv", index=False)
    print(f"   irradiance from {SENSOR_CHANGE.date()} multiplied by {factor:.3f}")

    print("5. gaps and coverage")
    by_month, coverage = gap_tables(df)
    by_month.to_csv(TABLES / "gaps_by_inverter_month.csv", index=False)
    coverage.to_csv(TABLES / "coverage_by_inverter.csv", index=False)
    df = df[df["inverter_id"].isin(coverage.loc[coverage["kept"], "inverter_id"])]
    step("coverage filter (>= 70 %)", df)
    print(f"   lowest coverage {coverage['coverage'].min():.1%}, "
          f"dropped {int((~coverage['kept']).sum())} inverters")

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df.to_parquet(PROCESSED / "hkust_15min.parquet", index=False)
    pd.DataFrame(summary).to_csv(TABLES / "prep_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

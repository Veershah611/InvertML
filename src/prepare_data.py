"""Stage 3: data preparation (A1).

Step 1: parse the Brick metadata (.ttl) into one row per inverter
        -> data/processed/inverter_metadata.parquet
Step 2: convert the raw CSVs to typed float32 Parquet (no resampling yet)
        -> data/processed/inverter_5min.parquet
        -> data/processed/weather_1min.parquet
Step 3: resample both to 15-min
Step 4: rebuild the full 15-min grid per inverter, merge weather + metadata,
        gap analysis, daylight-coverage filter, before/after summary
        -> data/processed/hkust_15min_v1.parquet
        -> reports/tables/{gaps_by_inverter_month, coverage_by_inverter,
                           prep_summary}.csv

Timestamps are kept as naive Hong Kong local time (HKT, UTC+8). Hong Kong has
no daylight saving, so there are no repeated or skipped hours.

Run from the repo root:  python -m src.prepare_data
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rdflib
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
RAW = ROOT / CONFIG["paths"]["raw"] / "Dataset"
PROCESSED = ROOT / CONFIG["paths"]["processed"]

TTL_PATH = RAW / "Metadata" / "PV generation system metadata.ttl"
INVERTER_DIR = (RAW / "Time series dataset" / "PV generation dataset"
                / "PV stations with panel level optimizer" / "Inverter level dataset")
WEATHER_DIR = RAW / "Time series dataset" / "Meteorological dataset"

# The raw files use two timestamp formats. We try these explicitly and fail
# loudly on anything else, instead of letting pandas guess.
TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"]

# Weather folder -> {raw column: our column}
WEATHER_COLUMNS = {
    "Irradiance": {"Irradiance (W/m2)": "irradiance_wm2"},
    "Temperature": {"Temp (Degree Celsius)": "temp_c"},
    "Wind": {"Wind Speed (m/s)": "wind_speed_ms",
             "Wind Direction (degree)": "wind_dir_deg"},
    "Relative Humidity": {"RH (%)": "rh_pct"},
    "Rainfall": {"Rainfall(mm)": "rainfall_mm"},
    "Sea Level Pressure": {"SLP (hPa)": "slp_hpa"},
    "Visibility": {"Vis (km)": "visibility_km"},
}
WEATHER_YEARS = [2021, 2022, 2023]

FREQ = CONFIG["resample_freq"]
MERGED_NAME = "hkust_15min_v1.parquet"
TABLES = ROOT / "reports" / "tables"

# The inverter loggers only record 06:00-17:59, so the 15-min grid runs from
# the 06:00 bin to the 17:45 bin. Generation after 18:00 is never recorded.
LOGGER_FIRST_BIN = "06:00"
LOGGER_LAST_BIN = "17:45"
DATA_END = pd.Timestamp(CONFIG["split"]["test_end"]) + pd.Timedelta(LOGGER_LAST_BIN + ":00")
DAYLIGHT_WM2 = CONFIG["daylight_irradiance_wm2"]
MIN_COVERAGE = CONFIG["min_daylight_coverage"]

# The raw .ttl has one syntax error: the Zone_D block ends a statement with
# "." instead of ";" after ratedPowerOutput, so rdflib refuses to parse it.
# We fix it in memory only; the raw file is never modified.
TTL_TYPO = ("brick:value 68.62 ] .\n    ext:azimuth",
            "brick:value 68.62 ] ;\n    ext:azimuth")

# CSV file names that do not match their Brick ID after normalise_name().
NAME_OVERRIDES = {
    "indoor_sports_centre_inverter": "indoor_sport_center_inverter",
}

SPARQL = """
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ext:   <https://www.HKUST_PV_System.com/schema/BrickExtension#>
SELECT ?station ?inverter ?station_rated_kw ?connection_date ?altitude_m
       ?contractor ?azimuth ?tilt ?inverter_brand ?inverter_model
       ?panel ?panel_count ?panel_w ?panel_type ?panel_efficiency
       ?deg_first_year ?deg_linear
WHERE {
  ?station a brick:PV_Generation_System ;
           brick:hasPart ?inverter .
  ?inverter a brick:Inverter .
  OPTIONAL { ?station ext:ratedPowerOutput/brick:value ?station_rated_kw }
  OPTIONAL { ?station ext:connectionDate/brick:value ?connection_date }
  OPTIONAL { ?station ext:altitude/brick:value ?altitude_m }
  OPTIONAL { ?station ext:contractor/brick:value ?contractor }
  OPTIONAL { ?station ext:azimuth/brick:value ?azimuth }
  OPTIONAL { ?station ext:tiltAngle/brick:value ?tilt }
  OPTIONAL { ?inverter ext:brand/brick:value ?inverter_brand }
  OPTIONAL { ?inverter ext:module/brick:value ?inverter_model }
  OPTIONAL {
    ?inverter brick:hasPart ?panel .
    ?panel a brick:PV_Panel .
    OPTIONAL { ?panel ext:number/brick:value ?panel_count }
    OPTIONAL { ?panel ext:ratedPowerOutput/brick:value ?panel_w }
    OPTIONAL { ?panel ext:specificType/brick:value ?panel_type }
    OPTIONAL { ?panel ext:ratedModuleConversionEfficiency/brick:value ?panel_efficiency }
    OPTIONAL { ?panel ext:panelDegradationFirstYear/brick:value ?deg_first_year }
    OPTIONAL { ?panel ext:panelLinearDegradation/brick:value ?deg_linear }
  }
}
"""


def normalise_name(name):
    """'SQ Apartment1-12_Inverter' -> 'sq_apartment1_12_inverter'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_brick_graph(path=TTL_PATH):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if TTL_TYPO[0] in text:
        print("Fixing known .ttl syntax error in Zone_D block (in memory only)")
        text = text.replace(*TTL_TYPO)
    return rdflib.Graph().parse(data=text, format="turtle")


def query_panels(graph):
    """One row per (inverter, panel group), as returned by SPARQL."""
    rows = []
    for r in graph.query(SPARQL):
        row = {k: (v.toPython() if v is not None else None)
               for k, v in r.asdict().items()}
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["station", "inverter", "panel"]:
        df[col] = df[col].astype(str).str.split("#").str[-1]
    # rdflib returns Decimal objects for numbers; convert them to float.
    numeric = ["station_rated_kw", "altitude_m", "panel_count", "panel_w",
               "panel_efficiency", "deg_first_year", "deg_linear"]
    df[numeric] = df[numeric].apply(pd.to_numeric).astype("float64")
    return df


def build_inverter_metadata(graph):
    """Collapse panel groups into one row per inverter (all 67 in the .ttl)."""
    panels = query_panels(graph)
    panels["group_kwp"] = panels["panel_count"] * panels["panel_w"] / 1000

    # Panel attributes come from the largest panel group of each inverter.
    largest = (panels.sort_values("group_kwp", ascending=False)
                     .drop_duplicates("inverter"))
    totals = panels.groupby("inverter").agg(
        capacity_kwp=("group_kwp", "sum"),
        panel_count=("panel_count", "sum"),
    )
    meta = largest.drop(columns=["panel", "panel_count", "group_kwp"]).join(
        totals, on="inverter")

    meta = meta.rename(columns={"inverter": "brick_inverter",
                                "station": "station_id",
                                "deg_first_year": "degradation_first_year_pct",
                                "deg_linear": "degradation_linear_pct"})
    meta["tilt_deg"] = pd.to_numeric(
        meta.pop("tilt").str.replace("deg", "", regex=False), errors="coerce")
    # Dates appear as both "2021-01-30T00:00:00" and "2021-08-31 00:00:00".
    meta["connection_date"] = pd.to_datetime(meta["connection_date"], format="ISO8601")
    meta["capacity_kwp"] = meta["capacity_kwp"].round(3)
    return meta


def match_inverter_files(meta, inverter_dir=INVERTER_DIR):
    """Keep only inverters that have a CSV file; fail if any file is unmatched."""
    files = sorted(inverter_dir.glob("*.csv"))
    file_ids = {}
    for f in files:
        key = normalise_name(f.stem)
        file_ids[NAME_OVERRIDES.get(key, key)] = f.name

    meta = meta.copy()
    meta["match_key"] = meta["brick_inverter"].map(normalise_name)
    unmatched = sorted(set(file_ids) - set(meta["match_key"]))
    if unmatched:
        raise ValueError(f"CSV files with no Brick inverter: {unmatched}")

    meta = meta[meta["match_key"].isin(file_ids)].copy()
    meta["file_name"] = meta["match_key"].map(file_ids)
    meta["inverter_id"] = meta["file_name"].str.removesuffix(".csv").map(normalise_name)
    meta = meta.drop(columns="match_key")

    first = ["inverter_id", "station_id", "capacity_kwp", "station_rated_kw"]
    meta = meta[first + [c for c in meta.columns if c not in first]]
    return meta.sort_values("inverter_id").reset_index(drop=True)


def parse_time(series, source):
    """Parse a timestamp column using the first format that fits every value."""
    for fmt in TIME_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"{source}: timestamps match none of {TIME_FORMATS}")


def inverter_column_name(raw):
    """'L1_acCurrent(A)' -> 'l1_ac_current_a'; 'totalActivePower(W)' -> 'ac_power_w'.

    Note: the raw files label reactive power in W; strictly it is var.
    """
    if raw == "totalActivePower(W)":
        return "ac_power_w"
    name, unit = re.match(r"(.+)\((.+)\)", raw).groups()
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()
    return f"{name}_{unit.lower()}"


def convert_inverters(inverter_dir=INVERTER_DIR):
    """Stack the 44 inverter CSVs into one long float32 table."""
    frames = []
    for f in sorted(inverter_dir.glob("*.csv")):
        df = pd.read_csv(f, na_values=["NA"])
        time = parse_time(df.pop("Time"), f.name)
        df = df.rename(columns=inverter_column_name)
        df.insert(0, "time", time)
        values = df.columns.drop("time")
        df[values] = df[values].astype("float32")
        df.insert(0, "inverter_id", normalise_name(f.stem))
        print(f"  {f.name:48s} rows={len(df):7d}  {df['time'].min()} -> "
              f"{df['time'].max()}  NaN cells={int(df[values].isna().sum().sum())}")
        frames.append(df)

    inv = pd.concat(frames, ignore_index=True)
    inv["inverter_id"] = inv["inverter_id"].astype("category")
    return inv


def read_weather_file(folder, year, weather_dir=WEATHER_DIR):
    path = weather_dir / folder / f"{folder}_{year}.csv"
    if path.exists():
        df = pd.read_csv(path, na_values=["NA"])
        df["Time"] = parse_time(df["Time"], path.name)
    else:  # Rainfall_2021 is only provided as .xlsx (Time is already datetime)
        path = path.with_suffix(".xlsx")
        df = pd.read_excel(path, na_values=["NA"])
    return df.rename(columns={"Time": "time", **WEATHER_COLUMNS[folder]}), path


def convert_weather():
    """Join the 7 weather variables on time; stack 2021-2023."""
    variables = []
    for folder in WEATHER_COLUMNS:
        parts = []
        for year in WEATHER_YEARS:
            df, path = read_weather_file(folder, year)
            print(f"  {path.name:32s} rows={len(df):7d}")
            parts.append(df)
        var = pd.concat(parts, ignore_index=True)
        dups = var["time"].duplicated().sum()
        if dups:
            raise ValueError(f"{folder}: {dups} duplicate timestamps; "
                             "decide how to handle before joining")
        variables.append(var.set_index("time"))

    weather = pd.concat(variables, axis=1, join="outer").sort_index()
    return weather.astype("float32").reset_index()


def resample_inverters(inv, freq=FREQ):
    """5-min -> 15-min mean per inverter.

    Bins are labelled by their start (10:00 covers 10:00, 10:05, 10:10).
    n_valid_5min counts the non-NaN ac_power_w readings in each bin, so the
    gap policy in Stage 4 can decide how many readings a bin needs.
    """
    bins = inv["time"].dt.floor(freq).rename("time")
    grouped = inv.drop(columns="time").groupby(
        [inv["inverter_id"], bins], observed=True)
    out = grouped.mean()
    out["n_valid_5min"] = grouped["ac_power_w"].count().astype("int8")
    return out.reset_index()


def resample_weather(weather, freq=FREQ):
    """1-min -> 15-min: mean of readings, sum of rainfall, circular mean of
    wind direction. n_valid_1min counts valid irradiance readings (0-15)."""
    df = weather.copy()
    df["time"] = df["time"].dt.floor(freq)
    rad = np.deg2rad(df["wind_dir_deg"])
    df["wind_sin"], df["wind_cos"] = np.sin(rad), np.cos(rad)

    g = df.groupby("time")
    mean_cols = ["irradiance_wm2", "temp_c", "wind_speed_ms", "rh_pct",
                 "slp_hpa", "visibility_km"]
    out = g[mean_cols].mean()
    out["rainfall_mm"] = g["rainfall_mm"].sum(min_count=1)
    # Circular mean of wind direction via the mean of its sin and cos
    # (a plain mean of 350 and 10 degrees would give 180 instead of 0).
    out["wind_dir_deg"] = np.rad2deg(np.arctan2(g["wind_sin"].mean(),
                                                g["wind_cos"].mean())) % 360
    out["n_valid_1min"] = g["irradiance_wm2"].count().astype("int8")
    return out.astype({c: "float32" for c in out.columns if c != "n_valid_1min"}).reset_index()


def merge_all(inv15, weather15, meta):
    """Inverter rows + weather (on time) + metadata (on inverter_id)."""
    meta = meta.drop(columns=["file_name", "brick_inverter"])
    text_cols = meta.select_dtypes(include=["object", "string"]).columns.drop("inverter_id")
    meta[text_cols] = meta[text_cols].astype("category")
    meta[meta.select_dtypes("float64").columns] = (
        meta.select_dtypes("float64").astype("float32"))

    inv15 = inv15.copy()
    inv15["inverter_id"] = inv15["inverter_id"].astype(str)
    merged = (inv15.merge(weather15, on="time", how="left", validate="many_to_one")
                   .merge(meta, on="inverter_id", how="left", validate="many_to_one"))
    merged["inverter_id"] = merged["inverter_id"].astype("category")
    # Bins with no weather rows at all have 0 valid readings, not NaN.
    merged["n_valid_1min"] = merged["n_valid_1min"].fillna(0).astype("int8")
    return merged


def build_full_grid(inv15, freq=FREQ):
    """Reindex each inverter to every 06:00-17:45 bin from its first bin to
    the end of 2023. Added rows are NaN with n_valid_5min = 0."""
    grids = []
    for inverter_id, first in inv15.groupby("inverter_id", observed=True)["time"].min().items():
        times = pd.date_range(first, DATA_END, freq=freq)
        times = times[(times.strftime("%H:%M") >= LOGGER_FIRST_BIN)
                      & (times.strftime("%H:%M") <= LOGGER_LAST_BIN)]
        grids.append(pd.DataFrame({"inverter_id": inverter_id, "time": times}))
    grid = pd.concat(grids, ignore_index=True)

    inv15 = inv15.copy()
    inv15["inverter_id"] = inv15["inverter_id"].astype(str)
    full = grid.merge(inv15, on=["inverter_id", "time"], how="left", validate="one_to_one")
    full["n_valid_5min"] = full["n_valid_5min"].fillna(0).astype("int8")
    return full


def gap_tables(df):
    """Missing = no valid power reading in the bin. Only daylight bins
    (irradiance >= DAYLIGHT_WM2) count; bins without weather are excluded."""
    daylight = df["irradiance_wm2"] >= DAYLIGHT_WM2
    d = pd.DataFrame({
        "inverter_id": df["inverter_id"].astype(str),
        "month": df["time"].dt.to_period("M").astype(str),
        "missing": df["n_valid_5min"] == 0,
    })[daylight]

    by_month = d.groupby(["inverter_id", "month"]).agg(
        daylight_bins=("missing", "size"), missing_bins=("missing", "sum"))
    by_month["missing_pct"] = (100 * by_month["missing_bins"] / by_month["daylight_bins"]).round(2)

    coverage = d.groupby("inverter_id").agg(
        daylight_bins=("missing", "size"), missing_bins=("missing", "sum"))
    coverage["first_time"] = df.groupby(df["inverter_id"].astype(str))["time"].min()
    coverage["coverage"] = (1 - coverage["missing_bins"] / coverage["daylight_bins"]).round(4)
    coverage["kept"] = coverage["coverage"] >= MIN_COVERAGE
    return by_month.reset_index(), coverage.sort_values("coverage").reset_index()


def summary_row(step, df):
    return {"step": step, "rows": len(df), "columns": df.shape[1],
            "inverters": df["inverter_id"].nunique()}


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("Step 1: Brick metadata")
    graph = load_brick_graph()
    all_meta = build_inverter_metadata(graph)
    print(f"Brick model: {all_meta['station_id'].nunique()} stations, "
          f"{len(all_meta)} inverters")

    meta = match_inverter_files(all_meta)
    print(f"Matched to inverter CSVs: {len(meta)} inverters at "
          f"{meta['station_id'].nunique()} stations")

    out = PROCESSED / "inverter_metadata.parquet"
    meta.to_parquet(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}")

    print("Step 2a: inverter CSVs -> Parquet")
    inv = convert_inverters()
    out = PROCESSED / "inverter_5min.parquet"
    inv.to_parquet(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}: {len(inv):,} rows x {inv.shape[1]} columns")

    print("Step 2b: weather CSVs -> Parquet")
    weather = convert_weather()
    out = PROCESSED / "weather_1min.parquet"
    weather.to_parquet(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}: {len(weather):,} rows x {weather.shape[1]} columns")

    summary = [summary_row("raw 5-min inverter data", inv)]

    print(f"Step 3: resample to {FREQ}")
    inv15 = resample_inverters(inv)
    summary.append(summary_row(f"resampled to {FREQ}", inv15))
    print(f"  inverters: {len(inv):,} -> {len(inv15):,} rows")
    weather15 = resample_weather(weather)
    print(f"  weather:   {len(weather):,} -> {len(weather15):,} rows")

    print("Step 4: full grid, merge, gap analysis, coverage filter")
    full = build_full_grid(inv15)
    summary.append(summary_row(f"full {LOGGER_FIRST_BIN}-{LOGGER_LAST_BIN} grid", full))
    print(f"  full grid: {len(inv15):,} -> {len(full):,} rows "
          f"({len(full) - len(inv15):,} empty bins added)")
    merged = merge_all(full, weather15, meta)
    summary.append(summary_row("merged with weather + metadata", merged))

    by_month, coverage = gap_tables(merged)
    by_month.to_csv(TABLES / "gaps_by_inverter_month.csv", index=False)
    coverage.to_csv(TABLES / "coverage_by_inverter.csv", index=False)
    dropped = coverage[~coverage["kept"]]
    print(f"  daylight coverage (irradiance >= {DAYLIGHT_WM2} W/m2): "
          f"min {coverage['coverage'].min():.1%}, median {coverage['coverage'].median():.1%}")
    print(f"  dropped (< {MIN_COVERAGE:.0%}): {len(dropped)} inverters")
    for _, r in dropped.iterrows():
        print(f"    {r['inverter_id']}: coverage {r['coverage']:.1%}")

    final = merged[merged["inverter_id"].isin(coverage.loc[coverage["kept"], "inverter_id"])]
    final = final.reset_index(drop=True)
    final["inverter_id"] = final["inverter_id"].cat.remove_unused_categories()
    summary.append(summary_row(f"coverage filter (>= {MIN_COVERAGE:.0%})", final))

    out = PROCESSED / MERGED_NAME
    final.to_parquet(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}: {len(final):,} rows x {final.shape[1]} columns")

    summary = pd.DataFrame(summary)
    summary.to_csv(TABLES / "prep_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

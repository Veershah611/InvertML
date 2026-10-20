"""A2 preprocessing using B1 targets and B2's shared chronological splits.

Run: python -m src.preprocess
Outputs: data/processed/a2/ and reports/tables/a2_preprocessing_*.csv.
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import (
    CONFIG, DEFAULT_UNSEEN_STATIONS, evaluate_classification,
    evaluate_regression, split_chronological, split_unseen_stations,
)
from src.features import B1_CORE_FEATURES, add_time_features, build_b1_features
from src.targets import (
    MIN_IRRADIANCE_DAYLIGHT, assign_operational_status, compute_health_score,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = pd.Timedelta("1h")
BASE_NUMERIC_FEATURES = [
    "ac_power_w", "irradiance_wm2", "temp_c", "wind_speed_ms", "rh_pct",
    "capacity_kwp", "tilt_deg", "altitude_m", "n_valid_5min",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + B1_CORE_FEATURES
CATEGORICAL_FEATURES = ["azimuth"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
WEATHER_BOUNDS = {
    "temp_c": (-50, 60), "rh_pct": (0, 100), "wind_speed_ms": (0, 100),
    "wind_dir_deg": (0, 360), "rainfall_mm": (0, np.inf),
    "visibility_km": (0, np.inf),
}


def clean_measurements(df, daylight_threshold=50.0):
    """Apply fixed physical rules; retain missing telemetry and audit flags.

    No learned statistics, interpolation, scaling, or label imputation occurs
    here. Input is A1's complete logger grid, before any daylight filtering.
    Capacity is DC array kWp; 1.1 * capacity is a screening proxy, not a measured
    inverter AC rating. Missing power is an offline/telemetry event, not proof
    of a physical fault.
    """
    required = {"time", "inverter_id", "ac_power_w", "irradiance_wm2", "capacity_kwp"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not np.isfinite(daylight_threshold) or daylight_threshold <= 0:
        raise ValueError("daylight_threshold must be positive and finite")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="raise")
    if out[["time", "inverter_id"]].isna().any().any():
        raise ValueError("Time and inverter ID must not be missing")
    if out.duplicated(["inverter_id", "time"]).any():
        raise ValueError("Duplicate inverter/time rows; resolve them in A1 preparation")
    out = out.sort_values(["inverter_id", "time"]).reset_index(drop=True)
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    # Missingness refers to the original observation, before rejecting errors.
    out["is_offline"] = out["ac_power_w"].isna()
    if "n_valid_5min" in out:
        out["is_offline"] |= out["n_valid_5min"].eq(0)
    out["invalid_capacity"] = out["capacity_kwp"].isna() | out["capacity_kwp"].le(0)
    out["invalid_power"] = (
        out["ac_power_w"].lt(0)
        | out["ac_power_w"].gt(1.1 * out["capacity_kwp"] * 1000)
    )
    out["invalid_irradiance"] = (
        out["irradiance_wm2"].lt(0) | out["irradiance_wm2"].gt(1500)
    )
    out.loc[out["invalid_power"] | out["is_offline"], "ac_power_w"] = np.nan
    out.loc[out["invalid_capacity"], "capacity_kwp"] = np.nan
    out.loc[out["invalid_irradiance"], "irradiance_wm2"] = np.nan
    out["invalid_weather_count"] = np.int8(0)
    for col, (low, high) in WEATHER_BOUNDS.items():
        if col in out:
            bad = out[col].notna() & ~out[col].between(low, high)
            out["invalid_weather_count"] += bad.astype("int8")
            out.loc[bad, col] = np.nan
    out["is_daylight"] = out["irradiance_wm2"].ge(daylight_threshold)
    # B1 currently fills missing power with zero. Restore undefined labels
    # locally so its physical formula is only used on observed valid data.
    out = compute_health_score(out, filter_daylight=False)
    observed = (
        out[["ac_power_w", "irradiance_wm2", "capacity_kwp"]].notna().all(axis=1)
        & ~out["is_offline"]
    )
    out.loc[~observed, ["health_score", "expected_power_w"]] = np.nan
    out["status"] = pd.Series(pd.NA, index=out.index, dtype="string")
    valid = out["health_score"].notna()
    out.loc[valid, "status"] = assign_operational_status(out.loc[valid, "health_score"])
    out = add_time_features(out)
    # Metadata gaps are handled by training-fitted imputers, not fixed guesses.
    for col in BASE_NUMERIC_FEATURES:
        if col not in out:
            out[col] = np.nan
    if "azimuth" not in out:
        out["azimuth"] = np.nan
    return out


def make_b2_splits(df, held_out_stations=None):
    """Use B2's split functions, excluding held-out inverter IDs from validation.

    B2's default holdout is defined at inverter level, not whole station level.
    `test` retains B2's full 2023 test; `test_unseen` is a subset of it.
    """
    held = (
        CONFIG.get("unseen_stations", DEFAULT_UNSEEN_STATIONS)
        if held_out_stations is None else held_out_stations
    )
    _, val, test = split_chronological(df)
    train, unseen = split_unseen_stations(df, held_out_stations=held)
    normalize = lambda s: s.lower().replace("-", "_").replace(" ", "_")
    held_clean = {normalize(s) for s in held}
    val = val[~val["inverter_id"].astype(str).map(normalize).isin(held_clean)]
    return {
        name: part.sort_values(["time", "inverter_id"]).reset_index(drop=True)
        for name, part in {
            "train": train, "val": val, "test": test, "test_unseen": unseen,
        }.items()
    }


def forecasting_rows(part):
    """Join B1 Health(t+1h) by exact timestamp within a B2 split.

    A row shift on daylight-filtered data can jump gaps or nights. Joining
    inside each partition also prevents labels crossing split boundaries.
    Offline rows remain in the cleaned event table but cannot supply labels.
    """
    future = part[["inverter_id", "time", "health_score"]].rename(
        columns={"health_score": "health_score_lead4"}
    )
    future = future.copy()
    future["target_time"] = future["time"]
    future["time"] = future["time"] - HORIZON
    out = part.merge(future, on=["inverter_id", "time"], how="left", validate="one_to_one")
    out = out.loc[out["health_score"].notna() & out["health_score_lead4"].notna()].copy()
    out["status_lead4"] = assign_operational_status(out["health_score_lead4"])
    return out.sort_values(["time", "inverter_id"]).reset_index(drop=True)


def engineer_b2_partitions(partitions):
    """B1 features after B2 splitting, with past-only context across boundaries.

    Held-out telemetry never contributes training or validation peer statistics.
    The test peer pool contains all test inverters, as available at inference.
    Unseen-test features are sliced from the test, not recomputed on fewer peers.
    Only preceding 30-day observations carry into the next partition; labels
    are still joined independently inside each partition.
    """
    engineered = {}
    history = None
    for name in ("train", "val", "test"):
        current = partitions[name]
        context = current.copy()
        context["_a2_current"] = True
        if history is not None and not current.empty:
            start = current["time"].min()
            prior = history[
                history["time"].ge(start - pd.Timedelta("30D"))
                & history["time"].lt(start)
            ].copy()
            prior["_a2_current"] = False
            context = pd.concat([prior, context], ignore_index=True)
        enriched = build_b1_features(context)
        engineered[name] = enriched.loc[enriched["_a2_current"]].drop(
            columns="_a2_current"
        ).sort_values(["time", "inverter_id"]).reset_index(drop=True)
        # Store only the latest 30 days of observed data, not a fitted statistic.
        if not context.empty:
            history = context[
                context["time"].ge(context["time"].max() - pd.Timedelta("30D"))
            ].drop(columns="_a2_current")
    keys = partitions["test_unseen"][["inverter_id", "time"]]
    engineered["test_unseen"] = engineered["test"].merge(
        keys, on=["inverter_id", "time"], how="inner", validate="one_to_one"
    )
    return engineered


def feature_frame(df):
    """Explicit input allowlist excludes labels, future timestamps and IDs."""
    out = df[FEATURES].copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].astype("float64")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    return out


def build_preprocessor(strategy="median", scale=False):
    """Unfitted transformer: place inside the model Pipeline for every CV fold.

    Imputation affects model inputs only, never the stored measurements or
    targets. Both mean and median are learned from training rows. No bidirectional
    interpolation is used because its right endpoint can be in the future.
    """
    if strategy not in {"mean", "median"}:
        raise ValueError("strategy must be mean or median")
    numeric_steps = [("imputer", SimpleImputer(
        strategy=strategy, add_indicator=True, keep_empty_features=True,
    ))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def time_series_folds(df, n_splits=3):
    """B2-style expanding TimeSeriesSplit over unique times with label purging.

    All inverters at one timestamp stay in the same fold. Requires globally
    chronological rows, not rows ordered by inverter first.
    """
    times = pd.to_datetime(df["time"])
    if not times.is_monotonic_increasing:
        raise ValueError("Sort rows by time before time-series cross-validation")
    unique_times = pd.Index(times.unique())
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(unique_times):
        train_mask = times.isin(unique_times[tr])
        if "target_time" in df:
            train_mask &= df["target_time"] < unique_times[va[0]]
        train_idx = np.flatnonzero(train_mask)
        if not len(train_idx):
            raise ValueError("No training rows remain after forecast-horizon purge")
        yield train_idx, np.flatnonzero(times.isin(unique_times[va]))


def compare_imputers(splits, max_train=30000, max_val=15000):
    """Small fixed RF probes compare input imputation on validation only.

    This is a preprocessing study, not A2's tuned Random Forest deliverable.
    Identical seeded samples and model settings are used for both strategies.
    """
    seed = CONFIG.get("random_state", 42)
    sample = lambda df, n: df.sample(n=min(n, len(df)), random_state=seed).sort_values(
        ["time", "inverter_id"]
    )
    train, val = sample(splits["train"], max_train), sample(splits["val"], max_val)
    if train.empty or val.empty:
        raise ValueError("Training and validation both need valid forecast labels")
    X_train, X_val = feature_frame(train), feature_frame(val)
    rows = []
    for strategy in ("median", "mean"):
        kwargs = dict(n_estimators=40, max_depth=12, min_samples_leaf=10,
                      random_state=seed, n_jobs=2)
        reg = Pipeline([("preprocess", build_preprocessor(strategy)),
                        ("model", RandomForestRegressor(**kwargs))])
        cls = Pipeline([("preprocess", build_preprocessor(strategy)),
                        ("model", RandomForestClassifier(class_weight="balanced", **kwargs))])
        reg.fit(X_train, train["health_score_lead4"])
        cls.fit(X_train, train["status_lead4"])
        rm = evaluate_regression(val["health_score_lead4"], reg.predict(X_val))
        cm = evaluate_classification(val["status_lead4"], cls.predict(X_val))
        rows.append({
            "strategy": strategy, "train_rows": len(train), "val_rows": len(val),
            "val_rmse": rm["rmse"], "val_mae": rm["mae"],
            "val_macro_f1": cm["macro_f1"], "val_faulty_recall": cm["faulty_recall"],
            "train_missing_cells": int(X_train.isna().sum().sum()),
            "val_missing_cells": int(X_val.isna().sum().sum()),
        })
    return pd.DataFrame(rows)


def run(input_path, output_dir, compare=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    settings = CONFIG.get("preprocessing", {})
    threshold = settings.get("daylight_irradiance_wm2", 50.0)
    raw = pd.read_parquet(input_path)
    cleaned = clean_measurements(raw, daylight_threshold=threshold)
    flag_cols = ["is_offline", "invalid_capacity", "invalid_power",
                 "invalid_irradiance", "invalid_weather_count"]
    audit = [{"stage": "input", "rows": len(raw), "columns": len(raw.columns)}]
    audit += [{"stage": col, "rows": int(cleaned[col].astype(bool).sum()),
               "columns": len(cleaned.columns)} for col in flag_cols]
    # Full grid preserves gaps and rejected measurements for A1 outage work.
    cleaned.to_parquet(output_dir / "cleaned_grid.parquet", index=False)
    eligible = cleaned[
        cleaned["is_daylight"] & ~cleaned["invalid_power"] & ~cleaned["invalid_capacity"]
    ].copy()
    audit.append({"stage": "eligible_daylight", "rows": len(eligible),
                  "columns": len(eligible.columns)})
    eligible.to_parquet(output_dir / "daylight.parquet", index=False)
    partitions = engineer_b2_partitions(make_b2_splits(eligible))
    splits = {name: forecasting_rows(part) for name, part in partitions.items()}
    summary = []
    for name, part in splits.items():
        part.to_parquet(output_dir / f"{name}.parquet", index=False)
        summary.append({
            "split": name, "rows": len(part), "inverters": part["inverter_id"].nunique(),
            "first_time": part["time"].min(), "last_time": part["time"].max(),
            "last_target_time": part["target_time"].max(),
            **{label: int(part["status_lead4"].eq(label).sum())
               for label in ["Healthy", "Warning", "Faulty"]},
        })
    pd.DataFrame(audit).to_csv(tables / "a2_preprocessing_audit.csv", index=False)
    pd.DataFrame(summary).to_csv(tables / "a2_preprocessing_splits.csv", index=False)
    fold_summary = []
    train = splits["train"]
    for number, (tr, va) in enumerate(time_series_folds(train), 1):
        fold_summary.append({
            "fold": number, "train_rows": len(tr), "val_rows": len(va),
            "train_last_time": train.iloc[tr]["time"].max(),
            "train_last_target_time": train.iloc[tr]["target_time"].max(),
            "val_first_time": train.iloc[va]["time"].min(),
            "val_last_time": train.iloc[va]["time"].max(),
        })
    pd.DataFrame(fold_summary).to_csv(tables / "a2_preprocessing_cv.csv", index=False)
    strategy = "median"
    if compare:
        comparison = compare_imputers(splits)
        comparison.to_csv(tables / "a2_preprocessing_imputation.csv", index=False)
        # Macro-F1 is primary; RMSE breaks ties; median wins exact ties.
        strategy = comparison.sort_values(
            ["val_macro_f1", "val_rmse"], ascending=[False, True], kind="stable"
        ).iloc[0]["strategy"]
        print(comparison.to_string(index=False), flush=True)
    for scale, name in [(False, "tree"), (True, "scaled")]:
        transformer = build_preprocessor(strategy=strategy, scale=scale)
        transformer.fit(feature_frame(train))
        joblib.dump(transformer, output_dir / f"preprocessor_{name}.joblib")
    manifest = {
        "input": str(input_path), "rows_input": len(raw),
        "daylight_threshold": threshold,
        "b1_label_daylight_threshold": MIN_IRRADIANCE_DAYLIGHT,
        "regression_target": "health_score_lead4",
        "classification_target": "status_lead4", "horizon_minutes": 60,
        "selected_imputation": strategy, "imputation_comparison_run": compare,
        "features": FEATURES, "splitting": "src.evaluate B2 functions",
        "b1_features": B1_CORE_FEATURES,
        "self_history": "Observed preceding 30 calendar days; same irradiance bin",
        "peer_population": "Known inverters in train/val; all test inverters at inference",
        "holdout_unit": "inverter_id", "test_includes_unseen": True,
        "cv": "3-fold TimeSeriesSplit on unique timestamps; target horizon purged",
        "saved_transformers": "Fit on full train only; clone/refit inside each CV fold",
        "missing_power_labels": "undefined, not zero",
        "interpolation": "No future-dependent interpolation; model-input mean/median only",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Preprocessing outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / CONFIG["paths"]["processed"] / "hkust_15min_v1.parquet")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / CONFIG["paths"]["processed"] / "a2")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output_dir, compare=not args.skip_comparison)

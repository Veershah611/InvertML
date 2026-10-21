"""Pair A HKUST synthetic-fault Isolation Forest benchmark.

Run after A2 preprocessing: python -m src.models.iforest_a2
Uses B2's shared fault injector and evaluator. Synthetic faults are not labels
from maintenance records and must not be described as real fault validation.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import inject_synthetic_faults, evaluate_fault_detection
from src.preprocess import ROOT

DATA = ROOT / "data" / "processed" / "a2"
TABLES = ROOT / "reports" / "tables"
SEED = 42
FEATURES = ["irradiance_wm2", "temp_c", "specific_yield"]


def anomaly_inputs(df):
    out = df[["irradiance_wm2", "temp_c"]].copy()
    out["specific_yield"] = df["ac_power_w"] / (df["capacity_kwp"] * 1000)
    return out[FEATURES]


def run():
    columns = ["inverter_id", "time", "ac_power_w", "capacity_kwp",
               "irradiance_wm2", "temp_c", "health_score"]
    train = pd.read_parquet(DATA / "train.parquet", columns=columns)
    test = pd.read_parquet(DATA / "test.parquet", columns=columns)
    # Normal training selection uses the engineered current score as a proxy;
    # it is never supplied to the unsupervised model as a feature.
    normal_train = train[train.health_score.ge(90)]
    normal_train = normal_train.sample(n=min(30000, len(normal_train)), random_state=SEED)
    normal_test = test[test.health_score.ge(90)].copy()
    start = pd.Timestamp("2023-01-01")
    first_window = normal_test[normal_test.time.between(
        start + pd.Timedelta("30D"), start + pd.Timedelta("33D")
    )].groupby("inverter_id").size().sort_values(ascending=False)
    second_window = normal_test[normal_test.time.between(
        start + pd.Timedelta("60D"), start + pd.Timedelta("62D")
    )].groupby("inverter_id").size().sort_values(ascending=False)
    first = first_window.index[0]
    second = next(inv for inv in second_window.index if inv != first)
    # Order the two chosen inverters so B2's injector modifies both intended
    # windows. Keep additional inverters as untouched controls.
    controls = normal_test.inverter_id.value_counts().index[:8].tolist()
    controls = [inv for inv in controls if inv not in {first, second}]
    order = [first, second] + controls
    normal_test = normal_test[normal_test.inverter_id.isin(order)]
    normal_test = pd.concat(
        [normal_test[normal_test.inverter_id.eq(inv)].sort_values("time") for inv in order],
        ignore_index=True,
    )
    # B2 places faults 30 and 60 days after the minimum test timestamp.
    # Preserve only the first half year to make the test set concrete.
    normal_test = normal_test[normal_test.time.lt("2023-07-01")].reset_index(drop=True)
    injected = inject_synthetic_faults(normal_test)
    if injected.is_injected_fault.sum() == 0:
        raise ValueError("B2 injection produced no fault rows")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(n_estimators=200, contamination=.05,
                                     max_samples=256, random_state=SEED, n_jobs=2)),
    ])
    model.fit(anomaly_inputs(normal_train))
    raw_predictions = model.predict(anomaly_inputs(injected))
    metrics = evaluate_fault_detection(injected.is_injected_fault, raw_predictions)
    summary = {
        "train_normal_rows": len(normal_train),
        "test_rows": len(injected),
        "injected_rows": int(injected.is_injected_fault.sum()),
        "inverter_drop": first, "inverter_outage": second,
        "contamination": .05,
        **metrics,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(TABLES / "a2_iforest_injected_faults.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()

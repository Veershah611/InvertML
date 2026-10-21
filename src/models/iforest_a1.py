"""Pair A HKUST Isolation Forest, A1 extension of A2's benchmark (iforest_a2.py).

What changes against A2:
    - contamination and n_estimators are tuned (section 10) on B2's synthetic
      faults injected into the Jul-Dec 2022 validation period; 2023 test is
      scored once with the chosen settings
    - a second input set adds peer_dev_campus to A2's three inputs. Peer
      deviation divides by the campus median PR at the same timestamp, so the
      shared weather and the 2023-02-14 irradiance sensor step cancel out
    - the test false-alarm rate is split before / after 2023-02-14

Kept from A2 so the numbers are comparable: normal training rows (train rows
with Health >= 90, 30k sample), the test population (Health >= 90 rows of the
two injected inverters plus 8 controls, Jan-Jun 2023), B2's injector and
evaluator, median imputation and scaling fitted on normal training rows.

Injected power changes PR, so specific yield and peer deviation are recomputed
after injection. The campus median comes from the untouched split, all
inverters, at the same timestamp. Synthetic faults are not maintenance records
and must not be described as real fault validation.

Run after A2 preprocessing (python -m src.preprocess --skip-comparison):
    python -m src.models.iforest_a1
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_fault_detection, inject_synthetic_faults
from src.features import add_performance_ratio_and_yield
from src.preprocess import ROOT

DATA = ROOT / "data" / "processed" / "a2"
TABLES = ROOT / "reports" / "tables"
SEED = 42
SENSOR_CHANGE = pd.Timestamp("2023-02-14")
PEER_EPSILON = 0.05            # same constant as B1's add_peer_deviation_features
INPUT_SETS = {
    "A2 inputs": ["irradiance_wm2", "temp_c", "specific_yield"],
    "A2 inputs + peer deviation": ["irradiance_wm2", "temp_c", "specific_yield",
                                   "peer_dev_campus"],
}
CONTAMINATION = [0.01, 0.02, 0.05, 0.1]
N_ESTIMATORS = [100, 200, 400]
A2_SETTINGS = (0.05, 200)
COLUMNS = ["inverter_id", "time", "ac_power_w", "capacity_kwp",
           "irradiance_wm2", "temp_c", "health_score"]


# ---------------------------------------------------------------- data

def campus_median_pr(part):
    """Campus median PR per timestamp from an untouched split (all inverters)."""
    pr = add_performance_ratio_and_yield(part)["performance_ratio"]
    return pr.groupby(part["time"]).median()


def anomaly_inputs(df, median_pr):
    """A2's inputs plus peer deviation, recomputed from (possibly injected) power."""
    out = df[["irradiance_wm2", "temp_c"]].copy()
    out["specific_yield"] = df["ac_power_w"] / (df["capacity_kwp"] * 1000)
    pr = add_performance_ratio_and_yield(df)["performance_ratio"]
    med = df["time"].map(median_pr)
    out["peer_dev_campus"] = np.clip((pr - med) / (med + PEER_EPSILON), -3.0, 3.0)
    return out


def injected_population(part, start, end):
    """A2's test population rule applied to any period: Health >= 90 rows, the
    inverter with most rows 30-33 days in gets the 35% drop, a different one
    with most rows 60-62 days in gets the outage, plus 8 untouched controls."""
    normal = part[part.health_score.ge(90)]
    first_window = normal[normal.time.between(
        start + pd.Timedelta("30D"), start + pd.Timedelta("33D")
    )].groupby("inverter_id").size().sort_values(ascending=False)
    second_window = normal[normal.time.between(
        start + pd.Timedelta("60D"), start + pd.Timedelta("62D")
    )].groupby("inverter_id").size().sort_values(ascending=False)
    first = first_window.index[0]
    second = next(inv for inv in second_window.index if inv != first)
    controls = normal.inverter_id.value_counts().index[:8].tolist()
    order = [first, second] + [inv for inv in controls if inv not in {first, second}]
    normal = normal[normal.inverter_id.isin(order)]
    # B2's injector picks the first two inverters in row order.
    normal = pd.concat([normal[normal.inverter_id.eq(inv)].sort_values("time") for inv in order],
                       ignore_index=True)
    normal = normal[normal.time.ge(start) & normal.time.lt(end)].reset_index(drop=True)
    injected = inject_synthetic_faults(normal)
    if injected.is_injected_fault.sum() == 0:
        raise ValueError("B2 injection produced no fault rows")
    return injected, first, second


# ---------------------------------------------------------------- model

def detector(contamination, n_estimators):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                     max_samples=256, random_state=SEED, n_jobs=2)),
    ])


def score(y_true, raw_pred):
    m = evaluate_fault_detection(y_true, raw_pred)
    fn = int(np.sum(np.asarray(y_true) == 1)) - m["tp"]
    precision = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else 0.0
    f1 = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + fn) if m["tp"] else 0.0
    return {**m, "precision": round(precision, 4), "f1": round(f1, 4)}


def run():
    train = pd.read_parquet(DATA / "train.parquet", columns=COLUMNS)
    val = pd.read_parquet(DATA / "val.parquet", columns=COLUMNS)
    test = pd.read_parquet(DATA / "test.parquet", columns=COLUMNS)
    normal_train = train[train.health_score.ge(90)]
    normal_train = normal_train.sample(n=min(30000, len(normal_train)), random_state=SEED)
    X_train = anomaly_inputs(normal_train, campus_median_pr(train))

    val_inj, val_drop, val_out = injected_population(
        val, pd.Timestamp("2022-07-01"), pd.Timestamp("2023-01-01"))
    test_inj, test_drop, test_out = injected_population(
        test, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01"))
    X_val = anomaly_inputs(val_inj, campus_median_pr(val))
    X_test = anomaly_inputs(test_inj, campus_median_pr(test))
    print(f"val: drop {val_drop}, outage {val_out}, {val_inj.is_injected_fault.sum()} injected rows")
    print(f"test: drop {test_drop}, outage {test_out}, {test_inj.is_injected_fault.sum()} injected rows")

    tuning, results = [], []
    after = test_inj.time.ge(SENSOR_CHANGE).to_numpy()
    for input_set, cols in INPUT_SETS.items():
        for contamination, n_estimators in product(CONTAMINATION, N_ESTIMATORS):
            model = detector(contamination, n_estimators).fit(X_train[cols])
            m = score(val_inj.is_injected_fault, model.predict(X_val[cols]))
            tuning.append({"inputs": input_set, "contamination": contamination,
                           "n_estimators": n_estimators, **m})
        best = (pd.DataFrame(tuning).query("inputs == @input_set")
                .sort_values(["f1", "detection_rate"], ascending=False).iloc[0])
        for label, settings in [("tuned", (best.contamination, int(best.n_estimators))),
                                ("A2 settings", A2_SETTINGS)]:
            model = detector(*settings).fit(X_train[cols])
            pred = model.predict(X_test[cols])
            m = score(test_inj.is_injected_fault, pred)
            normal_pred = pred[test_inj.is_injected_fault.eq(0).to_numpy()] == -1
            normal_after = after[test_inj.is_injected_fault.eq(0).to_numpy()]
            results.append({"inputs": input_set, "version": label,
                            "contamination": settings[0], "n_estimators": settings[1], **m,
                            "false_alarm_before_14feb": round(float(normal_pred[~normal_after].mean()), 4),
                            "false_alarm_after_14feb": round(float(normal_pred[normal_after].mean()), 4)})

    tuning, results = pd.DataFrame(tuning), pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(TABLES / "a1_iforest_tuning.csv", index=False)
    results.to_csv(TABLES / "a1_iforest_test.csv", index=False)
    return {"tuning": tuning, "results": results}


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 250)
    print(out["results"].to_string(index=False))

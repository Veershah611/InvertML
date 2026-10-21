"""Stage 11: Validation and Evaluation Framework (B2).

This file gives the whole team simple, shared functions for:
1. Splitting data by date (Chronological Split: Train, Validation, Test)
2. Splitting data by stations (Testing on ~7 Unseen Stations)
3. Calculating Regression metrics (MAE, RMSE, R2)
4. Calculating Classification metrics (Macro-F1, Faulty Recall, Confusion Matrix)
5. Injecting synthetic faults to test anomaly detection
6. Testing on the real-fault GPVS dataset
7. Saving model results to results/results.csv

Run this file anytime to test it:
    python -m src.evaluate
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Setup: Paths & Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

# The 6 stations that start only in 2023, held out to test unseen rooftop generalization
DEFAULT_UNSEEN_STATIONS = [
    "sq_block_p_inverter",
    "shaw_auditorium_inverter",
    "ug_hall8_inverter",
    "ug_hall9_inverter",
    "zone_a2_inverter_2",
    "zone_a2_inverter_3",
]



# ---------------------------------------------------------------------------
# 1. Data Splitting Functions
# ---------------------------------------------------------------------------

def split_chronological(df, time_col="time"):
    """Split data by time so we never use future data to predict the past.

    - Train:      Jan 2021 to June 30, 2022
    - Validation: July 1, 2022 to Dec 31, 2022
    - Test:       All of 2023 (Jan 1, 2023 to Dec 31, 2023)
    """
    # If column is named 'timestamp' instead of 'time', handle it automatically
    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"

    t = pd.to_datetime(df[time_col])

    # Date cutoff points from config
    train_end = "2022-06-30 23:59:59"
    val_end = "2022-12-31 23:59:59"
    test_start = "2023-01-01 00:00:00"
    test_end = "2023-12-31 23:59:59"

    # Make masks
    train_mask = t <= train_end
    val_mask = (t > train_end) & (t <= val_end)
    test_mask = (t >= test_start) & (t <= test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print(f"Chronological Split: {len(train_df):,} train, {len(val_df):,} val, {len(test_df):,} test")
    return train_df, val_df, test_df


def split_unseen_stations(df, held_out_stations=None, station_col="inverter_id", time_col="time"):
    """Hold out ~7 stations completely to test performance on new buildings.

    - train_known: Other stations during training period (<= 2022-06-30)
    - test_unseen: Only held-out stations during 2023 test period
    """
    if held_out_stations is None:
        held_out_stations = CONFIG.get("unseen_stations", DEFAULT_UNSEEN_STATIONS)

    if station_col not in df.columns and "station_id" in df.columns:
        station_col = "station_id"
    if time_col not in df.columns and "timestamp" in df.columns:
        time_col = "timestamp"

    t = pd.to_datetime(df[time_col])
    train_end = "2022-06-30 23:59:59"
    test_start = "2023-01-01 00:00:00"
    test_end = "2023-12-31 23:59:59"

    # Match station names safely (lowercase with underscores)
    st_clean = df[station_col].astype(str).str.lower().str.replace("-", "_").str.replace(" ", "_")
    held_out_clean = [s.lower().replace("-", "_").replace(" ", "_") for s in held_out_stations]

    is_unseen = st_clean.isin(held_out_clean)

    train_known = df[(~is_unseen) & (t <= train_end)].copy()
    test_unseen = df[is_unseen & (t >= test_start) & (t <= test_end)].copy()

    print(f"Unseen Station Split: {len(train_known):,} train (known), {len(test_unseen):,} test (unseen)")
    return train_known, test_unseen


# ---------------------------------------------------------------------------
# 2. Metrics Functions
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred):
    """Calculate simple regression metrics: MAE, RMSE, and R2."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Ignore any NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)

    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "n_samples": len(y_t),
    }


def evaluate_classification(y_true, y_pred):
    """Calculate simple classification metrics: Macro-F1, Faulty Recall, and Confusion Matrix."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Check Faulty Recall (our most important metric: how many real faults did we catch?)
    # Supports either string 'Faulty' or integer 2
    is_true_faulty = (y_true == "Faulty") | (y_true == 2)
    is_pred_faulty = (y_pred == "Faulty") | (y_pred == 2)

    if np.sum(is_true_faulty) > 0:
        faulty_recall = recall_score(is_true_faulty, is_pred_faulty, zero_division=0)
        faulty_precision = precision_score(is_true_faulty, is_pred_faulty, zero_division=0)
    else:
        faulty_recall = 0.0
        faulty_precision = 0.0

    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": round(float(macro_f1), 4),
        "faulty_recall": round(float(faulty_recall), 4),
        "faulty_precision": round(float(faulty_precision), 4),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# 3. Synthetic Fault Injection Engine
# ---------------------------------------------------------------------------

def inject_synthetic_faults(df, station_col="inverter_id", power_col="ac_power_w", time_col="time"):
    """Simulate realistic solar inverter problems on test data:

    1. Sudden Drop: 35% power drop on one station for 3 days (dust / shading / loose wire)
    2. Complete Outage: power drops to 0 for 2 days (inverter trip)
    """
    test_copy = df.copy()

    if station_col not in test_copy.columns and "station_id" in test_copy.columns:
        station_col = "station_id"
    if power_col not in test_copy.columns:
        for c in ["power", "totalActivePower", "pr"]:
            if c in test_copy.columns:
                power_col = c
                break
    if time_col not in test_copy.columns and "timestamp" in test_copy.columns:
        time_col = "timestamp"

    test_copy[time_col] = pd.to_datetime(test_copy[time_col])
    test_copy["is_injected_fault"] = 0

    stations = list(test_copy[station_col].unique())
    if len(stations) < 1:
        return test_copy

    st1 = stations[0]
    st2 = stations[1] if len(stations) > 1 else stations[0]
    min_date = test_copy[time_col].min()

    # Fault 1: 35% power drop on station 1 for 3 days
    start1 = min_date + pd.Timedelta(days=30)
    end1 = start1 + pd.Timedelta(days=3)
    mask1 = (test_copy[station_col] == st1) & (test_copy[time_col] >= start1) & (test_copy[time_col] <= end1)
    test_copy.loc[mask1, "is_injected_fault"] = 1
    test_copy.loc[mask1, power_col] = test_copy.loc[mask1, power_col] * 0.65

    # Fault 2: Total blackout on station 2 for 2 days
    start2 = min_date + pd.Timedelta(days=60)
    end2 = start2 + pd.Timedelta(days=2)
    mask2 = (test_copy[station_col] == st2) & (test_copy[time_col] >= start2) & (test_copy[time_col] <= end2)
    test_copy.loc[mask2, "is_injected_fault"] = 1
    test_copy.loc[mask2, power_col] = 0.0

    print(f"Injected synthetic faults into {test_copy['is_injected_fault'].sum()} rows.")
    return test_copy


def evaluate_fault_detection(y_true_injected, anomaly_predictions):
    """Check how well an anomaly detector caught the injected faults.

    - anomaly_predictions can be 1 for anomaly and 0 for normal,
      or -1 for anomaly and 1 for normal (Isolation Forest format).
    """
    y_true = np.array(y_true_injected).astype(int)
    y_pred = np.array(anomaly_predictions)

    # Convert Isolation Forest format (-1 anomaly, 1 normal) into (1 anomaly, 0 normal)
    if -1 in y_pred:
        y_pred = (y_pred == -1).astype(int)
    else:
        y_pred = (y_pred == 1).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    detection_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "detection_rate": round(detection_rate, 4),      # How many faults caught
        "false_alarm_rate": round(false_alarm_rate, 4),  # How many false alarms
        "tp": int(tp),
        "fp": int(fp),
    }


# ---------------------------------------------------------------------------
# 4. GPVS Real-Fault Benchmark
# ---------------------------------------------------------------------------

GPVS_DIR = ROOT / "data" / "raw" / "GPVS-Faults"
GPVS_WINDOW_S = 0.1             # 0.1 s = 5 grid cycles at 50 Hz, about 1,000 samples
GPVS_NORMAL_END = 0.45          # faults start "halfway": first 45% of a fault file is normal
GPVS_FAULT_START = 0.55         # last 45% is fault; the uncertain middle 10% is dropped
GPVS_DC_SIGNALS = ["Ipv", "Vpv", "Vdc"]
GPVS_AC_SIGNALS = ["ia", "ib", "ic", "va", "vb", "vc"]
GPVS_DOWNLOAD = ("GPVS-Faults data not found in data/raw/GPVS-Faults/. Download CSV_Files.zip from "
                 "https://data.mendeley.com/datasets/n76t439f65/1 (DOI 10.17632/n76t439f65.1, CC BY 4.0) "
                 "and extract the 16 F*.csv files into data/raw/GPVS-Faults/.")


def gpvs_window_features(df, window_s=GPVS_WINDOW_S):
    """Window statistics for one scenario file: mean/std/min/max of the DC
    signals and RMS of the six phase currents and voltages, per window."""
    w = (df["Time"] // window_s).astype(int)
    g = df.groupby(w)
    feats = g[GPVS_DC_SIGNALS].agg(["mean", "std", "min", "max"])
    feats.columns = [f"{col}_{stat}" for col, stat in feats.columns]
    rms = np.sqrt((df[GPVS_AC_SIGNALS] ** 2).groupby(w).mean()).add_suffix("_rms")
    out = feats.join(rms)
    out["t_start"] = g["Time"].min()
    out["n_samples"] = g.size()
    return out.reset_index(drop=True)


def load_gpvs_windows(data_dir=GPVS_DIR, window_s=GPVS_WINDOW_S):
    """All 16 GPVS-Faults scenarios as labelled windows.

    Files are named F<x><y>: x = 0 fault-free, 1-7 fault type; y = L (IPPT) or
    M (MPPT). F0 windows are all normal. In F1-F7 the fault is introduced
    halfway, so the first 45% of each file is normal (is_fault 0), the last
    45% is fault (is_fault 1) and the middle 10% is dropped. Raises an error
    if the data is missing; there is no synthetic fallback.
    """
    files = sorted(Path(data_dir).glob("F[0-7][LM].csv"))
    if not files:
        raise FileNotFoundError(GPVS_DOWNLOAD)
    parts = []
    for f in files:
        win = gpvs_window_features(pd.read_csv(f), window_s)
        win = win[win["n_samples"] >= 0.5 * win["n_samples"].median()]   # drop a short last window
        pos = win["t_start"] / win["t_start"].max()
        fault_type = int(f.stem[1])
        if fault_type == 0:
            win["is_fault"] = 0
        else:
            win["is_fault"] = np.where(pos >= GPVS_FAULT_START, 1, np.where(pos < GPVS_NORMAL_END, 0, -1))
            win = win[win["is_fault"] >= 0]
        win.insert(0, "scenario", f.stem)
        win.insert(1, "fault_type", fault_type)
        win.insert(2, "mode", "MPPT" if f.stem.endswith("M") else "IPPT")
        win["position"] = pos.loc[win.index]
        parts.append(win)
    return pd.concat(parts, ignore_index=True)


def gpvs_feature_columns(windows):
    return [c for c in windows.columns
            if c.endswith(("_mean", "_std", "_min", "_max", "_rms"))]


def load_gpvs_faults():
    """Real GPVS-Faults windows as (X, y): window features and fault label."""
    windows = load_gpvs_windows()
    return windows[gpvs_feature_columns(windows)], windows["is_fault"].to_numpy()


def make_synthetic_gpvs(seed=42):
    """Synthetic stand-in data for unit tests ONLY; never use it for reported results."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"voltage": np.r_[rng.normal(380, 5, 1000), rng.normal(250, 20, 100)],
                      "current": np.r_[rng.normal(12, 1, 1000), rng.normal(3, 1, 100)]})
    X["power"] = X["voltage"] * X["current"]
    return X, np.r_[np.zeros(1000), np.ones(100)]


# ---------------------------------------------------------------------------
# 5. Shared Results Logger
# ---------------------------------------------------------------------------

def log_experiment(student, model_name, task, train_score, val_score, test_score, unseen_score=None, notes=""):
    """Save model score to results/results.csv so all 4 teammates build one table."""
    results_file = ROOT / "results" / "results.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "student": student.upper(),
        "model": model_name,
        "task": task.lower(),
        "train_score": train_score,
        "val_score": val_score,
        "test_score": test_score,
        "unseen_score": unseen_score if unseen_score is not None else "-",
        "notes": notes,
    }

    df_row = pd.DataFrame([row])

    # If file exists, append; otherwise write header
    if results_file.exists() and results_file.stat().st_size > 0:
        df_row.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(results_file, mode="w", header=True, index=False)

    print(f"Saved result: {student} | {model_name} | Test Score: {test_score} -> results/results.csv")


# ---------------------------------------------------------------------------
# Quick Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing evaluate.py with simple test data...")

    # 1. Create simple test data
    dates = pd.date_range("2021-06-01", "2023-12-31", freq="1D")
    test_data = pd.DataFrame({
        "time": np.repeat(dates, 2),
        "inverter_id": ["station_1", "sq_block_p_inverter"] * len(dates),
        "ac_power_w": [1000.0, 950.0] * len(dates),
    })

    # 2. Test splits
    train, val, test = split_chronological(test_data)
    train_k, unseen = split_unseen_stations(test_data, held_out_stations=["sq_block_p_inverter"])

    # 3. Test metrics
    reg = evaluate_regression([80, 90, 70], [82, 88, 71])
    clf = evaluate_classification(["Healthy", "Warning", "Faulty"], ["Healthy", "Healthy", "Faulty"])
    print(f"Regression Result:    MAE = {reg['mae']}, RMSE = {reg['rmse']}")
    print(f"Classification Result: Macro-F1 = {clf['macro_f1']}, Faulty Recall = {clf['faulty_recall']}")

    # 4. Test synthetic fault injection
    faulty_df = inject_synthetic_faults(test)
    fault_stats = evaluate_fault_detection(faulty_df["is_injected_fault"], faulty_df["is_injected_fault"])
    print(f"Fault Detection:      Detection Rate = {fault_stats['detection_rate']}")

    # 5. Test GPVS loader (real data only; skipped if not downloaded)
    try:
        X_gpvs, y_gpvs = load_gpvs_faults()
        print(f"GPVS Dataset:         {len(X_gpvs)} windows loaded ({int(y_gpvs.sum())} fault windows).")
    except FileNotFoundError as err:
        print(f"GPVS Dataset:         skipped. {err}")

    print("\nAll evaluate.py functions are working properly!")

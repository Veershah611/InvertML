"""Track T5: Isolation Forest anomaly detection (no labels needed).

The forest learns what normal data looks like; points that are easy to isolate
with few random splits are flagged as anomalies.

1. HKUST + injected faults (A1, A2): fit on the training period, then add known
   faults to validation and test (a 35 % power drop for 7 days and a 5-day
   outage on two inverters) and check how many are caught.
   Settings are chosen on validation, the result is reported on test.
2. GPVS-Faults, real lab faults (B1, B2): fit on fault-free windows only,
   test on held-out normal windows and on faulty windows of fault types F1-F7.

Run: python -m src.models.iforest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import RAW, SEED, TABLES
from src.evaluate import evaluation_sets, save_results

HKUST_INPUTS = ["health_now", "peer_dev", "self_dev_30d", "irradiance_wm2"]


def detection_scores(is_fault, flagged):
    is_fault, flagged = np.asarray(is_fault, bool), np.asarray(flagged, bool)
    return {"detection_rate": round(recall_score(is_fault, flagged), 3),
            "false_alarm_rate": round(flagged[~is_fault].mean(), 3),
            "precision": round(precision_score(is_fault, flagged, zero_division=0), 3),
            "f1": round(f1_score(is_fault, flagged), 3)}


def detector(contamination, n_estimators):
    return make_pipeline(StandardScaler(), IsolationForest(
        contamination=contamination, n_estimators=n_estimators, random_state=SEED, n_jobs=-1))


# ---------------------------------------------------------------- 1. HKUST

def inject_faults(part):
    """Known faults on two inverters: a 35 % power drop for 7 days and an outage
    (power 0) for 5 days. Each fault goes on the inverter with the most data in
    that window, so the fault rows really exist."""
    part = part.dropna(subset=HKUST_INPUTS).copy()
    part["injected"] = False
    start = part["time"].min()
    for days_in, length, keep, rank in [(30, 7, 0.65, 0), (60, 5, 0.0, 1)]:
        in_window = part["time"].between(start + pd.Timedelta(days=days_in),
                                         start + pd.Timedelta(days=days_in + length))
        inverter = part.loc[in_window, "inverter_id"].value_counts().index[rank]
        window = in_window & (part["inverter_id"] == inverter)
        old_health = part.loc[window, "health_now"]
        part.loc[window, "health_now"] = old_health * keep
        part.loc[window, "peer_dev"] = (1 + part.loc[window, "peer_dev"]) * keep - 1
        part.loc[window, "self_dev_30d"] -= old_health * (1 - keep)
        part.loc[window, "injected"] = True
    return part


def hkust_benchmark():
    sets = evaluation_sets()
    train = sets["train"].dropna(subset=HKUST_INPUTS)
    val, test = inject_faults(sets["val"]), inject_faults(sets["test"])

    tuning = []
    for contamination in [0.01, 0.02, 0.05]:
        for n_estimators in [100, 200]:
            model = detector(contamination, n_estimators).fit(train[HKUST_INPUTS])
            flagged = model.predict(val[HKUST_INPUTS]) == -1
            tuning.append({"contamination": contamination, "n_estimators": n_estimators,
                           **detection_scores(val["injected"], flagged)})
    tuning = pd.DataFrame(tuning)
    best = tuning.sort_values("f1", ascending=False).iloc[0]
    model = detector(best["contamination"], int(best["n_estimators"])).fit(train[HKUST_INPUTS])
    flagged = model.predict(test[HKUST_INPUTS]) == -1
    result = pd.DataFrame([{"set": "test", "rows": len(test), "injected_rows": int(test["injected"].sum()),
                            "contamination": best["contamination"],
                            "n_estimators": int(best["n_estimators"]),
                            **detection_scores(test["injected"], flagged)}])
    tuning.to_csv(TABLES / "iforest_hkust_tuning.csv", index=False)
    save_results("A1/A2", "Isolation Forest (HKUST injected)", "anomaly", result)
    print(f"\nHKUST injected faults\n{tuning.to_string(index=False)}\n{result.to_string(index=False)}")


# ---------------------------------------------------------------- 2. GPVS-Faults

GPVS_DIR = RAW / "GPVS-Faults"
DC_SIGNALS, AC_SIGNALS = ["Ipv", "Vpv", "Vdc"], ["ia", "ib", "ic", "va", "vb", "vc"]


def gpvs_windows():
    """Each scenario file F<type><mode>.csv cut into 0.1 s windows.
    F0 = no fault. In F1-F7 the fault starts halfway: the first 45 % of the
    file is normal, the last 45 % is faulty, the uncertain middle is dropped."""
    files = sorted(GPVS_DIR.glob("F[0-7][LM].csv"))
    if not files:
        raise FileNotFoundError("Put the 16 GPVS-Faults CSV files in data/raw/GPVS-Faults/")
    parts = []
    for f in files:
        df = pd.read_csv(f)
        window = (df["Time"] // 0.1).astype(int)
        g = df.groupby(window)
        feats = g[DC_SIGNALS].agg(["mean", "std", "min", "max"])
        feats.columns = [f"{a}_{b}" for a, b in feats.columns]
        feats = feats.join(np.sqrt((df[AC_SIGNALS] ** 2).groupby(window).mean()).add_suffix("_rms"))
        feats = feats[g.size() >= 0.5 * g.size().median()]          # drop a short last window
        position = np.linspace(0, 1, len(feats))
        fault_type = int(f.stem[1])
        feats["is_fault"] = 0 if fault_type == 0 else np.where(position >= 0.55, 1,
                                                                np.where(position < 0.45, 0, -1))
        feats["scenario"], feats["fault_type"], feats["position"] = f.stem, fault_type, position
        parts.append(feats[feats["is_fault"] >= 0])
    return pd.concat(parts, ignore_index=True)


def gpvs_benchmark():
    windows = gpvs_windows()
    inputs = [c for c in windows.columns if c.endswith(("_mean", "_std", "_min", "_max", "_rms"))]
    no_fault_file = windows["fault_type"] == 0
    early = windows["position"] < 0.7
    # train: first 70 % of the fault-free files + the normal first part of fault files
    train = windows[(no_fault_file & early) | (~no_fault_file & (windows["is_fault"] == 0))]
    # test: last 30 % of the fault-free files (normal) + every faulty window
    test = windows[(no_fault_file & ~early) | (windows["is_fault"] == 1)]

    model = detector(0.05, 200).fit(train[inputs])
    flagged = model.predict(test[inputs]) == -1
    result = pd.DataFrame([{"set": "test", "rows": len(test), "fault_windows": int(test["is_fault"].sum()),
                            **detection_scores(test["is_fault"], flagged)}])
    per_scenario = (test.assign(flagged=flagged).groupby(["scenario", "is_fault"])["flagged"]
                    .mean().round(3).rename("flagged_share").reset_index())
    per_scenario.to_csv(TABLES / "iforest_gpvs_by_scenario.csv", index=False)
    save_results("B1/B2", "Isolation Forest (GPVS real faults)", "anomaly", result)
    print(f"\nGPVS-Faults\n{result.to_string(index=False)}\n{per_scenario.to_string(index=False)}")


def main():
    hkust_benchmark()
    gpvs_benchmark()


if __name__ == "__main__":
    main()

"""Stage 11 evaluation of the Track T1 linear models and baselines (A1).

Builds on linear.py (same target, features, splits and tuning) and adds the
section 11.2 diagnostics:
    - Ridge residual histogram and actual-vs-predicted plots (val, test)
    - error by season and by inverter on the test year
    - confusion matrices (fixed Healthy/Warning/Faulty order) and Faulty PR-AUC
      for every classifier on every split
    - the section 8 rule-based classifier on peer deviation: peer_dev_campus at
      t, with its two cut-offs tuned on train macro-F1 and then frozen. Offline
      bins (peer deviation undefined) are predicted Faulty, as B1's target
      scores them health 0.

Seasons are meteorological: DJF winter, MAM spring, JJA summer, SON autumn.

Run from the repo root:  python -m src.models.linear_evaluation
"""
import json
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from src.evaluate import evaluate_classification, evaluate_regression
from src.features import add_peer_deviation_features
from src.models.linear import (CLASSES, PROCESSED, ROOT, TABLES, build_dataset,
                               fill_from_train, fit_logistic, fit_ridge, split)

FIGURES = ROOT / "reports" / "figures"
SEASONS = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
           3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
           6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
           9: "Autumn (SON)", 10: "Autumn (SON)", 11: "Autumn (SON)"}
CUTOFF_GRID = np.round(np.arange(-0.60, 0.001, 0.025), 3)
OFFLINE_SCORE = 4.0


# ---------------------------------------------------------------- data

def add_peer_deviation(data):
    """peer_dev_campus at t, computed on the full grid so every inverter
    online at that timestamp counts toward the campus median."""
    grid = pd.read_parquet(PROCESSED / "hkust_15min_v1.parquet",
                           columns=["inverter_id", "station_id", "time", "ac_power_w",
                                    "n_valid_5min", "irradiance_wm2", "capacity_kwp"])
    grid["inverter_id"] = grid["inverter_id"].astype(str)
    peer = add_peer_deviation_features(grid)[["inverter_id", "time", "peer_dev_campus"]]
    return data.merge(peer, on=["inverter_id", "time"], how="left", validate="one_to_one")


# ---------------------------------------------------------------- peer rule

def peer_rule(peer_dev, warn_cut, fault_cut):
    pred = np.where(peer_dev < warn_cut, "Warning", "Healthy").astype(object)
    pred[(peer_dev < fault_cut) | np.isnan(peer_dev)] = "Faulty"
    return pred


def tune_peer_rule(train):
    """Grid-search (Warning, Faulty) cut-offs on train macro-F1, fault_cut < warn_cut."""
    x = train["peer_dev_campus"].to_numpy(dtype=float)
    y = train["status_lead"].to_numpy()
    best, rows = None, []
    for warn_cut, fault_cut in product(CUTOFF_GRID, CUTOFF_GRID):
        if fault_cut >= warn_cut:
            continue
        f1 = f1_score(y, peer_rule(x, warn_cut, fault_cut), average="macro", zero_division=0)
        rows.append({"warn_cut": warn_cut, "fault_cut": fault_cut, "train_macro_f1": round(f1, 4)})
        if best is None or f1 > best[2]:
            best = (warn_cut, fault_cut, f1)
    return best[0], best[1], pd.DataFrame(rows)


# ---------------------------------------------------------------- diagnostics

def classifier_table(parts, preds, scores):
    """Macro-F1, Faulty recall/precision, Faulty PR-AUC and confusion matrix."""
    rows = []
    for model in preds:
        for name, part in parts.items():
            y = part["status_lead"].to_numpy()
            m = evaluate_classification(y, preds[model][name])
            ap = average_precision_score(y == "Faulty", scores[model][name])
            rows.append({"model": model, "split": name, "n": m["n_samples"],
                         "macro_f1": m["macro_f1"], "faulty_recall": m["faulty_recall"],
                         "faulty_precision": m["faulty_precision"],
                         "faulty_pr_auc": round(float(ap), 4),
                         "confusion_matrix": json.dumps(confusion_matrix(
                             y, preds[model][name], labels=CLASSES).tolist())})
    return pd.DataFrame(rows)


def by_group(part, reg_preds, cls_preds, key):
    """Regression and classification scores per group (season or inverter)."""
    rows = []
    for group, idx in part.groupby(key, observed=True).indices.items():
        g = part.iloc[idx]
        row = {key: group, "n": len(g)}
        for model, pred in reg_preds.items():
            m = evaluate_regression(g["health_lead"], pred[idx])
            row[f"{model}_mae"], row[f"{model}_rmse"] = m["mae"], m["rmse"]
        for model, pred in cls_preds.items():
            m = evaluate_classification(g["status_lead"], pred[idx])
            row[f"{model}_macro_f1"], row[f"{model}_faulty_recall"] = m["macro_f1"], m["faulty_recall"]
        row["faulty_share"] = round(float(g["status_lead"].eq("Faulty").mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_residuals(parts, ridge_pred, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, name in enumerate(["val", "test"]):
        y = parts[name]["health_lead"].to_numpy()
        p = ridge_pred[name]
        resid = y - p
        ax = axes[0, col]
        ax.hist(resid, bins=80, color="#4C72B0")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"Ridge residuals, {name} (mean {resid.mean():.2f}, sd {resid.std():.2f})")
        ax.set_xlabel("actual - predicted Health(t+1h)")
        ax.set_ylabel("rows")
        ax = axes[1, col]
        hb = ax.hexbin(y, p, gridsize=50, bins="log", cmap="viridis", mincnt=1)
        ax.plot([0, 100], [0, 100], color="red", lw=1)
        ax.set_title(f"Actual vs predicted, {name}")
        ax.set_xlabel("actual Health(t+1h)")
        ax.set_ylabel("predicted")
        fig.colorbar(hb, ax=ax, label="rows (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- run

def run():
    data = add_peer_deviation(build_dataset())
    data["season"] = data["time"].dt.month.map(SEASONS)
    parts = split(data)
    for name, part in parts.items():
        print(f"  {name:18s} {len(part):8,} rows")

    train = parts["train"].sort_values("time")
    X_train = fill_from_train(train, train)
    X = {name: fill_from_train(train, p) for name, p in parts.items()}

    print("Tuning Ridge and Logistic (as in linear.py) ...")
    ridge = fit_ridge(X_train, train["health_lead"].to_numpy())
    logit = fit_logistic(X_train, train["status_lead"].to_numpy())
    print("Tuning peer-deviation rule cut-offs on train ...")
    warn_cut, fault_cut, rule_grid = tune_peer_rule(train)
    print(f"  Warning below {warn_cut}, Faulty below {fault_cut}")

    faulty = list(logit.best_estimator_.classes_).index("Faulty")
    preds = {"Logistic Regression": {}, "Rule: status of Health(t)": {}, "Rule: peer deviation": {}}
    scores = {model: {} for model in preds}
    ridge_pred, persist_pred = {}, {}
    for name, part in parts.items():
        peer = part["peer_dev_campus"].to_numpy(dtype=float)
        preds["Logistic Regression"][name] = logit.predict(X[name])
        scores["Logistic Regression"][name] = logit.predict_proba(X[name])[:, faulty]
        preds["Rule: status of Health(t)"][name] = part["status_now"].to_numpy()
        scores["Rule: status of Health(t)"][name] = -part["health_now"].to_numpy()
        preds["Rule: peer deviation"][name] = peer_rule(peer, warn_cut, fault_cut)
        # Peer deviation is clipped to [-3, 3]; offline rows rank above every online row.
        scores["Rule: peer deviation"][name] = np.where(np.isnan(peer), OFFLINE_SCORE, -peer)
        ridge_pred[name] = ridge.predict(X[name])
        persist_pred[name] = part["health_now"].to_numpy()

    classifiers = classifier_table(parts, preds, scores)
    test = parts["test"]
    reg = {"ridge": ridge_pred["test"], "persistence": persist_pred["test"]}
    cls = {"logistic": preds["Logistic Regression"]["test"],
           "rule_health": preds["Rule: status of Health(t)"]["test"],
           "rule_peer": preds["Rule: peer deviation"]["test"]}
    season = by_group(test, reg, cls, "season")
    station = by_group(test, reg, cls, "inverter_id").sort_values("ridge_rmse", ascending=False)
    rule = pd.DataFrame([{"warn_cut": warn_cut, "fault_cut": fault_cut,
                          "train_macro_f1": rule_grid["train_macro_f1"].max(),
                          "offline_rows_as_faulty": True}])

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    classifiers.to_csv(TABLES / "a1_eval_classifiers.csv", index=False)
    season.to_csv(TABLES / "a1_eval_by_season.csv", index=False)
    station.to_csv(TABLES / "a1_eval_by_inverter.csv", index=False)
    rule.to_csv(TABLES / "a1_peer_rule.csv", index=False)
    plot_residuals(parts, ridge_pred, FIGURES / "a1_ridge_residuals.png")
    return {"classifiers": classifiers, "season": season, "station": station, "rule": rule}


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 250)
    print(out["classifiers"].drop(columns="confusion_matrix").to_string(index=False), "\n")
    print(out["season"].to_string(index=False), "\n")
    print(out["station"].head(8).to_string(index=False))

"""Stage 8 baselines + Track T1 linear models (A1).

Target: B1's health score (src/targets.py), forecasting mode (Option A, §7.2):
predict Health(t + 1 h) from information available at time t.

Regression:      (a) persistence  Health(t+1h) = Health(t)
                 (b) Ridge regression
Classification:  (a) majority class
                 (b) rule-based: status implied by the current Health(t)
                 (c) Logistic Regression (class_weight="balanced")

Ridge and Logistic are tuned with GridSearchCV over a 5-fold TimeSeriesSplit on
the training period only. Every model is scored on train / val / test, on the
test year split at the 2023-02-14 irradiance sensor change, and on the unseen
inverters. Results go to reports/tables/a1_baseline_results.csv and
results/results.csv (shared logger).

Run from the repo root:  python -m src.models.linear          (no logging)
                         python -m src.models.linear --log    (also log to results.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import CONFIG, evaluate_classification, evaluate_regression, log_experiment
from src.features import B2_ALL_FEATURES, build_b2_features
from src.targets import assign_operational_status, compute_health_score

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / CONFIG["paths"]["processed"]
TABLES = ROOT / "reports" / "tables"
SEED = CONFIG["random_state"]

HORIZON = pd.Timedelta("1h")
SENSOR_CHANGE = pd.Timestamp("2023-02-14")   # irradiance step change found in A1 EDA
TRAIN_END = pd.Timestamp(CONFIG["split"]["train_end"]) + pd.Timedelta("1D")
VAL_END = pd.Timestamp(CONFIG["split"]["val_end"]) + pd.Timedelta("1D")
CLASSES = ["Healthy", "Warning", "Faulty"]

WEATHER_FEATURES = ["irradiance_wm2", "temp_c", "rh_pct", "wind_speed_ms"]
FEATURES = ["health_now", "n_valid_5min"] + WEATHER_FEATURES + B2_ALL_FEATURES

LOAD_COLUMNS = ["inverter_id", "time", "ac_power_w", "n_valid_5min", "capacity_kwp",
                "tilt_deg", "azimuth", "altitude_m"] + WEATHER_FEATURES


# ---------------------------------------------------------------- data

def build_dataset(path=PROCESSED / "hkust_15min_v1.parquet"):
    """One row per (inverter, t) with features at t and the target at t + 1 h.

    The lead is taken on the full 15-min grid and kept only when the row 4 bins
    ahead is exactly 1 hour later (same day), so "+1 h" never jumps overnight.
    Rows where Health(t) or Health(t+1h) is undefined (irradiance < 100 W/m2,
    B1's daylight rule) are dropped.
    """
    df = pd.read_parquet(path, columns=LOAD_COLUMNS)
    df["inverter_id"] = df["inverter_id"].astype(str)
    df = df.sort_values(["inverter_id", "time"]).reset_index(drop=True)

    df = compute_health_score(df)           # B1's target, unchanged
    df = df.rename(columns={"health_score": "health_now"})

    g = df.groupby("inverter_id")
    lead_time = g["time"].shift(-4)
    df["health_lead"] = g["health_now"].shift(-4).where(lead_time == df["time"] + HORIZON)

    df = df.dropna(subset=["health_now", "health_lead"])
    df["status_now"] = assign_operational_status(df["health_now"])
    df["status_lead"] = assign_operational_status(df["health_lead"])
    df = build_b2_features(df)
    df[WEATHER_FEATURES] = df[WEATHER_FEATURES].astype("float64")
    return df.reset_index(drop=True)


def split(df):
    """Chronological split (config dates) plus evaluation subsets of the test year."""
    t = df["time"]
    unseen = set(CONFIG["unseen_stations"])
    is_unseen = df["inverter_id"].isin(unseen)
    test = df[t >= VAL_END]
    return {
        # Unseen inverters never enter training or validation.
        "train": df[(t < TRAIN_END) & ~is_unseen],
        "val": df[(t >= TRAIN_END) & (t < VAL_END) & ~is_unseen],
        "test": test,
        "test_before_14feb": test[test["time"] < SENSOR_CHANGE],
        "test_after_14feb": test[test["time"] >= SENSOR_CHANGE],
        "test_unseen": test[test["inverter_id"].isin(unseen)],
    }


def fill_from_train(train, other):
    """Fill missing feature values with training-set means (fitted on train only)."""
    means = train[FEATURES].mean()
    return other[FEATURES].fillna(means)


# ---------------------------------------------------------------- models

def fit_ridge(X, y):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    search = GridSearchCV(pipe, {"model__alpha": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]},
                          cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X, y)
    return search


def fit_logistic(X, y):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                                  random_state=SEED))])
    search = GridSearchCV(pipe, {"model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                          cv=TimeSeriesSplit(n_splits=5), scoring="f1_macro", n_jobs=-1)
    search.fit(X, y)
    return search


def default_vs_tuned(X_train, train, X, y_reg, y_cls, ridge, logit):
    """Same pipelines with sklearn's default hyperparameters (alpha=1, C=1)
    against the tuned ones, scored on validation and test."""
    defaults = {
        "Ridge": (Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
                  .fit(X_train, train["health_lead"]), ridge.best_estimator_),
        "Logistic Regression": (Pipeline([("scaler", StandardScaler()),
                                          ("model", LogisticRegression(class_weight="balanced",
                                                                       max_iter=2000, random_state=SEED))])
                                .fit(X_train, train["status_lead"]), logit.best_estimator_),
    }
    rows = []
    for model, (default, tuned) in defaults.items():
        for version, est in [("default", default), ("tuned", tuned)]:
            row = {"model": model, "version": version,
                   "params": str(est.named_steps["model"].get_params()["alpha" if model == "Ridge" else "C"])}
            for name in ["val", "test"]:
                pred = est.predict(X[name])
                if model == "Ridge":
                    row[f"{name}_rmse"] = evaluate_regression(y_reg[name], pred)["rmse"]
                else:
                    m = evaluate_classification(y_cls[name], pred)
                    row[f"{name}_macro_f1"] = m["macro_f1"]
                    row[f"{name}_faulty_recall"] = m["faulty_recall"]
            rows.append(row)
    return pd.DataFrame(rows)


def permutation_importances(ridge, logit, X_val, y_reg_val, y_cls_val, n=20000):
    """Permutation importance on a validation sample: how much the score drops
    when one feature is shuffled (RMSE for Ridge, macro-F1 for Logistic)."""
    idx = np.random.RandomState(SEED).choice(len(X_val), size=min(n, len(X_val)), replace=False)
    Xs = X_val.iloc[idx]
    out = {}
    for model, est, y, scoring in [("Ridge", ridge.best_estimator_, y_reg_val[idx], "neg_root_mean_squared_error"),
                                   ("Logistic Regression", logit.best_estimator_, y_cls_val[idx], "f1_macro")]:
        r = permutation_importance(est, Xs, y, scoring=scoring, n_repeats=5,
                                   random_state=SEED, n_jobs=-1)
        out[model] = r.importances_mean
    imp = pd.DataFrame(out, index=FEATURES)
    return imp.sort_values("Ridge", ascending=False).round(4)


def failure_cases(part, ridge, X_part, n=3):
    """The n largest Ridge errors in a split, with the context needed to explain them."""
    pred = ridge.predict(X_part)
    cases = part.assign(predicted=pred, abs_error=np.abs(part["health_lead"] - pred))
    cols = ["inverter_id", "time", "health_now", "health_lead", "predicted", "abs_error",
            "irradiance_wm2", "n_valid_5min", "temp_c"]
    worst = cases.nlargest(n, "abs_error")[cols].reset_index(drop=True)
    numeric = worst.select_dtypes("number").columns
    worst[numeric] = worst[numeric].round(2)
    return worst


def coefficients(search, classes=None):
    """Coefficients on the standardised scale, so magnitudes are comparable."""
    model = search.best_estimator_.named_steps["model"]
    coef = np.atleast_2d(model.coef_)
    names = classes if classes is not None else ["coef"]
    return pd.DataFrame(coef.T, index=FEATURES, columns=names)


# ---------------------------------------------------------------- run

def run(log=False):
    """Fit and score all baselines. log=True appends one row per model to the
    shared results/results.csv (use it once, for the final run)."""
    data = build_dataset()
    parts = split(data)
    for name, part in parts.items():
        print(f"  {name:18s} {len(part):8,} rows")

    train = parts["train"].sort_values("time")      # TimeSeriesSplit needs time order
    X_train = fill_from_train(train, train)
    X = {name: fill_from_train(train, p) for name, p in parts.items()}
    y_reg = {name: p["health_lead"].to_numpy() for name, p in parts.items()}
    y_cls = {name: p["status_lead"].to_numpy() for name, p in parts.items()}

    print("Tuning Ridge ...")
    ridge = fit_ridge(X_train, train["health_lead"].to_numpy())
    print(f"  best alpha = {ridge.best_params_['model__alpha']}")
    print("Tuning Logistic Regression ...")
    logit = fit_logistic(X_train, train["status_lead"].to_numpy())
    print(f"  best C = {logit.best_params_['model__C']}")
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, train["status_lead"])

    predictors = {
        ("regression", "Persistence"): lambda name: parts[name]["health_now"].to_numpy(),
        ("regression", "Ridge"): lambda name: ridge.predict(X[name]),
        ("classification", "Majority class"): lambda name: majority.predict(X[name]),
        ("classification", "Rule: status of Health(t)"): lambda name: parts[name]["status_now"].to_numpy(),
        ("classification", "Logistic Regression"): lambda name: logit.predict(X[name]),
    }

    rows = []
    for (task, model), predict in predictors.items():
        for name in parts:
            pred = predict(name)
            if task == "regression":
                m = evaluate_regression(y_reg[name], pred)
                rows.append({"task": task, "model": model, "split": name, "n": m["n_samples"],
                             "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
            else:
                m = evaluate_classification(y_cls[name], pred)
                rows.append({"task": task, "model": model, "split": name, "n": m["n_samples"],
                             "macro_f1": m["macro_f1"], "faulty_recall": m["faulty_recall"],
                             "faulty_precision": m["faulty_precision"]})
    results = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES / "a1_baseline_results.csv", index=False)

    coef_ridge = coefficients(ridge)
    coef_logit = coefficients(logit, classes=list(logit.best_estimator_.classes_))
    coef_ridge.to_csv(TABLES / "a1_ridge_coefficients.csv")
    coef_logit.to_csv(TABLES / "a1_logistic_coefficients.csv")

    tuning = default_vs_tuned(X_train, train, X, y_reg, y_cls, ridge, logit)
    tuning.to_csv(TABLES / "a1_default_vs_tuned.csv", index=False)
    importance = permutation_importances(ridge, logit, X["val"], y_reg["val"], y_cls["val"])
    importance.to_csv(TABLES / "a1_permutation_importance.csv")
    failures = {name: failure_cases(parts[name], ridge, X[name]) for name in ["val", "test"]}
    pd.concat(failures, names=["split"]).to_csv(TABLES / "a1_failure_cases.csv")

    if log:
        log_results(results, ridge, logit)
    return {"data": data, "parts": parts, "results": results, "ridge": ridge,
            "logit": logit, "coef_ridge": coef_ridge, "coef_logit": coef_logit,
            "tuning": tuning, "importance": importance, "failures": failures}


def log_results(results, ridge, logit):
    """One row per model in the shared results/results.csv."""
    notes = {
        "Persistence": "Health(t+1h) = Health(t); B1 target, forecasting t+1h",
        "Ridge": f"alpha={ridge.best_params_['model__alpha']}; 5-fold TimeSeriesSplit",
        "Majority class": "always predicts the most frequent training class",
        "Rule: status of Health(t)": "status at t+1h = status implied by Health(t)",
        "Logistic Regression": f"C={logit.best_params_['model__C']}; class_weight=balanced",
    }
    for (task, model), r in results.groupby(["task", "model"], sort=False):
        r = r.set_index("split")
        if task == "regression":
            fmt = lambda s: f"RMSE: {r.loc[s, 'rmse']:.2f}"
        else:
            fmt = lambda s: f"F1: {r.loc[s, 'macro_f1']:.2f}, Faulty recall: {r.loc[s, 'faulty_recall']:.2f}"
        log_experiment(student="A1", model_name=model, task=task,
                       train_score=fmt("train"), val_score=fmt("val"), test_score=fmt("test"),
                       unseen_score=fmt("test_unseen"),
                       notes=f"{notes[model]}; test before/after 14 Feb: "
                             f"{fmt('test_before_14feb')} / {fmt('test_after_14feb')}")


if __name__ == "__main__":
    import sys
    out = run(log="--log" in sys.argv)
    pd.set_option("display.width", 200)
    print(out["results"].to_string(index=False))

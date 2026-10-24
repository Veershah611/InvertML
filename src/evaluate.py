"""Stage 11: shared evaluation (B2 lead), used by every model.

- metrics: RMSE / MAE / R2 for the health score, macro-F1 and Faulty recall for the status
- evaluation sets: train, val, test, test before / after the 2023-02-14 sensor change,
  and unseen inverters (2023 rows of the six held-out inverters)
- run_model(): the same recipe for every model family (Stages 9, 10, 11)
    1. fit with default settings
    2. tune on the training period with a 3-fold TimeSeriesSplit (no shuffling)
    3. score the tuned model on every evaluation set
    4. permutation importance (explainability) and worst predictions (failure cases)
    5. save tables and update results/results.csv
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.common import CONFIG, RESULTS, SEED, TABLES
from src.dataset import FEATURES, load

SENSOR_CHANGE = pd.Timestamp(CONFIG["irradiance_sensor_change"])
RESULTS_FILE = RESULTS / "results.csv"


# ---------------------------------------------------------------- data

def evaluation_sets(df=None):
    """Named subsets of the model table. Train is sorted by time for TimeSeriesSplit."""
    df = load() if df is None else df
    test = df[df["split"] == "test"]
    return {
        "train": df[df["split"] == "train"].sort_values("time"),
        "val": df[df["split"] == "val"],
        "test": test,
        "test_before_14feb": test[test["time"] < SENSOR_CHANGE],
        "test_after_14feb": test[test["time"] >= SENSOR_CHANGE],
        "unseen": test[test["unseen"]],
    }


# ---------------------------------------------------------------- metrics

def regression_scores(y_true, y_pred):
    return {"rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "r2": round(float(r2_score(y_true, y_pred)), 3)}


def classification_scores(y_true, y_pred):
    faulty_true, faulty_pred = np.asarray(y_true) == "Faulty", np.asarray(y_pred) == "Faulty"
    return {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 3),
            "faulty_recall": round(float(recall_score(faulty_true, faulty_pred)), 3),
            "faulty_precision": round(float(precision_score(faulty_true, faulty_pred,
                                                            zero_division=0)), 3)}


def score_all(task, predict, sets):
    """One row per evaluation set. `predict(part)` returns predictions for that part."""
    rows = []
    for name, part in sets.items():
        if task == "regression":
            scores = regression_scores(part["health_lead"], predict(part))
        else:
            scores = classification_scores(part["status_lead"], predict(part))
        rows.append({"set": name, "rows": len(part), **scores})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- results file

def save_results(owner, model, task, table):
    """Replace this model's rows in results/results.csv (no duplicates on re-runs)."""
    new = table.assign(owner=owner, model=model, task=task)
    if RESULTS_FILE.exists():
        old = pd.read_csv(RESULTS_FILE)
        old = old[~((old["model"] == model) & (old["task"] == task))]
        new = pd.concat([old, new], ignore_index=True)
    first = ["owner", "model", "task", "set", "rows"]
    new[first + [c for c in new.columns if c not in first]].to_csv(RESULTS_FILE, index=False)


# ---------------------------------------------------------------- models

def make_pipeline(model, scale):
    """Fill missing features with the training median, scale if the model needs it."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def fit(pipe, X, y, task):
    """Classifiers get balanced sample weights (Stage 5 decision): every class
    counts equally, so the rare Faulty class is not ignored."""
    if task == "classification":
        return pipe.fit(X, y, model__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


def tune(pipe, grid, train, task, tune_rows):
    """GridSearch with a 3-fold TimeSeriesSplit on (a time-ordered sample of) train."""
    if tune_rows and len(train) > tune_rows:
        train = train.sample(tune_rows, random_state=SEED).sort_values("time")
    target = "health_lead" if task == "regression" else "status_lead"
    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    search = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=3), scoring=scoring, n_jobs=-1)
    if task == "classification":
        weights = compute_sample_weight("balanced", train[target])
        search.fit(train[FEATURES], train[target], model__sample_weight=weights)
    else:
        search.fit(train[FEATURES], train[target])
    return search.best_params_


def run_model(owner, name, task, model, grid, scale=False, tune_rows=None):
    """Default fit -> tuning -> tuned fit -> scores, importance, failure cases."""
    print(f"\n{name} ({task})")
    sets = evaluation_sets()
    train = sets["train"]
    target = "health_lead" if task == "regression" else "status_lead"
    X, y = train[FEATURES], train[target]

    default = fit(make_pipeline(clone(model), scale), X, y, task)
    best_params = tune(make_pipeline(clone(model), scale), grid, train, task, tune_rows)
    tuned = fit(make_pipeline(clone(model), scale).set_params(**best_params), X, y, task)
    print(f"  best settings: {best_params}")

    scores = score_all(task, lambda part: tuned.predict(part[FEATURES]), sets)
    default_vs_tuned = pd.concat([
        score_all(task, lambda part: m.predict(part[FEATURES]),
                  {"val": sets["val"], "test": sets["test"]}).assign(version=v)
        for v, m in [("default", default), ("tuned", tuned)]])
    default_vs_tuned["tuned_settings"] = str(best_params)

    tag = f"{name.lower().replace(' ', '_')}_{task}"
    scores.to_csv(TABLES / f"{tag}_scores.csv", index=False)
    default_vs_tuned.to_csv(TABLES / f"{tag}_default_vs_tuned.csv", index=False)
    importance(tuned, sets["val"], task).to_csv(TABLES / f"{tag}_importance.csv")
    if task == "regression":
        failure_cases(tuned, sets["test"]).to_csv(TABLES / f"{tag}_failures.csv", index=False)
    save_results(owner, name, task, scores)
    print(scores.to_string(index=False))
    return tuned, scores


def importance(model, val, task, rows=5000):
    """Permutation importance: how much the validation score drops when one
    feature is shuffled. Bigger drop = the model relies more on that feature."""
    sample = val.sample(min(rows, len(val)), random_state=SEED)
    target = "health_lead" if task == "regression" else "status_lead"
    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1_macro"
    result = permutation_importance(model, sample[FEATURES], sample[target], scoring=scoring,
                                    n_repeats=3, random_state=SEED)
    return (pd.Series(result.importances_mean, index=FEATURES, name="importance")
            .sort_values(ascending=False).round(4))


def failure_cases(model, test, n=5):
    """The n largest health-score errors on the test year, with their context."""
    out = test.assign(predicted=model.predict(test[FEATURES]))
    out["error"] = (out["health_lead"] - out["predicted"]).abs()
    cols = ["inverter_id", "time", "health_now", "health_lead", "predicted", "error",
            "irradiance_wm2", "minutes_offline_24h"]
    return out.nlargest(n, "error")[cols].round({c: 2 for c in cols[2:]})

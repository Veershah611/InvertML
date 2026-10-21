"""Stage 5 data balancing for the Track T1 linear models (A1).

Classification (Logistic Regression), two strategies, as A2 did for RF:
    class_weight  class_weight="balanced" on the full training set
    undersampling random Healthy undersampling, 2 Healthy rows per non-Healthy
                  row (A2's rule), training rows only
SMOTE is not used: interpolating between rows of a time series invents
readings that never happened and mixes rows from different days and inverters.

C is re-tuned for each strategy with the same grid and 5-fold TimeSeriesSplit
as linear.py. Undersampling is applied inside each training fold only, so no
validation fold, the validation set or the test set is ever resampled. The
winner is picked on validation Faulty recall, then macro-F1 (A2's rule).

Regression is not balanced. Instead this reports the Health(t+1h) distribution
and Ridge's validation error per score range.

Run from the repo root:  python -m src.models.linear_balancing
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import evaluate_classification, evaluate_regression
from src.models.linear import (CLASSES, SEED, TABLES, build_dataset, fill_from_train,
                               fit_ridge, split)

C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
HEALTHY_PER_OTHER = 2
SCORE_RANGES = [0, 70, 90, 100.001]          # Faulty / Warning / Healthy thresholds
RANGE_LABELS = ["0-70 (Faulty)", "70-90 (Warning)", "90-100 (Healthy)"]


def logistic(C, class_weight):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=C, class_weight=class_weight,
                                                  max_iter=2000, random_state=SEED))])


def undersample_healthy(X, y):
    """Keep HEALTHY_PER_OTHER Healthy rows per non-Healthy row, in time order."""
    y = pd.Series(y, index=X.index)
    healthy = y.index[y.eq("Healthy")]
    other = y.index[y.ne("Healthy")]
    keep = min(len(healthy), max(1, HEALTHY_PER_OTHER * len(other)))
    sampled = pd.Index(np.random.RandomState(SEED).choice(healthy, size=keep, replace=False))
    rows = X.index[X.index.isin(sampled.union(other))]   # original (time) order
    return X.loc[rows], y.loc[rows].to_numpy()


STRATEGIES = {
    "class_weight": {"class_weight": "balanced", "resample": None},
    "undersampling": {"class_weight": None, "resample": undersample_healthy},
}


def resampled(X, y, resample):
    return (X, y) if resample is None else resample(X, y)


def tune_c(X, y, class_weight, resample):
    """Mean macro-F1 over a 5-fold TimeSeriesSplit for every C. Resampling is
    applied to each training fold only; validation folds keep every row."""
    scores = {}
    for C in C_GRID:
        fold_scores = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X):
            X_fold, y_fold = resampled(X.iloc[train_idx], y[train_idx], resample)
            model = logistic(C, class_weight).fit(X_fold, y_fold)
            fold_scores.append(f1_score(y[val_idx], model.predict(X.iloc[val_idx]),
                                        average="macro", zero_division=0))
        scores[C] = float(np.mean(fold_scores))
    best = max(scores, key=scores.get)
    return best, scores


def compare_balancing(X_train, y_train, X_val, y_val):
    rows, cv = [], []
    for strategy, cfg in STRATEGIES.items():
        best_c, scores = tune_c(X_train, y_train, **cfg)
        cv += [{"strategy": strategy, "C": c, "cv_macro_f1": round(s, 4)} for c, s in scores.items()]
        X_fit, y_fit = resampled(X_train, y_train, cfg["resample"])
        model = logistic(best_c, cfg["class_weight"]).fit(X_fit, y_fit)
        pred = model.predict(X_val)
        m = evaluate_classification(y_val, pred)
        faulty = list(model.named_steps["model"].classes_).index("Faulty")
        ap = average_precision_score(y_val == "Faulty", model.predict_proba(X_val)[:, faulty])
        rows.append({"strategy": strategy, "best_C": best_c, "train_rows": len(X_fit),
                     "val_rows": len(X_val), "macro_f1": m["macro_f1"],
                     "faulty_recall": m["faulty_recall"],
                     "faulty_precision": m["faulty_precision"],
                     "faulty_pr_auc": round(float(ap), 4),
                     "confusion_matrix": json.dumps(
                         confusion_matrix(y_val, pred, labels=CLASSES).tolist())})
    result = pd.DataFrame(rows)
    winner = result.sort_values(["faulty_recall", "macro_f1"], ascending=False).iloc[0]["strategy"]
    result["selected"] = result["strategy"].eq(winner)
    return result, pd.DataFrame(cv)


def regression_by_range(parts, X, ridge):
    """Health(t+1h) distribution per split, and Ridge validation error per score range."""
    dist = []
    for name in ["train", "val", "test"]:
        counts = pd.cut(parts[name]["health_lead"], SCORE_RANGES, labels=RANGE_LABELS,
                        right=False).value_counts(normalize=True).reindex(RANGE_LABELS)
        dist.append({"split": name, "rows": len(parts[name]),
                     **{label: round(float(v), 4) for label, v in counts.items()}})
    val = parts["val"].assign(predicted=ridge.predict(X["val"]))
    val["range"] = pd.cut(val["health_lead"], SCORE_RANGES, labels=RANGE_LABELS, right=False)
    errors = []
    for label, g in val.groupby("range", observed=True):
        m = evaluate_regression(g["health_lead"], g["predicted"])
        errors.append({"range": label, "rows": m["n_samples"], "mae": m["mae"],
                       "rmse": m["rmse"], "mean_error": round(float((g["predicted"] - g["health_lead"]).mean()), 3)})
    return pd.DataFrame(dist), pd.DataFrame(errors)


def run():
    data = build_dataset()
    parts = split(data)
    train = parts["train"].sort_values("time")      # TimeSeriesSplit needs time order
    X_train = fill_from_train(train, train)
    X = {name: fill_from_train(train, p) for name, p in parts.items()}
    y_train = train["status_lead"].to_numpy()

    print("Comparing balancing strategies (C re-tuned for each) ...")
    result, cv = compare_balancing(X_train, y_train, X["val"], parts["val"]["status_lead"].to_numpy())
    print("Tuning Ridge for the error-by-range table ...")
    ridge = fit_ridge(X_train, train["health_lead"].to_numpy())
    dist, errors = regression_by_range(parts, X, ridge)

    TABLES.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLES / "a1_balancing.csv", index=False)
    cv.to_csv(TABLES / "a1_balancing_cv.csv", index=False)
    dist.to_csv(TABLES / "a1_target_distribution.csv", index=False)
    errors.to_csv(TABLES / "a1_ridge_error_by_range.csv", index=False)
    return {"result": result, "cv": cv, "distribution": dist, "errors": errors}


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    for name in ["result", "distribution", "errors"]:
        print(out[name].to_string(index=False), "\n")

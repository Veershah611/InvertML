"""A2 Random Forest regression, classification, balancing and explanations.

Run after ``python -m src.preprocess``:
    python -m src.models.rf
All tuning/preprocessing decisions use training CV or the 2022 validation set.
The 2023 test is read only for final evaluation and error analysis.
"""

from pathlib import Path
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.evaluate import CONFIG, evaluate_classification, evaluate_regression, log_experiment
from src.preprocess import ROOT, build_preprocessor, feature_frame, time_series_folds
from src.targets import assign_operational_status

DATA = ROOT / "data" / "processed" / "a2"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
SEED = CONFIG.get("random_state", 42)
CLASSES = ["Healthy", "Warning", "Faulty"]
TRAIN_SAMPLE = 18000
BALANCE_SAMPLE = 30000
VALIDATION_SAMPLE = 15000


def read_splits():
    return {name: pd.read_parquet(DATA / f"{name}.parquet")
            for name in ("train", "val", "test", "test_unseen")}


def sample_rows(df, n):
    if len(df) <= n:
        return df.sort_values(["time", "inverter_id"]).reset_index(drop=True)
    return df.sample(n=n, random_state=SEED).sort_values(
        ["time", "inverter_id"]
    ).reset_index(drop=True)


def make_pipeline(task, params=None, class_weight=None, columns=None):
    params = params or {}
    kwargs = dict(n_estimators=80, max_depth=16, min_samples_leaf=10,
                  max_features=.7, n_jobs=2, random_state=SEED)
    kwargs.update(params)
    if task == "classification":
        model = RandomForestClassifier(class_weight=class_weight, **kwargs)
    elif task == "regression":
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError(task)
    return Pipeline([
        ("preprocess", build_preprocessor(strategy="mean")),
        ("model", model),
    ])


def split_scores(estimator, split, task, selected=None):
    X = feature_frame(split)
    if selected is not None:
        X = X[selected]
    pred = estimator.predict(X)
    if task == "regression":
        result = evaluate_regression(split["health_score_lead4"], pred)
    else:
        result = evaluate_classification(split["status_lead4"], pred)
        class_index = list(estimator.named_steps["model"].classes_).index("Faulty")
        result["faulty_pr_auc"] = round(float(average_precision_score(
            split["status_lead4"].eq("Faulty"),
            estimator.predict_proba(X)[:, class_index],
        )), 4)
    return result


def compare_balancing(train, val):
    """Class weights vs majority undersampling on identical validation rows."""
    train = sample_rows(train, BALANCE_SAMPLE)
    val = sample_rows(val, VALIDATION_SAMPLE)
    X_val = feature_frame(val)
    candidates = {
        "class_weight": train,
        "undersampling": undersample_healthy(train),
    }
    rows = []
    for strategy, data in candidates.items():
        model = make_pipeline("classification", class_weight=(
            "balanced" if strategy == "class_weight" else None
        ))
        model.fit(feature_frame(data), data["status_lead4"])
        pred = model.predict(X_val)
        metrics = evaluate_classification(val["status_lead4"], pred)
        cls_index = list(model.named_steps["model"].classes_).index("Faulty")
        ap = average_precision_score(val["status_lead4"].eq("Faulty"),
                                     model.predict_proba(X_val)[:, cls_index])
        rows.append({"strategy": strategy, "train_rows": len(data),
                     "val_rows": len(val), "macro_f1": metrics["macro_f1"],
                     "faulty_recall": metrics["faulty_recall"],
                     "faulty_precision": metrics["faulty_precision"],
                     "faulty_pr_auc": round(float(ap), 4),
                     "confusion_matrix": json.dumps(metrics["confusion_matrix"])})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "a2_balancing.csv", index=False)
    print("Balancing comparison:\n" + result.to_string(index=False), flush=True)
    # The project names missed Faulty cases as its primary operational risk.
    # Preserve both validation metrics for the report; use recall to select.
    return result.sort_values(["faulty_recall", "macro_f1"],
                              ascending=False).iloc[0]["strategy"]


def undersample_healthy(train):
    minority = train[train["status_lead4"].ne("Healthy")]
    majority = train[train["status_lead4"].eq("Healthy")]
    # Keep two Healthy rows per non-Healthy row, maintaining enough history
    # from the existing feature construction. This is training-only sampling.
    keep = min(len(majority), max(1, 2 * len(minority)))
    sampled = majority.sample(n=keep, random_state=SEED)
    return pd.concat([sampled, minority]).sort_values(
        ["time", "inverter_id"]
    ).reset_index(drop=True)


def tune(train, task, class_weight):
    train = sample_rows(train, TRAIN_SAMPLE)
    folds = list(time_series_folds(train))
    search = RandomizedSearchCV(
        make_pipeline(task, class_weight=class_weight),
        {
            "model__n_estimators": [60, 100],
            "model__max_depth": [12, 18, None],
            "model__min_samples_leaf": [5, 15, 30],
            "model__max_features": ["sqrt", .7],
        },
        n_iter=4, cv=folds, n_jobs=2, refit=True, random_state=SEED,
        scoring=("neg_root_mean_squared_error" if task == "regression" else "f1_macro"),
        error_score="raise",
    )
    y = train["health_score_lead4" if task == "regression" else "status_lead4"]
    search.fit(feature_frame(train), y)
    print(f"{task} tuning: {search.best_params_} CV score {search.best_score_:.4f}", flush=True)
    return {key.removeprefix("model__"): value for key, value in search.best_params_.items()}, search.best_score_


def evaluate_models(models, splits):
    rows = []
    for label, (model, task) in models.items():
        for name, part in splits.items():
            scores = split_scores(model, part, task)
            rows.append({"model": label, "task": task, "split": name,
                         "n": len(part), **scores,
                         "confusion_matrix": json.dumps(scores.get("confusion_matrix"))})
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "a2_rf_results.csv", index=False)
    return result


def exact_baselines(splits):
    """A1-style persistence/current-status/majority on identical A2 rows."""
    majority = splits["train"]["status_lead4"].value_counts().idxmax()
    rows = []
    for name, part in splits.items():
        reg = evaluate_regression(part["health_score_lead4"], part["health_score"])
        rows.append({"model": "Persistence", "task": "regression", "split": name,
                     "n": len(part), **reg})
        for label, pred in [
            ("Current status rule", assign_operational_status(part["health_score"])),
            ("Majority class", np.full(len(part), majority)),
        ]:
            metrics = evaluate_classification(part["status_lead4"], pred)
            rows.append({"model": label, "task": "classification", "split": name,
                         "n": len(part), **metrics,
                         "confusion_matrix": json.dumps(metrics["confusion_matrix"])})
    pd.DataFrame(rows).to_csv(TABLES / "a2_exact_baselines.csv", index=False)


def compare_selection(train, val):
    """Validate the correlation-pruned subset with the same fixed RF model."""
    removed = pd.read_csv(TABLES / "a2_feature_pruning.csv")["dropped_feature"].tolist()
    tr = sample_rows(train, 25000)
    va = sample_rows(val, 12000)
    rows = []
    # Use a small dedicated allowlist preprocessor to avoid introducing absent
    # columns. This comparison is explanatory; the final forest retains all
    # features unless the reduced set wins on both tasks.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline as SKPipeline
    for version in ("all", "pruned"):
        cols = list(feature_frame(tr).columns)
        if version == "pruned":
            cols = [c for c in cols if c not in removed]
        numeric = [c for c in cols if c != "azimuth"]
        transformer = ColumnTransformer([
            ("num", SimpleImputer(strategy="mean", keep_empty_features=True), numeric),
            ("cat", SKPipeline([
                ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ["azimuth"]),
        ])
        for task in ("regression", "classification"):
            estimator = (RandomForestRegressor if task == "regression" else RandomForestClassifier)(
                n_estimators=40, max_depth=12, min_samples_leaf=10,
                random_state=SEED, n_jobs=2,
                **({"class_weight": "balanced"} if task == "classification" else {}),
            )
            pipe = Pipeline([("preprocess", transformer), ("model", estimator)])
            target = "health_score_lead4" if task == "regression" else "status_lead4"
            pipe.fit(feature_frame(tr)[cols], tr[target])
            pred = pipe.predict(feature_frame(va)[cols])
            metric = (evaluate_regression(va[target], pred)["rmse"] if task == "regression"
                      else evaluate_classification(va[target], pred)["macro_f1"])
            rows.append({"version": version, "task": task, "features": len(cols),
                         "metric": "rmse" if task == "regression" else "macro_f1",
                         "validation_score": metric})
    pd.DataFrame(rows).to_csv(TABLES / "a2_selection_validation.csv", index=False)


def explanations(reg, cls, val, test):
    val_sample = sample_rows(val, 2500)
    rows = []
    for name, model, task in [("regression", reg, "regression"),
                               ("classification", cls, "classification")]:
        X = feature_frame(val_sample)
        target = val_sample["health_score_lead4" if task == "regression" else "status_lead4"]
        importance = permutation_importance(
            model, X, target, n_repeats=3,
            scoring="neg_root_mean_squared_error" if task == "regression" else "f1_macro",
            random_state=SEED, n_jobs=2,
        )
        rows.extend({"task": name, "feature": feature,
                     "importance": float(value), "std": float(std)}
                    for feature, value, std in zip(X.columns,
                                                   importance.importances_mean,
                                                   importance.importances_std))
    pd.DataFrame(rows).sort_values(["task", "importance"], ascending=[True, False]).to_csv(
        TABLES / "a2_permutation_importance.csv", index=False
    )

    # TreeSHAP acts on the fitted numeric matrix, including one-hot columns.
    import shap
    shap_rows = []
    local_rows = []
    for task, model in [("regression", reg), ("classification", cls)]:
        X = model.named_steps["preprocess"].transform(feature_frame(val_sample.head(80)))
        names = model.named_steps["preprocess"].get_feature_names_out()
        explainer = shap.TreeExplainer(model.named_steps["model"])
        values = np.asarray(explainer.shap_values(X, check_additivity=False))
        if values.ndim == 3:
            classes = list(model.named_steps["model"].classes_)
            values = values[:, :, classes.index("Faulty")]
        magnitudes = np.abs(values).mean(axis=0)
        shap_rows.extend({"task": task, "transformed_feature": feature,
                          "mean_abs_shap": float(mag)}
                         for feature, mag in zip(names, magnitudes))
        for sample_index in range(min(3, len(values))):
            top = np.argsort(np.abs(values[sample_index]))[-5:][::-1]
            local_rows.extend({"task": task, "validation_row": sample_index,
                               "feature": names[i], "shap_value": float(values[sample_index, i])}
                              for i in top)
    shap_df = pd.DataFrame(shap_rows).sort_values(["task", "mean_abs_shap"], ascending=[True, False])
    shap_df.to_csv(TABLES / "a2_shap_global.csv", index=False)
    pd.DataFrame(local_rows).to_csv(TABLES / "a2_shap_local.csv", index=False)
    for task, subset in shap_df.groupby("task"):
        top = subset.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top.transformed_feature, top.mean_abs_shap)
        ax.set(xlabel="Mean absolute SHAP value", title=f"A2 Random Forest {task} explanation")
        fig.tight_layout()
        fig.savefig(FIGURES / f"a2_shap_{task}.png", dpi=150)
        plt.close(fig)

    # Inspect the worst actual 2023 forecast errors and class disagreements.
    observed = sample_rows(test, 30000).copy()
    observed["predicted_score"] = reg.predict(feature_frame(observed))
    observed["absolute_error"] = np.abs(observed.health_score_lead4 - observed.predicted_score)
    observed["predicted_status"] = cls.predict(feature_frame(observed))
    columns = ["inverter_id", "time", "target_time", "irradiance_wm2", "temp_c",
               "health_score", "health_score_lead4", "predicted_score",
               "status_lead4", "predicted_status", "absolute_error"]
    observed.nlargest(5, "absolute_error")[columns].to_csv(
        TABLES / "a2_rf_failure_cases.csv", index=False
    )


def run(resume=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    splits = read_splits()
    print("Loaded A2 forecast partitions", {k: len(v) for k, v in splits.items()}, flush=True)
    if resume:
        prior = json.loads((DATA / "rf_manifest.json").read_text(encoding="utf-8"))
        balance = prior["balancing"]
        reg_params, cls_params = prior["regression_params"], prior["classification_params"]
        reg_cv, cls_cv = prior["regression_cv_score"], prior["classification_cv_score"]
        models = {}
        for label in ("RF default regression", "RF tuned regression",
                      "RF default classification", "RF tuned classification"):
            task = "regression" if "regression" in label else "classification"
            models[label] = (joblib.load(DATA / (label.lower().replace(" ", "_") + ".joblib")), task)
        result = pd.read_csv(TABLES / "a2_rf_results.csv")
        cls_train_rows = prior["classifier_fit_rows"]
    else:
        exact_baselines(splits)
        compare_selection(splits["train"], splits["val"])
        balance = compare_balancing(splits["train"], splits["val"])
        cls_train = (splits["train"] if balance == "class_weight"
                     else undersample_healthy(splits["train"]))
        cls_train_rows = len(cls_train)
        weight = "balanced" if balance == "class_weight" else None
        reg_params, reg_cv = tune(splits["train"], "regression", None)
        cls_params, cls_cv = tune(cls_train, "classification", weight)
        models = {}
        for label, task, data, params, class_weight in [
            ("RF default regression", "regression", splits["train"], {}, None),
            ("RF tuned regression", "regression", splits["train"], reg_params, None),
            ("RF default classification", "classification", cls_train, {}, weight),
            ("RF tuned classification", "classification", cls_train, cls_params, weight),
        ]:
            model = make_pipeline(task, params=params, class_weight=class_weight)
            target = "health_score_lead4" if task == "regression" else "status_lead4"
            print(f"Fitting {label} on {len(data):,} rows", flush=True)
            model.fit(feature_frame(data), data[target])
            models[label] = model, task
            joblib.dump(model, DATA / (label.lower().replace(" ", "_") + ".joblib"))
        result = evaluate_models(models, splits)
    print(result[["model", "split", "rmse", "macro_f1", "faulty_recall"]].to_string(index=False), flush=True)
    validation = result[result["split"].eq("val")].set_index("model")
    reg_name = min(("RF default regression", "RF tuned regression"),
                   key=lambda name: validation.loc[name, "rmse"])
    cls_name = max(("RF default classification", "RF tuned classification"),
                   key=lambda name: validation.loc[name, "faulty_recall"])
    print(f"Selected from validation: {reg_name}; {cls_name}", flush=True)
    reg = models[reg_name][0]
    cls = models[cls_name][0]
    explanations(reg, cls, splits["val"], splits["test"])
    for name, task, score in [(reg_name, "regression", "rmse"),
                               (cls_name, "classification", "macro_f1")]:
        subset = result[result.model.eq(name)].set_index("split")
        log_experiment("A2", name, task,
                       subset.loc["train", score], subset.loc["val", score],
                       subset.loc["test", score], subset.loc["test_unseen", score],
                       notes=f"one-hour future target; balancing={balance}; mean imputation")
    manifest = {"balancing": balance, "regression_params": reg_params,
                "classification_params": cls_params,
                "regression_cv_score": reg_cv, "classification_cv_score": cls_cv,
                "train_rows": len(splits["train"]), "classifier_fit_rows": cls_train_rows,
                "tuning_sample_size": TRAIN_SAMPLE,
                "selected_regression": reg_name,
                "selected_classification": cls_name,
                "selection": "all features; validation comparison recorded separately",
                "shap_rows": 80, "permutation_rows": 2500}
    (DATA / "rf_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("A2 RF analysis complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Use saved models/results to regenerate selected-model explanations")
    run(resume=parser.parse_args().resume)

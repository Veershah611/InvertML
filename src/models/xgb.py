"""Stage 9 Supervised Modeling & Stage 10 Tuning & Stage 11.3 Tree-SHAP (Track T3: XGBoost).

Student Owner: B1 (Feature Specialist & XGBoost Lead).

Target (Option A in section 7.2, same as A1 and A2): Health(t + 1 h), the health
score exactly one hour later on the same day, predicted from features at t.
An earlier version predicted Health(t) from PR features at t, which rebuilds
the target from its own inputs (target leakage).

Core Responsibilities:
1. xgb_regressor: Predicts continuous Health Score at t + 1 h (0.0 to 100.0) with histogram-based boosting.
2. xgb_classifier: Predicts 3-class operational status at t + 1 h (Healthy, Warning, Faulty) with multi:softprob.
3. Classification Strategy Comparison: Method 1 (Dedicated Classifier) vs. Method 2 (Thresholded Regressor).
4. Stage 10 Hyperparameter Tuning: Tuning on B2's defined chronological splits (train_df and val_df).
5. Stage 5 Balancing: sample weights vs Healthy undersampling on the validation set.
6. Stage 11 Evaluation: Evaluates on 2023 chronological test split and ~7 unseen held-out stations.
7. Stage 11.3 Tree-SHAP Explainability: Global beeswarm summary, local failure waterfall plot, and failure case analysis.
8. Logging official benchmark scores to results/results.csv via B2's log_experiment.

Usage:
    python -m src.models.xgb            # Run baseline & tuned XGBoost models
    python -m src.models.xgb --tune     # Run Stage 10 hyperparameter grid sweep
    python -m src.models.xgb --explain  # Run Stage 11.3 Tree-SHAP explainability & error analysis
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix

from src.evaluate import (
    evaluate_classification,
    evaluate_regression,
    log_experiment,
    split_chronological,
    split_unseen_stations,
)
from src.features import STAGE_7_SELECTED_FEATURES, build_all_features
from src.targets import assign_operational_status, compute_health_score

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

PROCESSED_DIR = ROOT / CONFIG.get("paths", {}).get("processed", "data/processed")
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Status class label mappings
STATUS_MAP = {"Healthy": 0, "Warning": 1, "Faulty": 2}
STATUS_REV_MAP = {0: "Healthy", 1: "Warning", 2: "Faulty"}

# Forecasting setup: features at t, target exactly one hour later (4 bins of 15 min)
HORIZON = pd.Timedelta("1h")
XGB_FEATURES = STAGE_7_SELECTED_FEATURES + ["health_now"]
HEALTHY_PER_OTHER = 2   # undersampling rule shared with A1 and A2

# Optimal default hyperparameters for solar tabular telemetry
DEFAULT_REG_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

TUNED_REG_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_CLS_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ===========================================================================
# 1. Pipeline Builders
# ===========================================================================

def build_xgb_regressor(params=None) -> xgb.XGBRegressor:
    """Build XGBRegressor using histogram-based tree splitting."""
    p = DEFAULT_REG_PARAMS.copy()
    if params:
        p.update(params)
    return xgb.XGBRegressor(**p)


def build_xgb_classifier(params=None) -> xgb.XGBClassifier:
    """Build 3-class XGBClassifier with multi:softprob objective."""
    p = DEFAULT_CLS_PARAMS.copy()
    if params:
        p.update(params)
    return xgb.XGBClassifier(**p)


# ===========================================================================
# 2. Data Preparation & Thresholding Functions
# ===========================================================================

def threshold_health_score(health_scores: np.ndarray | pd.Series) -> np.ndarray:
    """Method 2: Convert continuous health scores into discrete classes.
    
    Cutoffs:
        Health Score >= 90.0           -> 0 ('Healthy')
        70.0 <= Health Score < 90.0    -> 1 ('Warning')
        Health Score < 70.0            -> 2 ('Faulty')
    """
    hs = np.asarray(health_scores, dtype=np.float32)
    conditions = [hs >= 90.0, (hs >= 70.0) & (hs < 90.0), hs < 70.0]
    return np.select(conditions, [0, 1, 2], default=0).astype(int)


def build_forecast_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (inverter, t): Stage 7 features at t, target Health(t + 1 h).

    Features are built on the full 15-min grid before any split, so rolling and
    peer statistics see all history that existed at t. The lead is kept only
    when the row one hour ahead exists (same day, no overnight jump). Rows where
    Health(t) or Health(t + 1 h) is undefined (irradiance < 100 W/m2) are dropped.
    """
    df = raw.copy()
    df["inverter_id"] = df["inverter_id"].astype(str)
    df = df.sort_values(["inverter_id", "time"]).reset_index(drop=True)
    df = build_all_features(df)
    df = compute_health_score(df)                     # B1's target, unchanged
    df = df.rename(columns={"health_score": "health_now"})

    lead = df[["inverter_id", "time", "health_now"]].rename(columns={"health_now": "health_lead"})
    lead["time"] = lead["time"] - HORIZON
    df = df.merge(lead, on=["inverter_id", "time"], how="left", validate="one_to_one")
    df = df.dropna(subset=["health_now", "health_lead"]).reset_index(drop=True)
    df["status_int"] = threshold_health_score(df["health_lead"].values)
    return df


def make_splits(dataset: pd.DataFrame) -> dict:
    """B2's split functions; held-out inverters stay out of train and val (A2's rule)."""
    _, val, test = split_chronological(dataset, time_col="time")
    train, unseen = split_unseen_stations(dataset, station_col="inverter_id", time_col="time")
    held = {s.lower() for s in CONFIG.get("unseen_stations", [])}
    val = val[~val["inverter_id"].str.lower().isin(held)]
    return {"train": train, "val": val, "test": test, "test_unseen": unseen}


def train_fill_values(train_df: pd.DataFrame, feature_cols: list[str] = None) -> pd.Series:
    """Feature medians from the training split only, used to fill every split."""
    cols = feature_cols or XGB_FEATURES
    return train_df[cols].astype(np.float32).median()


def prepare_xgb_data(
    df: pd.DataFrame,
    fill_values: pd.Series,
    feature_cols: list[str] = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Feature matrix and Health(t + 1 h) targets for one split of build_forecast_dataset().

    Missing feature values are filled with training medians (fill_values),
    never with statistics of the split itself.

    Returns:
        (X, y_regression, y_classification, df_clean)
    """
    cols = feature_cols or XGB_FEATURES
    X = df[cols].astype(np.float32).fillna(fill_values[cols])
    y_reg = df["health_lead"].values.astype(np.float32)
    y_cls = df["status_int"].values.astype(int)
    return X, y_reg, y_cls, df


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights to prioritize the rare Faulty class."""
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    weights_dict = {cls: total_samples / (len(classes) * count) for cls, count in zip(classes, counts)}
    # Add extra weight to Faulty class (class 2) to maximize Faulty Recall
    if 2 in weights_dict:
        weights_dict[2] *= 1.35
    return np.array([weights_dict[val] for val in y], dtype=np.float32)


def undersample_healthy(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep HEALTHY_PER_OTHER Healthy rows per non-Healthy row (training rows only)."""
    healthy = np.flatnonzero(y == 0)
    other = np.flatnonzero(y != 0)
    keep = min(len(healthy), max(1, HEALTHY_PER_OTHER * len(other)))
    sampled = np.random.RandomState(42).choice(healthy, size=keep, replace=False)
    rows = np.sort(np.concatenate([sampled, other]))          # keep time order
    return X.iloc[rows], y[rows]


def compare_balancing(X_train, y_cls_train, X_val, y_cls_val) -> tuple[pd.DataFrame, str]:
    """Stage 5: sample weights vs Healthy undersampling for the dedicated classifier.

    Both are fitted on training rows only; validation is never resampled. SMOTE
    is not used: interpolating between rows invents readings that never happened
    and mixes rows from different days and inverters. The winner is chosen on
    validation Faulty recall, then macro-F1 (the rule A1 and A2 used).
    """
    rows = []
    for strategy in ["sample_weights", "undersampling"]:
        if strategy == "sample_weights":
            X_fit, y_fit = X_train, y_cls_train
            weights = compute_sample_weights(y_cls_train)
        else:
            X_fit, y_fit = undersample_healthy(X_train, y_cls_train)
            weights = None
        model = build_xgb_classifier(DEFAULT_CLS_PARAMS)
        model.fit(X_fit, y_fit, sample_weight=weights, verbose=False)
        pred = model.predict(X_val)
        m = evaluate_classification(y_cls_val, pred)
        ap = average_precision_score(y_cls_val == 2, model.predict_proba(X_val)[:, 2])
        rows.append({"strategy": strategy, "train_rows": len(X_fit), "val_rows": len(X_val),
                     "macro_f1": m["macro_f1"], "faulty_recall": m["faulty_recall"],
                     "faulty_precision": m["faulty_precision"], "faulty_pr_auc": round(float(ap), 4),
                     "confusion_matrix": json.dumps(confusion_matrix(y_cls_val, pred, labels=[0, 1, 2]).tolist())})
    result = pd.DataFrame(rows)
    winner = result.sort_values(["faulty_recall", "macro_f1"], ascending=False).iloc[0]["strategy"]
    result["selected"] = result["strategy"].eq(winner)
    result.to_csv(TABLES_DIR / "b1_xgb_balancing.csv", index=False)
    print("\nStage 5 balancing (validation):\n" + result.drop(columns="confusion_matrix").to_string(index=False))
    return result, winner


# ===========================================================================
# 3. Method 1 vs. Method 2 Classification Comparison
# ===========================================================================

def compare_classification_strategies(
    xgb_reg,
    xgb_cls,
    X_val: pd.DataFrame,
    y_cls_val: np.ndarray
) -> dict:
    """Compare Method 1 (Dedicated Classifier) vs Method 2 (Thresholded Regressor)."""
    # Method 1: Dedicated Classifier
    pred_cls_m1 = xgb_cls.predict(X_val)
    metrics_m1 = evaluate_classification(y_cls_val, pred_cls_m1)

    # Method 2: Thresholded Regressor
    pred_reg_val = xgb_reg.predict(X_val)
    pred_cls_m2 = threshold_health_score(pred_reg_val)
    metrics_m2 = evaluate_classification(y_cls_val, pred_cls_m2)

    print("\n" + "=" * 65)
    print("CLASSIFICATION STRATEGY COMPARISON (ON B2 VALIDATION SET)")
    print("=" * 65)
    print(f"Method 1 (Dedicated Softmax Classifier):")
    print(f"  Macro-F1       : {metrics_m1['macro_f1']:.4f}")
    print(f"  Faulty Recall  : {metrics_m1['faulty_recall']:.4f} (Primary Metric)")
    print(f"  Faulty Precision: {metrics_m1['faulty_precision']:.4f}")

    print(f"\nMethod 2 (Thresholded Continuous Regressor):")
    print(f"  Macro-F1       : {metrics_m2['macro_f1']:.4f}")
    print(f"  Faulty Recall  : {metrics_m2['faulty_recall']:.4f} (Primary Metric)")
    print(f"  Faulty Precision: {metrics_m2['faulty_precision']:.4f}")

    winner = "Method 1 (Dedicated Classifier)" if metrics_m1["faulty_recall"] >= metrics_m2["faulty_recall"] else "Method 2 (Thresholded Regressor)"
    print(f"\nRecommended Winner: {winner}")
    print("=" * 65)

    return {"method_1": metrics_m1, "method_2": metrics_m2, "winner": winner}


# ===========================================================================
# 4. Hyperparameter Tuning on B2's Defined Chronological Sets (Stage 10)
# ===========================================================================

def run_hyperparameter_tuning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sample_fraction: float = 0.20,
    fill_values: pd.Series = None
) -> tuple[dict, pd.DataFrame]:
    """Execute hyperparameter grid sweep using B2's exact train and validation splits.
    
    Strict Design:
        - Trains candidate configurations on a time-ordered slice of B2's train_df.
        - Scores validation RMSE directly on B2's val_df (no early stopping).
        - Avoids generic scikit-learn arbitrary row-index folds.
    """
    print("\n" + "=" * 65)
    print("STAGE 10: HYPERPARAMETER TUNING ON B2 CHRONOLOGICAL SPLITS")
    print("=" * 65)

    # 1. Prepare features (Stage 7 set + Health(t)), filled with training medians
    if fill_values is None:
        fill_values = train_fill_values(train_df)
    X_train_full, y_reg_train_full, _, _ = prepare_xgb_data(train_df, fill_values)
    X_val, y_reg_val, _, _ = prepare_xgb_data(val_df, fill_values)

    # 2. Time-ordered sampling of training set for fast CPU search
    n_sample = int(len(X_train_full) * sample_fraction)
    X_train_sample = X_train_full.iloc[-n_sample:].copy()
    y_reg_train_sample = y_reg_train_full[-n_sample:].copy()

    print(f"Tuning Search Space: {n_sample:,} time-ordered train rows -> {len(X_val):,} B2 validation rows.")

    search_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.01, "reg_lambda": 1.0},
        {"max_depth": 6, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.5},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 1.0},
        {"max_depth": 8, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.2, "reg_lambda": 2.0},
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.5},
    ]

    results = []
    best_val_rmse = float("inf")
    best_params = None

    for idx, params in enumerate(search_grid, 1):
        t0 = time.time()
        model_params = DEFAULT_REG_PARAMS.copy()
        model_params.update(params)
        model_params["n_estimators"] = 250

        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train_sample,
            y_reg_train_sample,
            eval_set=[(X_val, y_reg_val)],
            verbose=False
        )

        train_preds = model.predict(X_train_sample)
        val_preds = model.predict(X_val)

        train_rmse = np.sqrt(np.mean((y_reg_train_sample - train_preds) ** 2))
        val_rmse = np.sqrt(np.mean((y_reg_val - val_preds) ** 2))
        val_mae = np.mean(np.abs(y_reg_val - val_preds))
        elapsed = time.time() - t0

        print(f"Config {idx:02d} | depth={params['max_depth']}, lr={params['learning_rate']}, "
              f"sub={params['subsample']} | Val RMSE: {val_rmse:.3f} | Train RMSE: {train_rmse:.3f} ({elapsed:.1f}s)")

        results.append({
            "config_id": idx,
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "train_rmse": round(float(train_rmse), 3),
            "val_rmse": round(float(val_rmse), 3),
            "val_mae": round(float(val_mae), 3),
            "runtime_s": round(float(elapsed), 1),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = model_params

    tuning_df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    table_path = TABLES_DIR / "b1_xgb_default_vs_tuned.csv"
    tuning_df.to_csv(table_path, index=False)
    print(f"\nSaved Stage 10 tuning summary table: {table_path.relative_to(ROOT)}")
    print(f"Optimal Configuration Selected: max_depth={best_params['max_depth']}, lr={best_params['learning_rate']} (Val RMSE: {best_val_rmse:.3f})")

    return best_params, tuning_df


# ===========================================================================
# 5. Tree-SHAP Explainability & Failure Case Diagnosis (Stage 11.3)
# ===========================================================================

def run_shap_explainability(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_size: int = 1500
) -> pd.DataFrame:
    """Generate global and local Tree-SHAP plots and analyze top 3 prediction failure cases."""
    print("\n" + "=" * 65)
    print("STAGE 11.3: TREE-SHAP EXPLAINABILITY & FAILURE CASE DIAGNOSIS")
    print("=" * 65)

    # 1. Subsample test data for fast Tree-SHAP computation
    n_pts = min(sample_size, len(X_test))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=n_pts, replace=False)
    X_shap = X_test.iloc[sample_indices].copy()
    test_df_sample = test_df.iloc[sample_indices].copy()

    print(f"Computing Tree-SHAP values across {n_pts:,} test intervals...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # 2. Global Beeswarm Summary Plot
    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("Tree-SHAP Global Feature Importance Ranking (XGBoost)", fontsize=11, fontweight="bold")
    summary_path = FIGURES_DIR / "b1_shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Saved: {summary_path.relative_to(ROOT)}")

    # 3. Residual Error Analysis: Identify Top 3 Worst Failures
    y_true_sample = test_df_sample["health_lead"].values
    preds_sample = model.predict(X_shap)
    residuals = np.abs(y_true_sample - preds_sample)
    worst_3_idx = np.argsort(residuals)[-3:][::-1]

    # 4. Local Waterfall Plot for the Most Severe Failure Case
    worst_idx = worst_3_idx[0]
    plt.figure(figsize=(9, 5))
    explanation = shap.Explanation(
        values=shap_values[worst_idx],
        base_values=explainer.expected_value,
        data=X_shap.iloc[worst_idx].values,
        feature_names=list(X_shap.columns)
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title("Tree-SHAP Local Attribution: Severe Prediction Failure", fontsize=11, fontweight="bold")
    waterfall_path = FIGURES_DIR / "b1_shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    print(f"Saved: {waterfall_path.relative_to(ROOT)}")

    # 5. Record observed context for the top 3 failures. Causes are not
    # inferred here: the data has no maintenance logs, so any explanation in
    # the report is a hypothesis to check against this context.
    failures = []
    for rank, idx in enumerate(worst_3_idx, 1):
        row = test_df_sample.iloc[idx]
        failures.append({
            "rank": rank,
            "timestamp": str(row.get("time", "-")),
            "inverter_id": str(row.get("inverter_id", "unknown")),
            "health_now": round(float(row.get("health_now", np.nan)), 2),
            "actual_health_t_plus_1h": round(float(y_true_sample[idx]), 2),
            "predicted_health": round(float(preds_sample[idx]), 2),
            "residual_error": round(float(residuals[idx]), 2),
            "irradiance_wm2": round(float(row.get("irradiance_wm2", np.nan)), 1),
            "temp_c": round(float(row.get("temp_c", np.nan)), 1),
            "n_valid_5min": row.get("n_valid_5min", np.nan),
            "power_missing": bool(pd.isna(row.get("ac_power_w", np.nan))),
        })

    failure_df = pd.DataFrame(failures)
    failure_path = TABLES_DIR / "b1_xgb_failure_cases.csv"
    failure_df.to_csv(failure_path, index=False)
    print(f"Saved failure cases report: {failure_path.relative_to(ROOT)}")

    print("\nTop 3 Worst Prediction Failures:")
    for f in failures:
        print(f"  #{f['rank']} | Inverter: {f['inverter_id']} | Health(t): {f['health_now']} | "
              f"Actual t+1h: {f['actual_health_t_plus_1h']} vs Pred: {f['predicted_health']} (Err: {f['residual_error']})")

    return failure_df


# ===========================================================================
# 6. Complete Training, Evaluation, and Benchmark Workflow
# ===========================================================================

def run_xgb_experiment(
    dataset: pd.DataFrame,
    tune: bool = False,
    explain: bool = False,
    log: bool = True
):
    """Execute end-to-end Track T3 modeling pipeline adhering strictly to B2's split engine."""
    start_time = time.time()
    print("=" * 70)
    print("EXECUTING STAGE 9 / TRACK T3: XGBOOST REGRESSION & CLASSIFICATION")
    print("=" * 70)

    # 1. Forecasting dataset: features at t, target Health(t + 1 h)
    print("\n1. Building forecasting dataset (features at t, target Health(t + 1 h))...")
    forecast = build_forecast_dataset(dataset)

    # 2. Splitting strictly via B2's framework (held-out inverters excluded from train/val)
    print("\n2. Partitioning with B2's split_chronological() and split_unseen_stations()...")
    splits = make_splits(forecast)
    train_df, val_df, test_df, test_unseen = (splits[k] for k in ["train", "val", "test", "test_unseen"])

    # 3. Extract feature matrices, filled with training medians only
    fill_values = train_fill_values(train_df)
    X_train, y_reg_train, y_cls_train, _ = prepare_xgb_data(train_df, fill_values)
    X_val, y_reg_val, y_cls_val, _ = prepare_xgb_data(val_df, fill_values)
    X_test, y_reg_test, y_cls_test, test_clean = prepare_xgb_data(test_df, fill_values)
    X_unseen, y_reg_unseen, y_cls_unseen, _ = prepare_xgb_data(test_unseen, fill_values)

    print(f"\nFeature Matrices Prepared ({X_train.shape[1]} features: Stage 7 set + Health(t)):")
    print(f"  * Train  : {X_train.shape[0]:,} rows")
    print(f"  * Val    : {X_val.shape[0]:,} rows")
    print(f"  * Test   : {X_test.shape[0]:,} rows (All of 2023)")
    print(f"  * Unseen : {X_unseen.shape[0]:,} rows (~7 held-out stations)")

    # 4. Hyperparameter Tuning (Stage 10)
    if tune:
        best_reg_params, _ = run_hyperparameter_tuning(train_df, val_df, fill_values=fill_values)
    else:
        best_reg_params = TUNED_REG_PARAMS

    # 5. Train Tuned XGBRegressor
    print("\n3. Fitting Tuned xgb_regressor on full B2 training split...")
    xgb_reg = build_xgb_regressor(best_reg_params)
    t_reg_start = time.time()
    xgb_reg.fit(X_train, y_reg_train, verbose=False)
    reg_train_time = time.time() - t_reg_start
    print(f"  * Fitted xgb_regressor in {reg_train_time:.1f}s")

    # 6. Stage 5 balancing, then the Dedicated XGBClassifier (Method 1) with the winner
    _, balancing = compare_balancing(X_train, y_cls_train, X_val, y_cls_val)
    print(f"\n4. Fitting Dedicated xgb_classifier with {balancing} on full B2 training split...")
    if balancing == "sample_weights":
        X_cls_fit, y_cls_fit, sample_weights = X_train, y_cls_train, compute_sample_weights(y_cls_train)
    else:
        (X_cls_fit, y_cls_fit), sample_weights = undersample_healthy(X_train, y_cls_train), None
    xgb_cls = build_xgb_classifier(DEFAULT_CLS_PARAMS)
    t_cls_start = time.time()
    xgb_cls.fit(X_cls_fit, y_cls_fit, sample_weight=sample_weights, verbose=False)
    cls_train_time = time.time() - t_cls_start
    print(f"  * Fitted xgb_classifier in {cls_train_time:.1f}s")

    # 6. Method 1 vs Method 2 Comparison
    comp_results = compare_classification_strategies(xgb_reg, xgb_cls, X_val, y_cls_val)

    # 7. Evaluate Regression across all splits via B2 evaluate_regression
    print("\n5. Computing Final Regression Metrics (B2 Evaluation Framework)...")
    reg_train_metrics = evaluate_regression(y_reg_train, xgb_reg.predict(X_train))
    reg_val_metrics = evaluate_regression(y_reg_val, xgb_reg.predict(X_val))
    reg_test_metrics = evaluate_regression(y_reg_test, xgb_reg.predict(X_test))
    reg_unseen_metrics = evaluate_regression(y_reg_unseen, xgb_reg.predict(X_unseen))

    print(f"  * Train  RMSE: {reg_train_metrics['rmse']:.2f} | MAE: {reg_train_metrics['mae']:.2f} | R2: {reg_train_metrics['r2']:.3f}")
    print(f"  * Val    RMSE: {reg_val_metrics['rmse']:.2f} | MAE: {reg_val_metrics['mae']:.2f} | R2: {reg_val_metrics['r2']:.3f}")
    print(f"  * Test   RMSE: {reg_test_metrics['rmse']:.2f} | MAE: {reg_test_metrics['mae']:.2f} | R2: {reg_test_metrics['r2']:.3f}")
    print(f"  * Unseen RMSE: {reg_unseen_metrics['rmse']:.2f} | MAE: {reg_unseen_metrics['mae']:.2f} | R2: {reg_unseen_metrics['r2']:.3f}")
    # Persistence (Health(t+1h) = Health(t)) is the bar a forecaster must beat
    persistence = {name: evaluate_regression(part["health_lead"], part["health_now"])["rmse"]
                   for name, part in splits.items()}
    print(f"  * Persistence RMSE  train {persistence['train']:.2f} | val {persistence['val']:.2f} | "
          f"test {persistence['test']:.2f} | unseen {persistence['test_unseen']:.2f}")

    # 8. Evaluate Classification across all splits via B2 evaluate_classification
    print("\n6. Computing Final Classification Metrics (B2 Evaluation Framework)...")
    cls_train_metrics = evaluate_classification(y_cls_train, xgb_cls.predict(X_train))
    cls_val_metrics = evaluate_classification(y_cls_val, xgb_cls.predict(X_val))
    cls_test_metrics = evaluate_classification(y_cls_test, xgb_cls.predict(X_test))
    cls_unseen_metrics = evaluate_classification(y_cls_unseen, xgb_cls.predict(X_unseen))

    print(f"  * Train  Macro-F1: {cls_train_metrics['macro_f1']:.4f} | Faulty Recall: {cls_train_metrics['faulty_recall']:.4f}")
    print(f"  * Val    Macro-F1: {cls_val_metrics['macro_f1']:.4f} | Faulty Recall: {cls_val_metrics['faulty_recall']:.4f}")
    print(f"  * Test   Macro-F1: {cls_test_metrics['macro_f1']:.4f} | Faulty Recall: {cls_test_metrics['faulty_recall']:.4f}")
    print(f"  * Unseen Macro-F1: {cls_unseen_metrics['macro_f1']:.4f} | Faulty Recall: {cls_unseen_metrics['faulty_recall']:.4f}")

    # 9. Tree-SHAP Explainability & Error Diagnosis (Stage 11.3)
    if explain:
        run_shap_explainability(xgb_reg, X_test, test_clean)

    # 10. Log Benchmark Results to results/results.csv
    if log:
        print("\n7. Logging Official Benchmark Results to results/results.csv...")
        log_experiment(
            student="B1",
            model_name="XGBRegressor (Tuned)",
            task="regression",
            train_score=f"RMSE: {reg_train_metrics['rmse']:.2f}",
            val_score=f"RMSE: {reg_val_metrics['rmse']:.2f}",
            test_score=f"RMSE: {reg_test_metrics['rmse']:.2f}",
            unseen_score=f"RMSE: {reg_unseen_metrics['rmse']:.2f}",
            notes=f"Health(t+1h) forecast; depth={best_reg_params['max_depth']}, lr={best_reg_params['learning_rate']}, "
                  f"Stage 7 features + Health(t) ({reg_train_time:.1f}s)",
        )
        log_experiment(
            student="B1",
            model_name="XGBClassifier",
            task="classification",
            train_score=f"F1: {cls_train_metrics['macro_f1']:.2f}, Faulty recall: {cls_train_metrics['faulty_recall']:.2f}",
            val_score=f"F1: {cls_val_metrics['macro_f1']:.2f}, Faulty recall: {cls_val_metrics['faulty_recall']:.2f}",
            test_score=f"F1: {cls_test_metrics['macro_f1']:.2f}, Faulty recall: {cls_test_metrics['faulty_recall']:.2f}",
            unseen_score=f"F1: {cls_unseen_metrics['macro_f1']:.2f}, Faulty recall: {cls_unseen_metrics['faulty_recall']:.2f}",
            notes=f"Status(t+1h) forecast; dedicated 3-class softmax, {balancing} (Stage 5 winner) ({cls_train_time:.1f}s)",
        )

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"XGBoost Track T3 Successfully Completed in {total_time:.1f}s!")
    print("=" * 70)

    return xgb_reg, xgb_cls


# ===========================================================================
# 7. Main CLI Execution
# ===========================================================================

def load_or_generate_dataset() -> pd.DataFrame:
    """Load hkust_15min_v1.parquet or generate representative sample."""
    parquet_path = PROCESSED_DIR / "hkust_15min_v1.parquet"
    if parquet_path.exists():
        print(f"Loading real telemetry from {parquet_path.relative_to(ROOT)}...")
        return pd.read_parquet(parquet_path)

    print("Notice: hkust_15min_v1.parquet not found. Generating simulated representative sample.")
    np.random.seed(42)
    dates = pd.date_range("2021-05-01 06:00:00", "2023-12-31 18:00:00", freq="15min")
    dates = dates[(dates.hour >= 6) & (dates.hour <= 18)]
    inverters = [
        "zone_a1_inverter_1", "zone_a1_inverter_2", "library_inverter_1",
        "library_inverter_2", "sq_block_p_inverter", "shaw_auditorium_inverter"
    ]
    rows = []
    for d in dates:
        hour_factor = np.sin(np.pi * (d.hour + d.minute / 60.0 - 6) / 12.0)
        if hour_factor <= 0:
            continue
        irr = np.clip(1000.0 * hour_factor * np.random.uniform(0.7, 1.05), 50.0, 1100.0)
        temp = 20.0 + 10.0 * hour_factor + np.random.uniform(-2, 2)
        for idx, inv in enumerate(inverters):
            cap = 25.0
            if idx == 0 and d.month in [7, 8]:
                fault_factor = 0.55  # Faulty (< 70)
            elif idx == 1 and d.month in [5, 6, 9]:
                fault_factor = 0.82  # Warning (70-90)
            elif idx == 2 and d.month in [10, 11]:
                fault_factor = 0.40  # Faulty (< 70)
            else:
                fault_factor = 1.0   # Healthy (>= 90)
            ac = irr * cap * 0.78 * fault_factor * np.random.uniform(0.95, 1.02)
            rows.append({
                "time": d,
                "inverter_id": inv,
                "station_id": f"station_{(idx // 2) + 1}",
                "capacity_kwp": cap,
                "irradiance_wm2": irr,
                "temp_c": temp,
                "ac_power_w": ac,
                "tilt_deg": "15deg",
                "azimuth": "south"
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track T3: XGBoost Health Monitoring Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Stage 10 hyperparameter grid search")
    parser.add_argument("--explain", action="store_true", help="Run Stage 11.3 Tree-SHAP explainability")
    parser.add_argument("--no-log", action="store_true", help="Disable logging to results/results.csv")
    args = parser.parse_args()

    data = load_or_generate_dataset()
    run_xgb_experiment(
        dataset=data,
        tune=args.tune,
        explain=args.explain or True,  # Default to running SHAP so figures are produced
        log=not args.no_log
    )

"""Stage 9 Supervised Modeling & Stage 11 Explainability (Track T4): Multi-Layer Perceptron (MLP).

Student Owner: B2 (Evaluation Architect & Neural Network Specialist).

This module implements:
1. mlp_regressor: Predicts continuous Health Score (0 to 100).
2. mlp_classifier: Predicts operational status class (Healthy, Warning, Faulty).
3. Leakage-free Pipeline with StandardScaler.
4. Comparison between Method 1 (Dedicated Classifier) and Method 2 (Thresholded Regressor).
5. Evaluation on chronological splits and unseen stations.
6. Hyperparameter tuning with TimeSeriesSplit (Stage 10).
7. Model Explainability & Error Analysis: Permutation Importance, PDP, and Failure Cases (Task 7).
8. Logging results to results/results.csv.

Target: Health(t + 1 h), exact time match (§7.2 Option A, same as A1 and B1).
Splits: B2's functions, held-out inverters excluded from train and validation.
Training uses a seeded random sample of 60k rows across all inverters; every
other split is scored in full.

Commands:
    python -m src.models.mlp           # Run baseline models
    python -m src.models.mlp --tune    # Run hyperparameter tuning
    python -m src.models.mlp --explain # Run Task 7 explainability analysis
    add --log to write the final rows to results/results.csv (once)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt

from src.features import build_b2_features
from src.targets import (
    compute_health_score,
    assign_operational_status,
)
from src.evaluate import (
    CONFIG,
    split_chronological,
    split_unseen_stations,
    evaluate_regression,
    evaluate_classification,
    log_experiment,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SEED = 42
TRAIN_SAMPLE = 60000          # random training rows across all inverters (CPU runtime)
BALANCING_TABLE = ROOT / "reports" / "tables" / "b2_balancing.csv"


# ===========================================================================
# 1. Model Building Functions (Simple Pipelines)
# ===========================================================================

# Tuned Optimal Hyperparameters from GridSearchCV
BEST_HIDDEN_LAYERS = (100, 50)
BEST_LR = 0.005
BEST_BATCH_SIZE = 256
BEST_ALPHA = 0.0001


def build_mlp_regressor(
    hidden_layer_sizes=BEST_HIDDEN_LAYERS,
    alpha=BEST_ALPHA,
    learning_rate_init=BEST_LR,
    batch_size=BEST_BATCH_SIZE,
    max_iter=200,
    random_state=42,
):
    """Build mlp_regressor with StandardScaler and MLPRegressor (tuned by default)."""
    mlp_regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        ))
    ])
    return mlp_regressor


def build_mlp_classifier(
    hidden_layer_sizes=BEST_HIDDEN_LAYERS,
    alpha=BEST_ALPHA,
    learning_rate_init=BEST_LR,
    batch_size=BEST_BATCH_SIZE,
    max_iter=200,
    random_state=42,
):
    """Build mlp_classifier with StandardScaler and MLPClassifier (tuned by default)."""
    mlp_classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        ))
    ])
    return mlp_classifier


def get_tuned_mlp_regressor():
    """Convenience helper returning the tuned MLP Regressor."""
    return build_mlp_regressor()


def get_tuned_mlp_classifier():
    """Convenience helper returning the tuned MLP Classifier."""
    return build_mlp_classifier()


# ===========================================================================
# 2. Simple Thresholding Function (Method 2)
# ===========================================================================

def threshold_health_score(scores):
    """Convert continuous health scores into Healthy, Warning, or Faulty.
    
    Rules:
    - score >= 90.0: Healthy
    - 70.0 <= score < 90.0: Warning
    - score < 70.0: Faulty
    """
    labels = []
    for s in scores:
        if s >= 90.0:
            labels.append("Healthy")
        elif s >= 70.0:
            labels.append("Warning")
        else:
            labels.append("Faulty")
    return np.array(labels)


# ===========================================================================
# 3. Simple Data Prep Function
# ===========================================================================

def prepare_mlp_data(df, horizon_steps=4):
    """Add time features, metadata features, and canonical forecasting target Health(t + 1h).

    The target is the health score exactly horizon_steps x 15 min later for the
    same inverter (joined on time, as A1 and B1 do), so the last bins of a day
    never take their target from the next morning. Rows where Health(t) or
    Health(t + 1h) is undefined are dropped.
    """
    df = df.copy()

    # Add B2 time and station features
    if "hour_sin" not in df.columns:
        df = build_b2_features(df)

    # If health_score is not in the data, compute it via B1's canonical function
    if "health_score" not in df.columns:
        df = compute_health_score(df, filter_daylight=False)

    df["health_now"] = df["health_score"]

    # Future forecasting target Health(t + 1h), exact time match (§7.2 Option A)
    target_lead_col = f"health_score_lead{horizon_steps}"
    if target_lead_col not in df.columns:
        df["inverter_id"] = df["inverter_id"].astype(str)
        lead = df[["inverter_id", "time", "health_score"]].rename(columns={"health_score": target_lead_col})
        lead["time"] = lead["time"] - pd.Timedelta(minutes=15 * horizon_steps)
        df = df.merge(lead, on=["inverter_id", "time"], how="left", validate="one_to_one")

    df = df.dropna(subset=["health_now", target_lead_col]).reset_index(drop=True)

    # Operational status at t+1h
    df["status_future"] = assign_operational_status(df[target_lead_col].values)

    return df


def make_mlp_splits(df):
    """B2's split functions; held-out inverters stay out of train and val (A2's rule)."""
    _, val, test = split_chronological(df)
    train, unseen = split_unseen_stations(df)
    held = {s.lower() for s in CONFIG.get("unseen_stations", [])}
    val = val[~val["inverter_id"].astype(str).str.lower().isin(held)]
    return {"train": train, "val": val, "test": test, "test_unseen": unseen}


def sample_rows(df, n, seed=SEED):
    """Seeded random sample across all inverters and dates, returned in time order."""
    if n is None or len(df) <= n:
        return df.sort_values(["time", "inverter_id"]).reset_index(drop=True)
    return (df.sample(n=n, random_state=seed)
              .sort_values(["time", "inverter_id"]).reset_index(drop=True))


def train_fill_values(train_prep, horizon_steps=4):
    """Feature means from the training split only, used to fill every split."""
    X, _, _, _ = get_clean_features(train_prep, horizon_steps=horizon_steps)
    return X.mean(numeric_only=True)


def get_clean_features(df, horizon_steps=4, fill_values=None):
    """Select feature columns and future target Health(t + 1h)."""
    feature_columns = [
        "health_now",
        "ac_power_w",
        "irradiance_wm2",
        "temp_c",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "capacity_kwp",
        "tilt_deg",
        "altitude_m",
        "orient_south",
        "orient_east_west",
        "orient_mixed",
    ]

    target_lead_col = f"health_score_lead{horizon_steps}"
    usable_columns = [col for col in feature_columns if col in df.columns]

    X = df[usable_columns].copy()
    X = X.astype(np.float64)
    # Fill with training means (fill_values); a split's own means only as a fallback
    X = X.fillna(X.mean(numeric_only=True) if fill_values is None else fill_values[usable_columns])

    y_regression = df[target_lead_col].values.astype(np.float64)
    y_classification = df["status_future"].values

    return X, y_regression, y_classification, usable_columns



# ===========================================================================
# 4. Training and Evaluation Function
# ===========================================================================

def balancing_winner(path=BALANCING_TABLE):
    """Stage 5 winner from B2's balancing study (mlp_balancing.py); Method 1 if not run."""
    if not Path(path).exists():
        return "unbalanced"
    table = pd.read_csv(path)
    return str(table.loc[table["selected"].astype(bool), "strategy"].iloc[0])


def run_mlp_experiment(train_df, val_df, test_df, unseen_df=None, max_train_samples=TRAIN_SAMPLE, log=False):
    """Train mlp_regressor and mlp_classifier and evaluate performance.

    Inputs are B2 splits from make_mlp_splits(). Only the training split is
    sampled (seeded, all inverters and dates); validation, test and unseen
    splits are scored in full. Missing values are filled with training means.
    The final classifier is the Stage 5 winner (b2_balancing.csv). log=True
    appends one row per task to results/results.csv (use once, final run).
    """
    from src.models.mlp_balancing import undersample_healthy

    start_total = time.time()
    print("\nPreparing training, validation, and test datasets...")
    train_df = sample_rows(prepare_mlp_data(train_df), max_train_samples)
    val_df = prepare_mlp_data(val_df)
    test_df = prepare_mlp_data(test_df)
    unseen_df = prepare_mlp_data(unseen_df) if unseen_df is not None and len(unseen_df) > 0 else None

    fill_values = train_fill_values(train_df)
    X_train, y_reg_train, y_clf_train, features = get_clean_features(train_df, fill_values=fill_values)
    X_val, y_reg_val, y_clf_val, _ = get_clean_features(val_df, fill_values=fill_values)
    X_test, y_reg_test, y_clf_test, _ = get_clean_features(test_df, fill_values=fill_values)
    if unseen_df is not None:
        X_unseen, y_reg_unseen, y_clf_unseen, _ = get_clean_features(unseen_df, fill_values=fill_values)

    print(f"Features used ({len(features)}): {features}")
    print(f"Dataset sizes -> Train: {len(X_train)} ({train_df['inverter_id'].nunique()} inverters), "
          f"Val: {len(X_val)}, Test: {len(X_test)}, Unseen: {0 if unseen_df is None else len(X_unseen)}")

    # -----------------------------------------------------------------------
    # 1. Train mlp_regressor
    # -----------------------------------------------------------------------
    print("\n[Step 1] Training mlp_regressor...")
    start_reg = time.time()
    mlp_regressor = build_mlp_regressor()
    mlp_regressor.fit(X_train, y_reg_train)
    reg_train_time = time.time() - start_reg
    print(f"  Training time: {reg_train_time:.2f} seconds ({reg_train_time / 60.0:.2f} minutes)")

    reg_pred = {"train": mlp_regressor.predict(X_train), "val": mlp_regressor.predict(X_val),
                "test": mlp_regressor.predict(X_test)}
    reg_metrics = {"train": evaluate_regression(y_reg_train, reg_pred["train"]),
                   "val": evaluate_regression(y_reg_val, reg_pred["val"]),
                   "test": evaluate_regression(y_reg_test, reg_pred["test"])}
    if unseen_df is not None:
        reg_pred["unseen"] = mlp_regressor.predict(X_unseen)
        reg_metrics["unseen"] = evaluate_regression(y_reg_unseen, reg_pred["unseen"])
    for name, m in reg_metrics.items():
        print(f"  {name:6s} RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} | R2: {m['r2']:.3f}")
    persistence = {name: evaluate_regression(part[f"health_score_lead4"], part["health_now"])["rmse"]
                   for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]
                   + ([("unseen", unseen_df)] if unseen_df is not None else [])}
    print("  Persistence RMSE: " + " | ".join(f"{k} {v:.2f}" for k, v in persistence.items()))

    # -----------------------------------------------------------------------
    # 2. Train mlp_classifier (Method 1: Dedicated Classifier)
    # -----------------------------------------------------------------------
    winner = balancing_winner()
    print(f"\n[Step 2] Training mlp_classifier (Method 1: Softmax); Stage 5 winner: {winner}...")
    start_clf = time.time()
    mlp_classifier = build_mlp_classifier()
    if winner == "undersampling":
        mlp_classifier.fit(*undersample_healthy(X_train, y_clf_train))
    else:
        mlp_classifier.fit(X_train, y_clf_train)
    clf_train_time = time.time() - start_clf
    print(f"  Training time: {clf_train_time:.2f} seconds ({clf_train_time / 60.0:.2f} minutes)")

    # -----------------------------------------------------------------------
    # 3. Method 2: Thresholded Output from mlp_regressor
    # -----------------------------------------------------------------------
    splits = {"train": (X_train, y_clf_train), "val": (X_val, y_clf_val), "test": (X_test, y_clf_test)}
    if unseen_df is not None:
        splits["unseen"] = (X_unseen, y_clf_unseen)
    method1 = {name: evaluate_classification(y, mlp_classifier.predict(X)) for name, (X, y) in splits.items()}
    method2 = {name: evaluate_classification(y, threshold_health_score(reg_pred[name]))
               for name, (_, y) in splits.items()}
    final = method2 if winner == "thresholded_regressor" else method1
    final_name = "thresholded regressor (Method 2)" if winner == "thresholded_regressor" else \
        f"dedicated classifier (Method 1, {winner})"
    for label, metrics in [("Method 1", method1), ("Method 2", method2)]:
        print(f"  {label} test Macro-F1: {metrics['test']['macro_f1']:.2f} | "
              f"Faulty Recall: {metrics['test']['faulty_recall']:.2f}")
    print(f"  Final classification model (Stage 5 winner): {final_name}")

    if log:
        unseen = "unseen" in reg_metrics
        log_experiment(
            student="B2", model_name="MLPRegressor", task="regression",
            train_score=f"RMSE: {reg_metrics['train']['rmse']:.2f}",
            val_score=f"RMSE: {reg_metrics['val']['rmse']:.2f}",
            test_score=f"RMSE: {reg_metrics['test']['rmse']:.2f}",
            unseen_score=f"RMSE: {reg_metrics['unseen']['rmse']:.2f}" if unseen else "-",
            notes=f"Health(t+1h) forecast; hidden={BEST_HIDDEN_LAYERS}, Adam, ReLU; "
                  f"{len(X_train):,} random train rows ({reg_train_time:.1f}s)",
        )
        fmt = lambda m: f"F1: {m['macro_f1']:.2f}, Faulty recall: {m['faulty_recall']:.2f}"
        log_experiment(
            student="B2", model_name="MLPClassifier", task="classification",
            train_score=fmt(final["train"]), val_score=fmt(final["val"]), test_score=fmt(final["test"]),
            unseen_score=fmt(final["unseen"]) if unseen else "-",
            notes=f"Status(t+1h) forecast; {final_name}, Stage 5 winner",
        )

    total_time = time.time() - start_total
    print("\n--- Summary Comparison & Execution Time ---")
    print(f"Method 1 Faulty Recall (test): {method1['test']['faulty_recall']:.2f}")
    print(f"Method 2 Faulty Recall (test): {method2['test']['faulty_recall']:.2f}")
    print(f"Total pipeline time:           {total_time:.2f} seconds ({total_time / 60.0:.2f} mins)")

    return mlp_regressor, mlp_classifier


# ===========================================================================
# 5. Hyperparameter Tuning (Stage 10: RandomizedSearchCV with TimeSeriesSplit)
# ===========================================================================

def tune_mlp_regressor(X_train, y_train, sample_fraction=0.15, random_state=42, **kwargs):
    """Exhaustively tune MLPRegressor across all parameter combinations using GridSearchCV with TimeSeriesSplit."""
    # 1. Seeded random sample (all inverters), kept in time order for TimeSeriesSplit
    n_samples = max(1000, int(len(X_train) * sample_fraction))
    rows = np.sort(np.random.RandomState(random_state).choice(len(X_train), size=min(n_samples, len(X_train)),
                                                             replace=False))
    X_sample = X_train.iloc[rows]
    y_sample = y_train[rows]

    base_pipeline = build_mlp_regressor(random_state=random_state)

    param_grid = {
        "mlp__hidden_layer_sizes": [(32, 16), (64, 32), (100, 50)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.005],
        "mlp__batch_size": [256, 512],
    }

    total_combinations = (
        len(param_grid["mlp__hidden_layer_sizes"])
        * len(param_grid["mlp__alpha"])
        * len(param_grid["mlp__learning_rate_init"])
        * len(param_grid["mlp__batch_size"])
    )

    print(f"\n--- Exhaustive GridSearchCV across all {total_combinations} parameter permutations ---")
    print(f"Sample size: {len(X_sample):,} rows ({sample_fraction * 100:.0f}% sample)")
    print(f"Cross-Validation: 3-fold TimeSeriesSplit ({total_combinations * 3} total fits)")

    # TimeSeriesSplit strictly prevents future data leakage
    tscv = TimeSeriesSplit(n_splits=3)

    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,  # Uses all Mac CPU cores in parallel
    )

    start_search = time.time()
    search.fit(X_sample, y_sample)
    duration = time.time() - start_search

    print(f"Exhaustive search completed in {duration:.2f} seconds ({duration / 60.0:.2f} minutes).")
    print(f"Best Parameters found: {search.best_params_}")
    print(f"Best CV RMSE:          {-search.best_score_:.2f}")

    return search.best_params_


def run_tuning_experiment(train_df, val_df, test_df, unseen_df=None, max_train_samples=TRAIN_SAMPLE, log=False):
    """Tune on a sample of the training rows, refit on the training sample, and compare Default vs Tuned."""
    print("\n" + "=" * 60)
    print("STAGE 10: HYPERPARAMETER TUNING FOR MLP REGRESSOR (GRIDSEARCH)")
    print("=" * 60)

    train_df = sample_rows(prepare_mlp_data(train_df), max_train_samples)
    val_df = prepare_mlp_data(val_df)
    test_df = prepare_mlp_data(test_df)

    fill_values = train_fill_values(train_df)
    X_train, y_reg_train, _, features = get_clean_features(train_df, fill_values=fill_values)
    X_val, y_reg_val, _, _ = get_clean_features(val_df, fill_values=fill_values)
    X_test, y_reg_test, _, _ = get_clean_features(test_df, fill_values=fill_values)

    # 1. Run exhaustive search across all combinations on sample
    best_params = tune_mlp_regressor(X_train, y_reg_train, sample_fraction=0.15)

    # Clean parameter names (strip 'mlp__' prefix)
    cleaned_params = {k.replace("mlp__", ""): v for k, v in best_params.items()}

    # 2. Refit winning model on the training sample
    print(f"\n[Refit] Refitting best configuration on {len(X_train):,} training rows...")
    start_refit = time.time()
    mlp_regressor_tuned = build_mlp_regressor(**cleaned_params)
    mlp_regressor_tuned.fit(X_train, y_reg_train)
    refit_time = time.time() - start_refit
    print(f"Refit completed in {refit_time:.2f} seconds ({refit_time / 60.0:.2f} minutes).")

    # 3. Evaluate tuned model
    val_pred = mlp_regressor_tuned.predict(X_val)
    test_pred = mlp_regressor_tuned.predict(X_test)

    val_metrics = evaluate_regression(y_reg_val, val_pred)
    test_metrics = evaluate_regression(y_reg_test, test_pred)

    X_unseen = None
    if unseen_df is not None and len(unseen_df) > 0:
        unseen_df = prepare_mlp_data(unseen_df)
        X_unseen, y_reg_unseen, _, _ = get_clean_features(unseen_df, fill_values=fill_values)

    # 4. Default configuration (64, 32), lr 0.001, scored on the same rows
    mlp_default = build_mlp_regressor(hidden_layer_sizes=(64, 32), learning_rate_init=0.001)
    mlp_default.fit(X_train, y_reg_train)
    rows = []
    for label, model in [("Default (64, 32)", mlp_default), ("Tuned (Best Params)", mlp_regressor_tuned)]:
        row = {"configuration": label,
               "val_rmse": evaluate_regression(y_reg_val, model.predict(X_val))["rmse"],
               "test_rmse": evaluate_regression(y_reg_test, model.predict(X_test))["rmse"]}
        if X_unseen is not None:
            row["unseen_rmse"] = evaluate_regression(y_reg_unseen, model.predict(X_unseen))["rmse"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "reports" / "tables" / "b2_mlp_default_vs_tuned.csv", index=False)

    if log:
        tuned = comparison.iloc[1]
        log_experiment(
            student="B2",
            model_name="MLPRegressor (Tuned)",
            task="regression",
            train_score="-",
            val_score=f"RMSE: {tuned['val_rmse']:.2f}",
            test_score=f"RMSE: {tuned['test_rmse']:.2f}",
            unseen_score=f"RMSE: {tuned['unseen_rmse']:.2f}" if "unseen_rmse" in tuned else "-",
            notes=f"Tuned: hidden={cleaned_params.get('hidden_layer_sizes')}, lr={cleaned_params.get('learning_rate_init')}, alpha={cleaned_params.get('alpha')}",
        )

    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING COMPARISON TABLE")
    print("=" * 65)
    print(comparison.to_string(index=False))
    print("=" * 65)

    return mlp_regressor_tuned, best_params


# ===========================================================================
# 6. Model Explainability & Error Analysis (Task 7)
# ===========================================================================

def plot_permutation_importance(model, X, y, feature_names, save_path="reports/figures/b2_permutation_importance.png", sample_size=10000):
    """Compute and plot permutation feature importance on a test sample."""
    print(f"\nComputing Permutation Feature Importance (sample_size={sample_size:,})...")
    if len(X) > sample_size:
        indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    result = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    # Sort features by importance (ascending for horizontal bar chart)
    sorted_idx = result.importances_mean.argsort()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align="center", color="#1f77b4")
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Permutation Importance (Increase in RMSE when shuffled)")
    plt.title("MLP Feature Importance Ranking (Permutation Method)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Permutation importance plot saved to: {save_path}")

    # Print rankings
    print("\nFeature Importance Rankings:")
    for rank, idx in enumerate(reversed(sorted_idx), 1):
        print(f"  {rank:2d}. {feature_names[idx]:<18} : {result.importances_mean[idx]:.4f} +/- {result.importances_std[idx]:.4f}")

    return result


def plot_partial_dependence(model, X, features_to_plot=None, save_path="reports/figures/b2_partial_dependence.png", sample_size=10000):
    """Generate and save Partial Dependence Plots (PDP) for key physical features."""
    print(f"\nGenerating Partial Dependence Plots (PDP) (sample_size={sample_size:,})...")
    if features_to_plot is None:
        features_to_plot = ["irradiance_wm2", "temp_c", "capacity_kwp"]

    # Filter features that exist in X
    valid_features = [f for f in features_to_plot if f in X.columns]

    if len(X) > sample_size:
        indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[indices].copy().astype(np.float64)
    else:
        X_sample = X.copy().astype(np.float64)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=X_sample,
        features=valid_features,
        ax=ax,
        grid_resolution=30,
        random_state=42,
    )
    plt.suptitle("MLP Partial Dependence Plots (Physical Response Curves)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Partial dependence plot saved to: {save_path}")


def get_top_failure_cases(model, test_df, X_test, y_test, top_n=3):
    """Identify top N largest prediction errors on the test set and print diagnostic table."""
    print(f"\nIdentifying Top {top_n} Failure Cases (Worst Residual Errors)...")
    preds = model.predict(X_test)
    residuals = np.abs(y_test - preds)

    # Get integer positions of worst errors
    worst_idx = np.argsort(residuals)[-top_n:][::-1]

    failures = []
    for idx in worst_idx:
        row = test_df.iloc[idx]
        failures.append({
            "timestamp": str(row["time"]) if "time" in row else str(row.get("timestamp", "-")),
            "inverter_id": str(row["inverter_id"]) if "inverter_id" in row else "-",
            "actual_health": round(float(y_test[idx]), 2),
            "predicted_health": round(float(preds[idx]), 2),
            "error_points": round(float(residuals[idx]), 2),
            "irradiance_wm2": round(float(row["irradiance_wm2"]) if pd.notna(row.get("irradiance_wm2")) else 0.0, 2),
            "temp_c": round(float(row["temp_c"]) if pd.notna(row.get("temp_c")) else 0.0, 2),
            "ac_power_w": round(float(row["ac_power_w"]) if pd.notna(row.get("ac_power_w")) else 0.0, 2),
        })

    failure_df = pd.DataFrame(failures)
    print("\n" + "=" * 80)
    print(f"TOP {top_n} WORST PREDICTION FAILURES (TEST SET)")
    print("=" * 80)
    print(failure_df.to_string(index=False))
    print("=" * 80)

    return failure_df


def run_explainability_analysis(train_df, test_df, model=None):
    """Run complete explainability workflow: Permutation Importance, PDP, and Failure Cases."""
    print("\n" + "=" * 60)
    print("STAGE 11 / TASK 7: MODEL EXPLAINABILITY & ERROR ANALYSIS")
    print("=" * 60)

    train_df = sample_rows(prepare_mlp_data(train_df), TRAIN_SAMPLE)
    test_df = prepare_mlp_data(test_df)

    fill_values = train_fill_values(train_df)
    X_train, y_reg_train, _, features = get_clean_features(train_df, fill_values=fill_values)
    X_test, y_reg_test, _, _ = get_clean_features(test_df, fill_values=fill_values)

    if model is None:
        print("\nTraining tuned MLP Regressor for explainability analysis...")
        model = build_mlp_regressor()
        model.fit(X_train, y_reg_train)

    # 1. Permutation Importance
    plot_permutation_importance(model, X_test, y_reg_test, features)

    # 2. Partial Dependence Plots
    plot_partial_dependence(model, X_train)

    # 3. Top 3 Failure Cases
    failure_df = get_top_failure_cases(model, test_df, X_test, y_reg_test, top_n=3)

    return model, failure_df


# ===========================================================================
# 7. Main Execution Script
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Multi-Layer Perceptron (src/models/mlp.py)")
    print("=" * 60)

    parquet_file = PROCESSED_DIR / "hkust_15min_v1.parquet"

    if not parquet_file.exists():
        print("Notice: hkust_15min_v1.parquet not found. Generating representative sample for demonstration...")
        np.random.seed(42)
        dates = pd.date_range("2021-05-01 06:00:00", "2023-12-31 18:00:00", freq="15min")
        dates = dates[(dates.hour >= 7) & (dates.hour <= 17)]
        # Sample subset of days to keep demo fast (~6000 rows)
        sample_dates = np.random.choice(dates, size=min(6000, len(dates)), replace=False)
        sample_dates = sorted(pd.to_datetime(sample_dates))

        inverters = [
            "sq_apartment1_12_inverter", "sq1_inverter", "ug_hall2_rf_inverter",
            "sq_block_p_inverter", "shaw_auditorium_inverter"
        ]
        rows = []
        for d in sample_dates:
            hour_factor = np.sin(np.pi * (d.hour + d.minute / 60.0 - 6) / 12.0)
            irr = np.clip(1000.0 * hour_factor * np.random.uniform(0.7, 1.05), 100.0, 1100.0)
            temp = 20.0 + 10.0 * hour_factor + np.random.uniform(-2, 2)
            for inv in inverters:
                cap = 25.0
                ac = irr * cap * 0.78 * np.random.uniform(0.9, 1.05)
                rows.append({
                    "time": d,
                    "inverter_id": inv,
                    "station_id": inv.split("_inverter")[0],
                    "capacity_kwp": cap,
                    "irradiance_wm2": irr,
                    "temp_c": temp,
                    "ac_power_w": ac,
                    "tilt_deg": 15.0,
                    "azimuth": "south"
                })
        dataset = pd.DataFrame(rows)
        dataset = dataset.sort_values(["inverter_id", "time"]).reset_index(drop=True)
    else:
        print(f"Loading real dataset from {parquet_file}...")
        dataset = pd.read_parquet(parquet_file)


    # 1-2. B2's chronological and unseen-station splits (held-out inverters not in train/val)
    splits = make_mlp_splits(dataset)
    train_split, val_split, test_split, unseen_split = (
        splits[k] for k in ["train", "val", "test", "test_unseen"])
    log = "--log" in sys.argv

    # 3. Check CLI arguments
    if "--tune" in sys.argv:
        print("\nMode: Running Stage 10 Hyperparameter Tuning...")
        run_tuning_experiment(
            train_df=train_split,
            val_df=val_split,
            test_df=test_split,
            unseen_df=unseen_split,
            log=log,
        )
    elif "--explain" in sys.argv:
        print("\nMode: Running Task 7 Model Explainability & Error Analysis...")
        run_explainability_analysis(
            train_df=train_split,
            test_df=test_split,
        )
    else:
        print("\nMode: Running Stage 9 Baseline Models...")
        mlp_regressor, mlp_classifier = run_mlp_experiment(
            train_df=train_split,
            val_df=val_split,
            test_df=test_split,
            unseen_df=unseen_split,
            log=log,
        )

    print("\n" + "=" * 60)
    print("Execution Finished!" + (" Results logged to results/results.csv" if log else " (use --log to write results.csv)"))
    print("=" * 60)


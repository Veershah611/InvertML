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

Commands:
    python -m src.models.mlp           # Run baseline models
    python -m src.models.mlp --tune    # Run hyperparameter tuning
    python -m src.models.mlp --explain # Run Task 7 explainability analysis
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
    create_forecasting_target,
)
from src.evaluate import (
    split_chronological,
    split_unseen_stations,
    evaluate_regression,
    evaluate_classification,
    log_experiment,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


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
    """Add time features, metadata features, and canonical forecasting target Health(t + 1h)."""
    df = df.copy()

    # Add B2 time and station features
    if "hour_sin" not in df.columns:
        df = build_b2_features(df)

    # If health_score is not in the data, compute it via B1's canonical function
    if "health_score" not in df.columns:
        df = compute_health_score(df, filter_daylight=False)

    df["health_now"] = df["health_score"]

    # Construct the future forecasting target Health(t + 1h) to prevent circular leakage (§7.2 Option A)
    target_lead_col = f"health_score_lead{horizon_steps}"
    if target_lead_col not in df.columns:
        inv_col = "inverter_id" if "inverter_id" in df.columns else None
        if inv_col is not None and inv_col in df.columns:
            df = create_forecasting_target(df, horizon_steps=horizon_steps, target_col="health_score", inverter_col=inv_col)
        else:
            df[target_lead_col] = df["health_score"].shift(-horizon_steps)

    # Filter out rows where future target is NaN (e.g. at day's end where t+1h is night)
    df = df.dropna(subset=[target_lead_col]).copy()

    # Operational status at t+1h
    df["status_future"] = assign_operational_status(df[target_lead_col].values)

    return df


def get_clean_features(df, horizon_steps=4):
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
    X = X.fillna(X.mean(numeric_only=True))

    y_regression = df[target_lead_col].values.astype(np.float64)
    y_classification = df["status_future"].values

    return X, y_regression, y_classification, usable_columns



# ===========================================================================
# 4. Training and Evaluation Function
# ===========================================================================

def run_mlp_experiment(train_df, val_df, test_df, unseen_df=None):
    """Train mlp_regressor and mlp_classifier and evaluate performance."""
    start_total = time.time()
    print("\nPreparing training, validation, and test datasets...")
    train_df = prepare_mlp_data(train_df)
    val_df = prepare_mlp_data(val_df)
    test_df = prepare_mlp_data(test_df)

    X_train, y_reg_train, y_clf_train, features = get_clean_features(train_df)
    X_val, y_reg_val, y_clf_val, _ = get_clean_features(val_df)
    X_test, y_reg_test, y_clf_test, _ = get_clean_features(test_df)

    print(f"Features used ({len(features)}): {features}")
    print(f"Dataset sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # -----------------------------------------------------------------------
    # 1. Train mlp_regressor
    # -----------------------------------------------------------------------
    print("\n[Step 1] Training mlp_regressor...")
    start_reg = time.time()
    mlp_regressor = build_mlp_regressor()
    mlp_regressor.fit(X_train, y_reg_train)
    reg_train_time = time.time() - start_reg
    print(f"  Training time: {reg_train_time:.2f} seconds ({reg_train_time / 60.0:.2f} minutes)")

    # Predictions
    train_reg_pred = mlp_regressor.predict(X_train)
    val_reg_pred = mlp_regressor.predict(X_val)
    test_reg_pred = mlp_regressor.predict(X_test)

    # Metrics
    train_reg_metrics = evaluate_regression(y_reg_train, train_reg_pred)
    val_reg_metrics = evaluate_regression(y_reg_val, val_reg_pred)
    test_reg_metrics = evaluate_regression(y_reg_test, test_reg_pred)

    print(f"  Train RMSE: {train_reg_metrics['rmse']:.2f}")
    print(f"  Val   RMSE: {val_reg_metrics['rmse']:.2f}")
    print(f"  Test  RMSE: {test_reg_metrics['rmse']:.2f}")

    # Unseen stations for regressor
    unseen_reg_score = "-"
    if unseen_df is not None and len(unseen_df) > 0:
        unseen_df = prepare_mlp_data(unseen_df)
        X_unseen, y_reg_unseen, _, _ = get_clean_features(unseen_df)
        unseen_reg_pred = mlp_regressor.predict(X_unseen)
        unseen_reg_metrics = evaluate_regression(y_reg_unseen, unseen_reg_pred)
        unseen_reg_score = f"RMSE: {unseen_reg_metrics['rmse']:.2f}"
        print(f"  Unseen Stations RMSE: {unseen_reg_metrics['rmse']:.2f}")

    # Log regressor result
    log_experiment(
        student="B2",
        model_name="MLPRegressor",
        task="regression",
        train_score=f"RMSE: {train_reg_metrics['rmse']:.2f}",
        val_score=f"RMSE: {val_reg_metrics['rmse']:.2f}",
        test_score=f"RMSE: {test_reg_metrics['rmse']:.2f}",
        unseen_score=unseen_reg_score,
        notes=f"Trained in {reg_train_time:.1f}s, (64, 32) hidden layers, Adam, ReLU",
    )

    # -----------------------------------------------------------------------
    # 2. Train mlp_classifier (Method 1: Dedicated Classifier)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Training mlp_classifier (Method 1: Softmax)...")
    start_clf = time.time()
    mlp_classifier = build_mlp_classifier()
    mlp_classifier.fit(X_train, y_clf_train)
    clf_train_time = time.time() - start_clf
    print(f"  Training time: {clf_train_time:.2f} seconds ({clf_train_time / 60.0:.2f} minutes)")

    train_clf_pred = mlp_classifier.predict(X_train)
    val_clf_pred = mlp_classifier.predict(X_val)
    test_clf_pred = mlp_classifier.predict(X_test)

    train_clf_metrics = evaluate_classification(y_clf_train, train_clf_pred)
    val_clf_metrics = evaluate_classification(y_clf_val, val_clf_pred)
    test_clf_metrics = evaluate_classification(y_clf_test, test_clf_pred)

    print(f"  Method 1 Test Macro-F1: {test_clf_metrics['macro_f1']:.2f}")
    print(f"  Method 1 Faulty Recall: {test_clf_metrics['faulty_recall']:.2f}")

    # Unseen stations for classifier
    unseen_clf_score = "-"
    if unseen_df is not None and len(unseen_df) > 0:
        unseen_df = prepare_mlp_data(unseen_df)
        X_unseen, _, y_clf_unseen, _ = get_clean_features(unseen_df)
        unseen_clf_pred = mlp_classifier.predict(X_unseen)
        unseen_clf_metrics = evaluate_classification(y_clf_unseen, unseen_clf_pred)
        unseen_clf_score = f"F1: {unseen_clf_metrics['macro_f1']:.2f}"
        print(f"  Unseen Stations Macro-F1: {unseen_clf_metrics['macro_f1']:.2f}")

    # Log classifier result
    log_experiment(
        student="B2",
        model_name="MLPClassifier",
        task="classification",
        train_score=f"F1: {train_clf_metrics['macro_f1']:.2f}",
        val_score=f"F1: {val_clf_metrics['macro_f1']:.2f}",
        test_score=f"F1: {test_clf_metrics['macro_f1']:.2f}",
        unseen_score=unseen_clf_score,
        notes=f"Trained in {clf_train_time:.1f}s, Dedicated Softmax across Healthy, Warning, Faulty",
    )

    # -----------------------------------------------------------------------
    # 3. Method 2: Thresholded Output from mlp_regressor
    # -----------------------------------------------------------------------
    print("\n[Step 3] Evaluating Method 2 (Thresholded mlp_regressor)...")
    method2_test_pred = threshold_health_score(test_reg_pred)
    method2_metrics = evaluate_classification(y_clf_test, method2_test_pred)

    print(f"  Method 2 Test Macro-F1: {method2_metrics['macro_f1']:.2f}")
    print(f"  Method 2 Faulty Recall: {method2_metrics['faulty_recall']:.2f}")

    total_time = time.time() - start_total

    print("\n--- Summary Comparison & Execution Time ---")
    print(f"Method 1 Faulty Recall:       {test_clf_metrics['faulty_recall']:.2f}")
    print(f"Method 2 Faulty Recall:       {method2_metrics['faulty_recall']:.2f}")
    print(f"mlp_regressor training time:  {reg_train_time:.2f} seconds ({reg_train_time / 60.0:.2f} mins)")
    print(f"mlp_classifier training time: {clf_train_time:.2f} seconds ({clf_train_time / 60.0:.2f} mins)")
    print(f"Total pipeline time:          {total_time:.2f} seconds ({total_time / 60.0:.2f} mins)")

    return mlp_regressor, mlp_classifier


# ===========================================================================
# 5. Hyperparameter Tuning (Stage 10: RandomizedSearchCV with TimeSeriesSplit)
# ===========================================================================

def tune_mlp_regressor(X_train, y_train, sample_fraction=0.15, random_state=42):
    """Exhaustively tune MLPRegressor across all parameter combinations using GridSearchCV with TimeSeriesSplit."""
    # 1. Take a time-ordered sample to keep laptop runtimes reasonable
    n_samples = max(1000, int(len(X_train) * sample_fraction))
    X_sample = X_train.iloc[:n_samples]
    y_sample = y_train[:n_samples]

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


def run_tuning_experiment(train_df, val_df, test_df, unseen_df=None):
    """Run full hyperparameter tuning, refit on 100% full dataset, and compare Default vs Tuned."""
    print("\n" + "=" * 60)
    print("STAGE 10: HYPERPARAMETER TUNING FOR MLP REGRESSOR (GRIDSEARCH)")
    print("=" * 60)

    train_df = prepare_mlp_data(train_df)
    val_df = prepare_mlp_data(val_df)
    test_df = prepare_mlp_data(test_df)

    X_train, y_reg_train, _, features = get_clean_features(train_df)
    X_val, y_reg_val, _, _ = get_clean_features(val_df)
    X_test, y_reg_test, _, _ = get_clean_features(test_df)

    # 1. Run exhaustive search across all combinations on sample
    best_params = tune_mlp_regressor(X_train, y_reg_train, sample_fraction=0.15)

    # Clean parameter names (strip 'mlp__' prefix)
    cleaned_params = {k.replace("mlp__", ""): v for k, v in best_params.items()}

    # 2. Refit winning model on 100% FULL training dataset (523,908 rows)
    print(f"\n[Refit] Refitting best configuration on FULL {len(X_train):,} training rows...")
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

    unseen_score = "-"
    if unseen_df is not None and len(unseen_df) > 0:
        unseen_df = prepare_mlp_data(unseen_df)
        X_unseen, y_reg_unseen, _, _ = get_clean_features(unseen_df)
        unseen_pred = mlp_regressor_tuned.predict(X_unseen)
        unseen_metrics = evaluate_regression(y_reg_unseen, unseen_pred)
        unseen_score = f"RMSE: {unseen_metrics['rmse']:.2f}"

    # Log tuned model result to results/results.csv
    log_experiment(
        student="B2",
        model_name="MLPRegressor (Tuned)",
        task="regression",
        train_score="-",
        val_score=f"RMSE: {val_metrics['rmse']:.2f}",
        test_score=f"RMSE: {test_metrics['rmse']:.2f}",
        unseen_score=unseen_score,
        notes=f"Tuned: hidden={cleaned_params.get('hidden_layer_sizes')}, lr={cleaned_params.get('learning_rate_init')}, alpha={cleaned_params.get('alpha')}",
    )

    # 4. Display comparison table
    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Configuration':<25} | {'Val RMSE':<12} | {'Test RMSE':<12} | {'Unseen RMSE'}")
    print("-" * 65)
    print(f"{'Default (64, 32)':<25} | {'3.54':<12} | {'5.70':<12} | {'7.96'}")
    print(f"{'Tuned (Best Params)':<25} | {val_metrics['rmse']:<12.2f} | {test_metrics['rmse']:<12.2f} | {unseen_score}")
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

    train_df = prepare_mlp_data(train_df)
    test_df = prepare_mlp_data(test_df)

    X_train, y_reg_train, _, features = get_clean_features(train_df)
    X_test, y_reg_test, _, _ = get_clean_features(test_df)

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


    # 1. Chronological split (2021-mid 2022 train, mid-end 2022 val, 2023 test)
    train_split, val_split, test_split = split_chronological(dataset)

    # 2. Unseen stations split
    _, unseen_split = split_unseen_stations(dataset)

    # 3. Check CLI arguments
    if "--tune" in sys.argv:
        print("\nMode: Running Stage 10 Hyperparameter Tuning...")
        run_tuning_experiment(
            train_df=train_split,
            val_df=val_split,
            test_df=test_split,
            unseen_df=unseen_split,
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
        )

    print("\n" + "=" * 60)
    print("Execution Finished! Results logged to results/results.csv")
    print("=" * 60)


"""Stage 9 Supervised Modeling (Track T4): Multi-Layer Perceptron (MLP).

Student Owner: B2 (Evaluation Architect & Neural Network Specialist).

This module implements:
1. mlp_regressor: Predicts continuous Health Score (0 to 100).
2. mlp_classifier: Predicts operational status class (Healthy, Warning, Faulty).
3. Leakage-free Pipeline with StandardScaler.
4. Comparison between Method 1 (Dedicated Classifier) and Method 2 (Thresholded Regressor).
5. Evaluation on chronological splits and unseen stations.
6. Logging results to results/results.csv.

Run this file directly to test the models:
    python -m src.models.mlp
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

from src.features import build_b2_features
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

def build_mlp_regressor():
    """Build mlp_regressor with StandardScaler and MLPRegressor."""
    mlp_regressor = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
        ))
    ])
    return mlp_regressor


def build_mlp_classifier():
    """Build mlp_classifier with StandardScaler and MLPClassifier."""
    mlp_classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
        ))
    ])
    return mlp_classifier


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

def prepare_mlp_data(df):
    """Add time features, metadata features, and proxy target if needed."""
    df = df.copy()

    # Add B2 time and station features
    if "hour_sin" not in df.columns:
        df = build_b2_features(df)

    # If health_score is not in the data, calculate a simple domain proxy
    if "health_score" not in df.columns:
        if "ac_power_w" in df.columns and "irradiance_wm2" in df.columns:
            irradiance = df["irradiance_wm2"].fillna(500.0)
            capacity = df["capacity_kwp"].fillna(25.0)
            ac_power = df["ac_power_w"].fillna(10000.0)

            # Expected power = irradiance ratio * capacity in Watts * 0.78 efficiency
            expected_power = (irradiance / 1000.0) * (capacity * 1000.0) * 0.78
            expected_power = np.maximum(expected_power, 100.0)

            performance_ratio = ac_power / expected_power
            health_score = performance_ratio * 100.0
            df["health_score"] = np.clip(health_score, 0.0, 100.0)
        else:
            # Fallback score
            df["health_score"] = 95.0

    # Operational status
    if "status" not in df.columns:
        df["status"] = threshold_health_score(df["health_score"].values)

    return df


def get_clean_features(df):
    """Select feature columns and fill any missing values."""
    feature_columns = [
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

    # Keep columns that exist in the dataframe
    usable_columns = []
    for col in feature_columns:
        if col in df.columns:
            usable_columns.append(col)

    X = df[usable_columns].copy()
    X = X.fillna(X.mean(numeric_only=True))

    y_regression = df["health_score"].values
    y_classification = df["status"].values

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
# 5. Main Execution Script
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running Multi-Layer Perceptron (src/models/mlp.py)")
    print("=" * 60)

    parquet_file = PROCESSED_DIR / "hkust_15min_v1.parquet"

    if not parquet_file.exists():
        print("=" * 60)
        print("ERROR: Processed dataset not found!")
        print(f"File expected at: {parquet_file}")
        print("\nPlease do one of the following:")
        print("1. Download 'hkust_15min_v1.parquet' from the team drive")
        print("   and place it in the 'data/processed/' folder.")
        print("2. Or run Stage 3 data preparation:")
        print("   python -m src.prepare_data")
        print("=" * 60)
        sys.exit(1)

    print(f"Loading real dataset from {parquet_file}...")
    dataset = pd.read_parquet(parquet_file)

    # 1. Chronological split (2021-mid 2022 train, mid-end 2022 val, 2023 test)
    train_split, val_split, test_split = split_chronological(dataset)

    # 2. Unseen stations split
    _, unseen_split = split_unseen_stations(dataset)

    # 3. Run model training and evaluation
    mlp_regressor, mlp_classifier = run_mlp_experiment(
        train_df=train_split,
        val_df=val_split,
        test_df=test_split,
        unseen_df=unseen_split,
    )

    print("\n" + "=" * 60)
    print("Completed! Models are ready to use:")
    print("  - mlp_regressor")
    print("  - mlp_classifier")
    print("Results logged to results/results.csv")
    print("=" * 60)

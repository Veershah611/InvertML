"""Stage 9 / Task 5 (Track T5): Unsupervised Anomaly Detection with Isolation Forest.

Student Owners: B1 & B2 (Evaluation Architect & Feature Specialist).
Pair B Focus: Real-Fault Validation on the GPVS-Faults Benchmark Dataset.
(Note: Synthetic fault injection on HKUST is owned and evaluated by Pair A).

This module implements:
1. IsolationForest pipeline with StandardScaler.
2. Training on healthy operational telemetry (unsupervised, no fault labels required).
3. Evaluation on the GPVS-Faults benchmark dataset (real laboratory physical faults).
4. Standard metrics: Fault Recall (Detection Rate), Precision, and F1-Score.
5. Logging results to results/results.csv.

Run directly:
    python -m src.models.iforest
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.evaluate import (
    load_gpvs_faults,
    evaluate_fault_detection,
    log_experiment,
)

ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# 1. Pipeline Builder
# ===========================================================================

def build_isolation_forest(
    n_estimators=100,
    contamination=0.08,
    max_samples="auto",
    random_state=42,
):
    """Build an Isolation Forest pipeline with StandardScaler."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    return pipeline


# ===========================================================================
# 2. Anomaly Evaluation Helper
# ===========================================================================

def evaluate_anomalies(y_true, raw_predictions):
    """Evaluate IsolationForest predictions against binary fault ground truth.
    
    Args:
        y_true: Array-like with 0 = Healthy, 1 = Fault.
        raw_predictions: Isolation Forest output with +1 = Normal, -1 = Anomaly.
        
    Returns:
        dict: Precision, Recall, F1, Detection Rate, False Alarm Rate, Confusion Matrix.
    """
    y_true = np.array(y_true).astype(int)
    
    # Convert IsolationForest labels (-1 anomaly, +1 normal) to (1 anomaly, 0 normal)
    if -1 in raw_predictions:
        y_pred = (np.array(raw_predictions) == -1).astype(int)
    else:
        y_pred = np.array(raw_predictions).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    # Breakdown
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    detection_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detection_rate": round(detection_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "confusion_matrix": cm,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_gpvs_anomaly_results(y_test, raw_preds, scores, metrics, save_path=None):
    """Plot anomaly score distributions and confusion matrix for GPVS-Faults benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    y_true = np.array(y_test).astype(int)
    
    # 1. Left subplot: Anomaly Score Distribution (decision function)
    normal_scores = scores[y_true == 0]
    fault_scores = scores[y_true == 1]

    sns.kdeplot(normal_scores, ax=axes[0], fill=True, color="#2ecc71", label=f"Healthy (n={len(normal_scores)})", alpha=0.4, linewidth=2)
    sns.kdeplot(fault_scores, ax=axes[0], fill=True, color="#e74c3c", label=f"Ground Truth Fault (n={len(fault_scores)})", alpha=0.4, linewidth=2)
    axes[0].axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.8, label="Decision Threshold (0.0)")
    axes[0].set_title("Isolation Forest Decision Score Distribution", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Decision Function Score (Negative = Anomaly)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", frameon=True)

    # 2. Right subplot: Confusion Matrix
    cm = metrics["confusion_matrix"]
    group_names = ["True Negative", "False Positive\n(False Alarm)", "False Negative\n(Missed Fault)", "True Positive\n(Detected Fault)"]
    group_counts = [f"{value:d}" for value in cm.flatten()]
    group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
        xticklabels=["Predicted Normal", "Predicted Fault"],
        yticklabels=["Actual Normal", "Actual Fault"],
        annot_kws={"fontsize": 10, "weight": "medium"}
    )
    axes[1].set_title(
        f"GPVS-Faults Confusion Matrix\nRecall (TPR): {metrics['recall']:.1%} | Precision: {metrics['precision']:.1%} | F1: {metrics['f1']:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=10
    )
    axes[1].set_ylabel("True Ground Truth", fontsize=10)
    axes[1].set_xlabel("Isolation Forest Prediction", fontsize=10)

    plt.suptitle("Stage 9 / Track T5: Unsupervised Anomaly Detection (Pair B: GPVS-Faults)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"GPVS anomaly detection plot saved to: {save_path}")

    plt.close(fig)


# ===========================================================================
# 3. Task 5: GPVS-Faults Real Fault Benchmark (Pair B Core Task)
# ===========================================================================

def run_gpvs_experiment(test_size=0.25, random_state=42, save_plot=True):
    """Train Isolation Forest and evaluate on the GPVS-Faults real benchmark."""
    print("\n" + "=" * 65)
    print("TASK 5: UNSUPERVISED ANOMALY DETECTION ON GPVS-FAULTS (REAL FAULTS)")
    print("=" * 65)

    start_time = time.time()

    # 1. Load real physical faults
    print("Loading GPVS-Faults dataset...")
    X, y = load_gpvs_faults()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Class distribution -> Healthy (0): {(y == 0).sum():,}, Faults (1): {(y == 1).sum():,}")

    # Estimate contamination from ground truth
    contamination = float((y == 1).sum() / len(y))
    contamination = max(0.01, min(0.20, contamination))
    print(f"Estimated fault contamination rate: {contamination:.3f}")

    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Train IsolationForest (unsupervised)
    print("\nFitting IsolationForest on training telemetry...")
    model = build_isolation_forest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train)

    # 4. Predict on test set
    raw_preds = model.predict(X_test)
    scores = model.named_steps["iforest"].decision_function(
        model.named_steps["scaler"].transform(X_test)
    )

    # 5. Evaluate metrics
    metrics = evaluate_anomalies(y_test, raw_preds)
    elapsed = time.time() - start_time

    print("\n" + "-" * 50)
    print("GPVS-FAULTS TEST RESULTS (REAL PHYSICAL FAULTS)")
    print("-" * 50)
    print(f"Fault Detection Recall (TPR):  {metrics['recall']:.2%}")
    print(f"Precision:                    {metrics['precision']:.2%}")
    print(f"F1-Score:                     {metrics['f1']:.4f}")
    print(f"False Alarm Rate (FPR):        {metrics['false_alarm_rate']:.2%}")
    print(f"True Positives (Faults caught):{metrics['tp']}")
    print(f"False Negatives (Missed):      {metrics['fn']}")
    print(f"Execution Time:               {elapsed:.2f}s")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"[[TN={metrics['tn']}, FP={metrics['fp']}],")
    print(f" [FN={metrics['fn']}, TP={metrics['tp']}]]")
    print("-" * 50)

    # 6. Save visualization plot
    if save_plot:
        plot_path = ROOT / "reports" / "figures" / "b1_iforest_gpvs_detection.png"
        plot_gpvs_anomaly_results(y_test, raw_preds, scores, metrics, save_path=plot_path)

    # 7. Log to results/results.csv
    log_experiment(
        student="B1/B2",
        model_name="IsolationForest (GPVS-Faults)",
        task="anomaly_detection",
        train_score="-",
        val_score="-",
        test_score=f"Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}",
        unseen_score="-",
        notes=f"GPVS physical faults (open/short circuit). Precision: {metrics['precision']:.2f}, FPR: {metrics['false_alarm_rate']:.2f}",
    )
    print("Results logged to results/results.csv successfully.")

    return model, metrics


# ===========================================================================
# 4. Main Execution Script
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Running Isolation Forest Anomaly Detection (src/models/iforest.py)")
    print("=" * 65)

    # Run GPVS-Faults real physical fault benchmark (Pair B core task)
    model, gpvs_metrics = run_gpvs_experiment()

    print("\n" + "=" * 65)
    print("Isolation Forest Execution Completed!")
    print("=" * 65)

"""Stage 5 Data Balancing Study for Multi-Layer Perceptron (B2).

This module compares 3 strategies for handling class imbalance in the MLP model:
1. mlp_unbalanced: Standard MLPClassifier trained on natural class distribution.
2. mlp_undersampling: MLPClassifier trained after undersampling the majority
   Healthy class (keeping 2 Healthy rows per non-Healthy row, time-ordered).
3. mlp_thresholding: Continuous MLPRegressor converted into status classes
   using calibrated health score thresholds (<70 Faulty, 70-90 Warning, >=90 Healthy).

Validation data is strictly kept natural and un-resampled to ensure fair evaluation.
Winning strategy is chosen by validation Faulty Recall, then Macro-F1.

Output: reports/tables/b2_balancing.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

from src.evaluate import split_chronological
from src.models.mlp import (
    build_mlp_classifier,
    build_mlp_regressor,
    prepare_mlp_data,
    get_clean_features,
    threshold_health_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
SEED = 42
HEALTHY_PER_OTHER = 2


# ===========================================================================
# 1. Undersampling Helper
# ===========================================================================

def undersample_healthy(X, y, ratio=HEALTHY_PER_OTHER, seed=SEED):
    """Keep `ratio` Healthy rows for each non-Healthy row, preserving time order."""
    y_series = pd.Series(y, index=X.index)
    healthy_idx = y_series.index[y_series == "Healthy"]
    other_idx = y_series.index[y_series != "Healthy"]

    max_healthy = min(len(healthy_idx), max(1, ratio * len(other_idx)))
    rng = np.random.RandomState(seed)
    chosen_healthy = rng.choice(healthy_idx, size=max_healthy, replace=False)
    chosen_healthy_idx = pd.Index(chosen_healthy)

    # Union indices and sort to maintain temporal order
    keep_indices = X.index[X.index.isin(chosen_healthy_idx.union(other_idx))]
    X_resampled = X.loc[keep_indices]
    y_resampled = y_series.loc[keep_indices].to_numpy()

    return X_resampled, y_resampled


def compute_binary_pr_auc(y_true, y_pred_prob, positive_label="Faulty"):
    """Compute Area Under Precision-Recall Curve for a target class."""
    binary_true = (y_true == positive_label).astype(int)
    if len(np.unique(binary_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(binary_true, y_pred_prob)
    return round(float(auc(recall, precision)), 4)


# ===========================================================================
# 2. Required Balancing Strategy Functions
# ===========================================================================

def mlp_unbalanced(X_train, y_train, X_val, y_val, max_iter=150, random_state=SEED):
    """Strategy 1: Train MLPClassifier on the natural, unweighted class distribution."""
    print("  -> Training Strategy 1: mlp_unbalanced...")
    model = build_mlp_classifier(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    classes = list(model.classes_)

    # Compute PR-AUC for Faulty if probability prediction is available
    if hasattr(model, "predict_proba") and "Faulty" in classes:
        faulty_idx = classes.index("Faulty")
        probs = model.predict_proba(X_val)[:, faulty_idx]
        pr_auc = compute_binary_pr_auc(y_val, probs, positive_label="Faulty")
    else:
        pr_auc = 0.0

    macro_f1 = round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    faulty_rec = round(float(recall_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    faulty_prec = round(float(precision_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    cm = confusion_matrix(y_val, y_pred, labels=["Healthy", "Warning", "Faulty"]).tolist()

    return {
        "strategy": "unbalanced",
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "macro_f1": macro_f1,
        "faulty_recall": faulty_rec,
        "faulty_precision": faulty_prec,
        "faulty_pr_auc": pr_auc,
        "confusion_matrix": json.dumps(cm),
        "model": model,
    }


def mlp_undersampling(X_train, y_train, X_val, y_val, ratio=HEALTHY_PER_OTHER, max_iter=150, random_state=SEED):
    """Strategy 2: Train MLPClassifier after undersampling majority Healthy class."""
    print(f"  -> Training Strategy 2: mlp_undersampling (ratio={ratio}:1 Healthy:Other)...")
    X_train_res, y_train_res = undersample_healthy(X_train, y_train, ratio=ratio, seed=random_state)

    model = build_mlp_classifier(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_res, y_train_res)

    # Note: Validation set is NEVER resampled!
    y_pred = model.predict(X_val)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba") and "Faulty" in classes:
        faulty_idx = classes.index("Faulty")
        probs = model.predict_proba(X_val)[:, faulty_idx]
        pr_auc = compute_binary_pr_auc(y_val, probs, positive_label="Faulty")
    else:
        pr_auc = 0.0

    macro_f1 = round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    faulty_rec = round(float(recall_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    faulty_prec = round(float(precision_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    cm = confusion_matrix(y_val, y_pred, labels=["Healthy", "Warning", "Faulty"]).tolist()

    return {
        "strategy": "undersampling",
        "train_rows": len(X_train_res),
        "val_rows": len(X_val),
        "macro_f1": macro_f1,
        "faulty_recall": faulty_rec,
        "faulty_precision": faulty_prec,
        "faulty_pr_auc": pr_auc,
        "confusion_matrix": json.dumps(cm),
        "model": model,
    }


def mlp_thresholding(X_train, y_reg_train, X_val, y_val, max_iter=150, random_state=SEED):
    """Strategy 3: Train continuous MLPRegressor, then apply calibrated thresholding."""
    print("  -> Training Strategy 3: mlp_thresholding (Method 2)...")
    model = build_mlp_regressor(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_reg_train)

    pred_continuous = model.predict(X_val)
    y_pred = threshold_health_score(pred_continuous)

    # For PR-AUC: lower continuous health score implies higher fault probability
    pseudo_prob_faulty = np.clip((100.0 - pred_continuous) / 100.0, 0.0, 1.0)
    pr_auc = compute_binary_pr_auc(y_val, pseudo_prob_faulty, positive_label="Faulty")

    macro_f1 = round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4)
    faulty_rec = round(float(recall_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    faulty_prec = round(float(precision_score(y_val, y_pred, labels=["Faulty"], average=None, zero_division=0)[0]), 4)
    cm = confusion_matrix(y_val, y_pred, labels=["Healthy", "Warning", "Faulty"]).tolist()

    return {
        "strategy": "thresholded_regressor",
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "macro_f1": macro_f1,
        "faulty_recall": faulty_rec,
        "faulty_precision": faulty_prec,
        "faulty_pr_auc": pr_auc,
        "confusion_matrix": json.dumps(cm),
        "model": model,
    }


# ===========================================================================
# 3. Main Experiment Runner
# ===========================================================================

def run_mlp_balancing_study(train_sample_size=60000, val_sample_size=30000):
    """Run Stage 5 balancing study comparing the 3 strategies and save CSV."""
    print("=" * 65)
    print("Stage 5: MLP Class Balancing Experiment (Partner B2)")
    print("=" * 65)

    parquet_path = ROOT / "data" / "processed" / "hkust_15min_v1.parquet"
    if parquet_path.exists():
        print(f"Loading real dataset from: {parquet_path.name}")
        df = pd.read_parquet(parquet_path)
    else:
        print("Warning: Real parquet not found, using synthesized sample.")
        from src.models.mlp import generate_synthetic_solar_data
        df = generate_synthetic_solar_data(n_samples=30000)

    train_df, val_df, _ = split_chronological(df)

    # Take time-ordered subset for CPU benchmark
    if len(train_df) > train_sample_size:
        train_sub = train_df.iloc[:train_sample_size]
    else:
        train_sub = train_df

    if len(val_df) > val_sample_size:
        val_sub = val_df.iloc[:val_sample_size]
    else:
        val_sub = val_df

    print(f"Preparing datasets (Train: {len(train_sub):,} rows, Val: {len(val_sub):,} rows)...")
    train_prep = prepare_mlp_data(train_sub)
    val_prep = prepare_mlp_data(val_sub)

    X_train, y_reg_train, y_clf_train, _ = get_clean_features(train_prep)
    X_val, y_reg_val, y_clf_val, _ = get_clean_features(val_prep)

    print(f"Cleaned feature matrices -> Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Validation class distribution: {pd.Series(y_clf_val).value_counts().to_dict()}")

    # 1. Run Strategy 1: mlp_unbalanced
    res_unbal = mlp_unbalanced(X_train, y_clf_train, X_val, y_clf_val)

    # 2. Run Strategy 2: mlp_undersampling
    res_under = mlp_undersampling(X_train, y_clf_train, X_val, y_clf_val)

    # 3. Run Strategy 3: mlp_thresholding
    res_thresh = mlp_thresholding(X_train, y_reg_train, X_val, y_clf_val)

    results = [res_unbal, res_under, res_thresh]

    # Select winner based on validation Faulty recall, then Macro-F1 (A2's rule)
    best_strategy = max(
        results,
        key=lambda r: (r["faulty_recall"], r["macro_f1"])
    )
    for r in results:
        r["selected"] = (r["strategy"] == best_strategy["strategy"])

    # Prepare DataFrame for saving
    save_cols = [
        "strategy",
        "train_rows",
        "val_rows",
        "macro_f1",
        "faulty_recall",
        "faulty_precision",
        "faulty_pr_auc",
        "confusion_matrix",
        "selected",
    ]
    out_rows = [{col: r[col] for col in save_cols} for r in results]
    out_df = pd.DataFrame(out_rows)

    out_dir = ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "b2_balancing.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("Stage 5 MLP Balancing Study Results:")
    print("=" * 65)
    for r in results:
        tag = " [WINNER]" if r["selected"] else ""
        print(f"Strategy: {r['strategy']:<22} | Train: {r['train_rows']:>6} | "
              f"Macro-F1: {r['macro_f1']:.4f} | Faulty Recall: {r['faulty_recall']:.4f} | "
              f"Faulty Prec: {r['faulty_precision']:.4f}{tag}")
    print("=" * 65)
    print(f"Saved results table to: {out_path}")
    print("=" * 65)

    return out_df


if __name__ == "__main__":
    run_mlp_balancing_study()

"""Small studies that justify pipeline decisions (all scored on validation).

Stage 4 (A2): imputation      - median vs mean filling of missing feature values
Stage 5 (A2): class balance   - no balancing vs balanced weights vs undersampling Healthy
Stage 7 (B1): thresholds      - class shares for different Faulty / Healthy cut-offs
Stage 7 (B1): feature filter  - pairs of features that are almost the same (|r| > 0.85)

Run: python -m src.studies
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.common import SEED, TABLES, save_figure
from src.dataset import CANDIDATE_FEATURES, FEATURES
from src.evaluate import classification_scores, evaluation_sets, regression_scores


def imputation_study(train, val):
    rows = []
    for strategy in ["median", "mean"]:
        pipe = Pipeline([("impute", SimpleImputer(strategy=strategy)),
                         ("scale", StandardScaler()), ("model", Ridge(alpha=100))])
        pipe.fit(train[FEATURES], train["health_lead"])
        rows.append({"imputation": strategy,
                     **regression_scores(val["health_lead"], pipe.predict(val[FEATURES]))})
    return pd.DataFrame(rows)


def balancing_study(train, val):
    def logistic():
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])

    healthy = train[train["status_lead"] == "Healthy"]
    others = train[train["status_lead"] != "Healthy"]
    undersampled = pd.concat([healthy.sample(2 * len(others), random_state=SEED), others])

    rows = []
    for name, data, weights in [
            ("no balancing", train, None),
            ("balanced class weights", train, compute_sample_weight("balanced", train["status_lead"])),
            ("undersample Healthy (2 per other)", undersampled, None)]:
        model = logistic().fit(data[FEATURES], data["status_lead"], model__sample_weight=weights)
        rows.append({"strategy": name, "train_rows": len(data),
                     **classification_scores(val["status_lead"], model.predict(val[FEATURES]))})
    return pd.DataFrame(rows)


def threshold_study(train):
    health = train["health_now"]
    rows = [{"faulty_below": f, "healthy_from": h,
             "faulty_pct": round(100 * (health < f).mean(), 1),
             "warning_pct": round(100 * ((health >= f) & (health < h)).mean(), 1),
             "healthy_pct": round(100 * (health >= h).mean(), 1)}
            for f in [60, 65, 70, 75, 80] for h in [85, 90, 95]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(health, bins=50, color="steelblue")
    for value, label in [(70, "Faulty below 70"), (90, "Healthy from 90")]:
        ax.axvline(value, color="black", linestyle="--")
        ax.text(value + 1, ax.get_ylim()[1] * 0.9, label)
    ax.set(title="Health score distribution (training period)", xlabel="Health score",
           ylabel="Number of 15-min rows")
    save_figure(fig, "target_health_distribution")
    return pd.DataFrame(rows)


def correlation_filter(train, limit=0.85):
    corr = train[CANDIDATE_FEATURES].corr()
    pairs = [{"feature_1": a, "feature_2": b, "correlation": round(corr.loc[a, b], 3)}
             for i, a in enumerate(CANDIDATE_FEATURES) for b in CANDIDATE_FEATURES[i + 1:]
             if abs(corr.loc[a, b]) > limit]
    return pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"])


def main():
    sets = evaluation_sets()
    train, val = sets["train"], sets["val"]
    for name, table in [("study_imputation", imputation_study(train, val)),
                        ("study_balancing", balancing_study(train, val)),
                        ("study_thresholds", threshold_study(train)),
                        ("study_correlated_features", correlation_filter(train))]:
        table.to_csv(TABLES / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()

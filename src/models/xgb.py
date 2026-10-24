"""Track T3: XGBoost (B1) - boosting: small trees added one by one, each fixing
the errors of the previous ones. Also saves a SHAP summary plot (explainability).

Run: python -m src.models.xgb
"""
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBClassifier, XGBRegressor

from src.common import SEED, save_figure
from src.dataset import FEATURES
from src.evaluate import evaluation_sets, run_model

SETTINGS = dict(n_estimators=300, tree_method="hist", n_jobs=-1, random_state=SEED)
GRID = {"model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1]}


class XGBStatusClassifier(XGBClassifier):
    """XGBoost needs class numbers (0, 1, 2); this turns the status names into
    numbers for training and back into names for predictions."""

    def fit(self, X, y, **kwargs):
        self.labels_, codes = np.unique(y, return_inverse=True)
        return super().fit(X, codes, **kwargs)

    def predict(self, X):
        return self.labels_[super().predict(X).astype(int)]


def shap_summary(model):
    """Which features push the predicted health score up or down (5,000 test rows)."""
    sample = evaluation_sets()["test"].sample(5000, random_state=SEED)
    X = model.named_steps["impute"].transform(sample[FEATURES])
    values = shap.TreeExplainer(model.named_steps["model"]).shap_values(X)
    shap.summary_plot(values, X, feature_names=FEATURES, show=False)
    save_figure(plt.gcf(), "xgb_shap_summary")


def main():
    regressor, _ = run_model("B1", "XGBoost", "regression", XGBRegressor(**SETTINGS), GRID)
    run_model("B1", "XGBoost", "classification", XGBStatusClassifier(**SETTINGS), GRID)
    shap_summary(regressor)


if __name__ == "__main__":
    main()

"""Track T2: Random Forest (A2) - bagging: many deep trees on random samples, averaged.

Run: python -m src.models.rf
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.common import SEED
from src.evaluate import run_model

GRID = {"model__max_depth": [10, 20],
        "model__min_samples_leaf": [5, 20]}


def main():
    run_model("A2", "Random Forest", "regression",
              RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED),
              GRID, tune_rows=50000)
    run_model("A2", "Random Forest", "classification",
              RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
              GRID, tune_rows=50000)


if __name__ == "__main__":
    main()

"""Run the whole project from raw data to final comparison.

Run: python -m src.run_all          (about 15-25 minutes on a laptop)
"""
from src import compare, dataset, eda, prepare_data, studies
from src.models import iforest, linear, mlp, rf, xgb

STEPS = [
    ("Stage 3   data preparation", prepare_data.main),
    ("Stage 4/7 cleaning, target, features, splits", dataset.build),
    ("Stage 6   EDA", eda.main),
    ("Stage 4/5/7 studies", studies.main),
    ("Stage 8-10 linear models and baselines", linear.main),
    ("Stage 9-10 Random Forest", rf.main),
    ("Stage 9-10 XGBoost", xgb.main),
    ("Stage 9-10 MLP", mlp.main),
    ("Stage 9   Isolation Forest", iforest.main),
    ("Stage 11  comparison", compare.main),
]

if __name__ == "__main__":
    for title, step in STEPS:
        print(f"\n===== {title} =====")
        step()

"""Stage 8 baselines + Track T1 linear models (A1).

Baselines (no learning, the bar every model must beat):
  regression      persistence      Health(t+1h) = Health(t)
  classification  majority class   always the most common training class
                  health rule      status implied by Health(t)
                  peer rule        status implied by 100 x (1 + peer_dev): the inverter
                                   compared with the campus median at the same time
Linear models (tuned with the shared recipe in evaluate.run_model):
  Ridge regression, Logistic Regression

Run: python -m src.models.linear
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from src.dataset import status
from src.evaluate import evaluation_sets, run_model, save_results, score_all


def baselines():
    sets = evaluation_sets()
    majority = sets["train"]["status_lead"].mode()[0]
    rules = {
        ("regression", "Persistence"): lambda p: p["health_now"],
        ("classification", "Majority class"): lambda p: np.full(len(p), majority),
        ("classification", "Health rule"): lambda p: p["status_now"],
        ("classification", "Peer rule"): lambda p: status(np.clip(100 * (1 + p["peer_dev"]), 0, 100)),
    }
    for (task, name), predict in rules.items():
        scores = score_all(task, predict, sets)
        save_results("A1", name, task, scores)
        print(f"\n{name} ({task})\n{scores.to_string(index=False)}")


def main():
    baselines()
    run_model("A1", "Ridge", "regression", Ridge(),
              {"model__alpha": [0.01, 1, 100, 10000]}, scale=True)
    run_model("A1", "Logistic Regression", "classification", LogisticRegression(max_iter=2000),
              {"model__C": [0.01, 1, 100]}, scale=True)


if __name__ == "__main__":
    main()

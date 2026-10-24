"""Track T4: Multi-layer perceptron (B2) - a small neural network.

Architecture: inputs (20 scaled features) -> hidden layers (ReLU) -> output
  regression: 1 output, squared-error loss; classification: 3 outputs, log-loss
Optimizer Adam, L2 penalty alpha, early stopping on 10 % of the training rows,
at most 200 epochs, batch size 256. Inputs must be scaled (scale=True).

Run: python -m src.models.mlp
"""
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.common import SEED
from src.evaluate import run_model

SETTINGS = dict(activation="relu", solver="adam", batch_size=256, max_iter=200,
                early_stopping=True, random_state=SEED)
GRID = {"model__hidden_layer_sizes": [(32,), (64, 32)],
        "model__alpha": [0.0001, 0.01]}


def main():
    run_model("B2", "MLP", "regression", MLPRegressor(**SETTINGS), GRID,
              scale=True, tune_rows=50000)
    run_model("B2", "MLP", "classification", MLPClassifier(**SETTINGS), GRID,
              scale=True, tune_rows=50000)


if __name__ == "__main__":
    main()

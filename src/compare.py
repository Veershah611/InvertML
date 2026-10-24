"""Stage 11: final comparison of every model (reads results/results.csv).

Selection rule: choose on validation, confirm once on test.
  regression     -> lowest validation RMSE
  classification -> highest validation Faulty recall (missed faults are the costly error)

Run: python -m src.compare
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.common import RESULTS, TABLES, save_figure

SETS = ["val", "test", "test_before_14feb", "test_after_14feb", "unseen"]


def comparison(results, task, metric):
    table = results[results["task"] == task].pivot(index="model", columns="set", values=metric)
    return table[SETS].sort_values("val", ascending=(task == "regression"))


def bar_chart(table, title, ylabel, name):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    table[["val", "test", "unseen"]].plot.bar(ax=ax, rot=20)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    save_figure(fig, name)


def main():
    results = pd.read_csv(RESULTS / "results.csv")
    rmse = comparison(results, "regression", "rmse")
    f1 = comparison(results, "classification", "macro_f1")
    recall = comparison(results, "classification", "faulty_recall").sort_values("val", ascending=False)
    anomaly = results[results["task"] == "anomaly"].drop(columns=["task", "set"]).dropna(axis=1, how="all")

    for name, table in [("compare_regression_rmse", rmse), ("compare_classification_f1", f1),
                        ("compare_classification_faulty_recall", recall)]:
        table.to_csv(TABLES / f"{name}.csv")
        print(f"\n{name}\n{table.round(3).to_string()}")
    print(f"\nanomaly detection\n{anomaly.to_string(index=False)}")

    bar_chart(rmse, "Health forecast error (lower is better)", "RMSE (health points)", "compare_regression")
    bar_chart(recall, "Faults caught one hour ahead (higher is better)", "Faulty recall",
              "compare_faulty_recall")
    print(f"\nbest regressor (val RMSE): {rmse.index[0]}")
    print(f"best classifier (val Faulty recall): {recall.index[0]}")


if __name__ == "__main__":
    main()

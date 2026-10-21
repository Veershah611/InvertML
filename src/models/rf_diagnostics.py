"""A2 final Random Forest diagnostics after ``python -m src.models.rf``."""

import joblib
import json
import numpy as np
import pandas as pd

from src.evaluate import evaluate_classification, evaluate_regression
from src.preprocess import ROOT, feature_frame

DATA = ROOT / "data" / "processed" / "a2"
TABLES = ROOT / "reports" / "tables"
SENSOR_CHANGE = pd.Timestamp("2023-02-14")


def run():
    test = pd.read_parquet(DATA / "test.parquet")
    manifest = json.loads((DATA / "rf_manifest.json").read_text(encoding="utf-8"))
    reg_name = manifest["selected_regression"].lower().replace(" ", "_")
    cls_name = manifest["selected_classification"].lower().replace(" ", "_")
    reg = joblib.load(DATA / f"{reg_name}.joblib")
    cls = joblib.load(DATA / f"{cls_name}.joblib")
    X = feature_frame(test)
    predicted_score = reg.predict(X)
    predicted_status = cls.predict(X)
    rows = []
    for name, mask in [
        ("before_2023_02_14", test.time.lt(SENSOR_CHANGE)),
        ("from_2023_02_14", test.time.ge(SENSOR_CHANGE)),
    ]:
        part = test.loc[mask]
        scores = evaluate_regression(part.health_score_lead4, predicted_score[mask])
        cls_scores = evaluate_classification(part.status_lead4, predicted_status[mask])
        rows.append({"period": name, "rows": len(part), "rmse": scores["rmse"],
                     "mae": scores["mae"], "r2": scores["r2"],
                     "macro_f1": cls_scores["macro_f1"],
                     "faulty_recall": cls_scores["faulty_recall"],
                     "faulty_prevalence": part.status_lead4.eq("Faulty").mean()})
    pd.DataFrame(rows).to_csv(TABLES / "a2_rf_sensor_change.csv", index=False)
    score_bins = pd.cut(test.health_score_lead4, [-.001, 50, 70, 90, 100.001],
                        labels=["0-50", "50-70", "70-90", "90-100"], right=False)
    residual = pd.DataFrame({"score_range": score_bins,
                             "error": predicted_score - test.health_score_lead4})
    by_range = residual.groupby("score_range", observed=True).agg(
        rows=("error", "size"),
        mae=("error", lambda x: float(np.abs(x).mean())),
        bias=("error", "mean"),
        rmse=("error", lambda x: float(np.sqrt(np.square(x).mean()))),
    ).reset_index()
    by_range.to_csv(TABLES / "a2_rf_error_by_score.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(by_range.to_string(index=False))


if __name__ == "__main__":
    run()

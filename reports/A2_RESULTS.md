# A2 Random Forest and anomaly results

All supervised results forecast B1's engineered health score or operational
status at exactly `t + 1 hour`. Models use B2's chronological partitions and
six available held-out inverter IDs. They were fitted on 260,819 rows through
June 2022. Validation is July–December 2022; the final test is 2023. The
44,634-row unseen-inverter test is a subset of the 2023 test. The labels are
rules based on measured power and irradiance, not verified maintenance faults.

## Choices made without using the test set

- Mean imputation had the higher validation macro-F1 in the fixed preprocessing
  comparison (0.6410 versus 0.6374 for median). The difference is small.
- Class weights and Healthy-class undersampling were compared on the same
  15,000 validation rows after fitting on a 30,000-row training sample.
  Class weights gave Faulty recall **0.6651** and macro-F1 **0.6436**;
  undersampling gave **0.6174** and **0.6736**, respectively. Class weights
  were selected because missing Faulty cases is the stated operational cost.
  The undersampling strategy used 13,626 training rows. SMOTE was not used:
  interpolated operating states may be physically implausible, and any
  resampling would have to remain inside each training fold.
- A 39-input versus 37-input correlation-pruned check was nearly tied:
  regression RMSE 10.056 versus 10.064, and classifier macro-F1 0.6371 versus
  0.6380. The full 39-input set was retained. The check uses training data
  for correlations and a fixed validation sample for performance.
- RandomizedSearchCV examined four parameter combinations on 18,000 sampled
  training rows with three expanding time folds. Both searches chose 60 trees,
  five samples per leaf, 70% of features per split, and no depth limit. Their
  mean CV scores were regression RMSE **10.863** and classification macro-F1
  **0.5900**. Preprocessing was fitted inside each fold; fold boundaries purge
  training examples whose one-hour targets would reach validation.

The "default" label below means the project's **untuned reference
configuration** (80 trees, depth 16, minimum leaf 10, 70% of features), not
scikit-learn's library defaults. The tuned models fit the entire training
partition after search.

## Full-partition comparison

| Model | Train | Validation | 2023 test | Unseen inverter test |
|---|---:|---:|---:|---:|
| RF reference regression, RMSE ↓ | 7.018 | **9.953** | 21.845 | 19.678 |
| RF tuned regression, RMSE ↓ | 5.191 | 10.007 | 21.243 | 18.922 |
| Persistence regression, RMSE ↓ | — | 13.074 | **16.878** | **17.268** |
| RF reference classification, macro-F1 ↑ | 0.8059 | 0.6597 | 0.4615 | 0.3390 |
| RF tuned classification, macro-F1 ↑ | 0.9097 | **0.6674** | 0.3825 | 0.2531 |
| Current-status rule, macro-F1 ↑ | — | 0.5955 | **0.6365** | **0.5225** |
| Majority class, macro-F1 ↑ | — | 0.2947 | 0.1001 | 0.1017 |

The baselines use the **same A2 rows and targets** as the forests. A1's
previously logged scores use different missing-power handling and should not
be placed directly beside these numbers. The persistence baseline predicts
the one-hour health score using current health; the current-status rule uses
current health thresholds. The majority class is learned on training labels.

The reference regression has the lower validation RMSE, and the reference
classifier has the higher validation Faulty recall, **0.6452** versus
**0.6050** for the tuned classifier. Those two reference models are A2's
selected models. The tuned classifier has slightly higher validation macro-F1,
but it does worse on the primary missed-Faulty measure. Tuning also produced
much lower training error without improving the selected validation criteria.
The 2023 scores were not used to reverse these choices.

For the selected classifier, Faulty recall is **0.4061** on the full test and
**0.1972** on unseen inverters. Faulty precision is 0.7588 and 0.5257, and
Faulty-versus-rest average precision (PR-AUC) is 0.7834 and 0.5065. The
classification confusion matrix is stored in `tables/a2_rf_results.csv`,
with rows and columns ordered **Faulty, Healthy, Warning**. The selected
regressor's test MAE is 17.821 and R² is -0.149; unseen-inverter MAE is
15.696 and R² is -0.429. The negative R² shows poor 2023 generalization.

The simpler baselines outperform the selected forests on 2023 data. The RF
models should not be recommended for deployment from these results.

## Distribution shift and failure analysis

A1 documented an irradiance sensor change on **14 February 2023**. Splitting
the full 2023 test at that date gives:

| Period | Rows | Faulty prevalence | Selected RF RMSE | RF macro-F1 | Faulty recall |
|---|---:|---:|---:|---:|---:|
| Before 14 February | 34,891 | 9.91% | 9.216 | 0.7051 | 0.7609 |
| From 14 February | 406,314 | 48.94% | 22.603 | 0.4086 | 0.3999 |

This change is consistent with a sensor or target-definition shift, but a
before/after association does not prove the sensor alone caused the decline.
Because irradiance enters B1's label formula and model inputs, sensor
calibration needs investigation before treating the post-change Faulty rate as
physical inverter failures. The exact period scores are in
`tables/a2_rf_sensor_change.csv`.

For true health scores 0–50 in 2023, the selected regressor's MAE is **36.0**
and mean signed error is **+35.9**: it systematically predicts too high. For
90–100, MAE is **5.57** and signed error **-4.90**. The worst sampled cases
include a current score of 100 followed by a future score of zero, for which
the model predicted 99.3 and Healthy. Three more examples predict Healthy for
future Faulty readings near zero. These may represent abrupt outages, sensor
events, or other transitions; the available labels do not resolve the cause.
See `tables/a2_rf_error_by_score.csv` and `tables/a2_rf_failure_cases.csv`.

## Explanations

Permutation importance was measured on 2,500 validation rows using the
selected models and the task score. The largest regression RMSE increases
after shuffling were campus peer deviation (**2.54**), its one-day rolling
mean (**1.06**), and current performance ratio (**0.89**). The largest
classifier macro-F1 decreases were current performance ratio (**0.0397**),
one-day rolling campus peer deviation (**0.0191**), and hour sine (**0.0166**).
These are model reliance measures on the validation period, not causal effects.

TreeSHAP on 80 validation rows also ranks performance ratio and campus peer
deviation highly. For Faulty-class probability, the top mean absolute SHAP
values are 0.0703 for performance ratio and 0.0471 for current campus peer
deviation. Local contributions for three validation examples are in
`tables/a2_shap_local.csv`. Figures are `figures/a2_shap_regression.png` and
`figures/a2_shap_classification.png`. The SHAP sample is small and explanatory;
it is not an evaluation estimate.

## Synthetic anomaly check

An Isolation Forest was fitted on 30,000 training rows labeled Healthy by
the project's rule and evaluated with B2's injected 35% power drops and
complete outages. In the test exercise it detected **36 of 108 injected
rows (33.33%)** and flagged **2,396 of 12,345 control rows (19.41%)**.
This is a weak result on synthetic faults, not evidence of real-world fault
detection. A1's joint review remains pending. The raw counts are in
`tables/a2_iforest_injected_faults.csv`.

## Reproduce and inspect

Run preprocessing first, then `python -m src.models.rf` for all fits and
explanations, `python -m src.models.rf_diagnostics` for the sensor and score
breakdowns, and `python -m src.models.iforest_a2` for the anomaly check.
`python -m src.models.rf --resume` regenerates selected-model explanations
from saved local model files without retraining. The data and fitted models in
`data/processed/a2/` are ignored by Git. The source, reports, figures and CSV
tables are local project files. No changes were pushed as part of this work.

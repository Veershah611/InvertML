# A2 Random Forest and anomaly results

> **Updated 2026-10-21.** Re-run after two project-wide fixes to B1's target: irradiance after the 2023-02-14 sensor change is rescaled in `prepare_data.py` (factor 0.570), and missing AC power now gives an undefined health score instead of 0. All numbers below are from that re-run; the modelling choices are unchanged.

All supervised results forecast B1's engineered health score or operational
status at exactly `t + 1 hour`. Models use B2's chronological partitions and
six available held-out inverter IDs. They were fitted on 260,819 rows through
June 2022. Validation is July–December 2022; the final test is 2023. The
37,060-row unseen-inverter test is a subset of the 2023 test. The labels are
rules based on measured power and irradiance, not verified maintenance faults.

## Choices made without using the test set

- Mean imputation had the higher validation macro-F1 in the fixed preprocessing
  comparison (0.6406 versus 0.6382 for median). The difference is small.
- Class weights and Healthy-class undersampling were compared on the same
  15,000 validation rows after fitting on a 30,000-row training sample.
  Class weights gave Faulty recall **0.6637** and macro-F1 **0.6443**;
  undersampling gave **0.6263** and **0.6774**, respectively. Class weights
  were selected because missing Faulty cases is the stated operational cost.
  The undersampling strategy used 13,626 training rows. SMOTE was not used:
  interpolated operating states may be physically implausible, and any
  resampling would have to remain inside each training fold.
- A 39-input versus 37-input correlation-pruned check was nearly tied:
  regression RMSE 10.055 versus 10.064, and classifier macro-F1 0.6386 versus
  0.6372. The full 39-input set was retained. The check uses training data
  for correlations and a fixed validation sample for performance.
- RandomizedSearchCV examined four parameter combinations on 18,000 sampled
  training rows with three expanding time folds. Both searches chose 60 trees,
  five samples per leaf, 70% of features per split, and no depth limit. Their
  mean CV scores were regression RMSE **10.911** and classification macro-F1
  **0.5877**. Preprocessing was fitted inside each fold; fold boundaries purge
  training examples whose one-hour targets would reach validation.

The "default" label below means the project's **untuned reference
configuration** (80 trees, depth 16, minimum leaf 10, 70% of features), not
scikit-learn's library defaults. The tuned models fit the entire training
partition after search.

## Full-partition comparison

| Model | Train | Validation | 2023 test | Unseen inverter test |
|---|---:|---:|---:|---:|
| RF reference regression, RMSE ↓ | 7.015 | **9.913** | **9.731** | **9.073** |
| RF tuned regression, RMSE ↓ | 5.185 | 9.987 | 9.896 | 9.299 |
| Persistence regression, RMSE ↓ | — | 13.074 | 12.350 | 10.728 |
| RF reference classification, macro-F1 ↑ | 0.8051 | 0.6534 | 0.6122 | **0.5020** |
| RF tuned classification, macro-F1 ↑ | 0.9099 | **0.6697** | **0.6224** | 0.5008 |
| Current-status rule, macro-F1 ↑ | — | 0.5955 | 0.5922 | 0.4728 |
| Majority class, macro-F1 ↑ | — | 0.2947 | 0.3018 | 0.3232 |

The baselines use the **same A2 rows and targets** as the forests. A1's
previously logged scores use different missing-power handling and should not
be placed directly beside these numbers (A1's test set has about 1% more rows;
see `reports/Final_Model_Comparison.md`). The persistence baseline predicts
the one-hour health score using current health; the current-status rule uses
current health thresholds. The majority class is learned on training labels.

The reference regression has the lower validation RMSE, and the reference
classifier has the higher validation Faulty recall, **0.6226** versus
**0.6195** for the tuned classifier. Those two reference models are A2's
selected models. The tuned classifier has slightly higher validation macro-F1,
but it does worse on the primary missed-Faulty measure. Tuning also produced
much lower training error without improving the selected validation criteria.
The 2023 scores were not used to reverse these choices.

For the selected classifier, Faulty recall is **0.5708** on the full test and
**0.3285** on unseen inverters. Faulty precision is 0.6279 and 0.4564, and
Faulty-versus-rest average precision (PR-AUC) is 0.6733 and 0.3248. The
classification confusion matrix is stored in `tables/a2_rf_results.csv`,
with rows and columns ordered **Faulty, Healthy, Warning**. The selected
regressor's test MAE is 5.044 and R² is 0.627; unseen-inverter MAE is
3.629 and R² is 0.372.

On the corrected target the selected forests beat the simpler baselines on
2023 data: regression RMSE 9.73 against persistence 12.35, and classification
macro-F1 0.612 against the current-status rule 0.592 (0.502 against 0.473 on
unseen inverters). The reference regressor is the project's selected health
forecaster (`reports/Final_Model_Comparison.md`).

## Distribution shift and failure analysis

A1 documented an irradiance sensor change on **14 February 2023**. Splitting
the full 2023 test at that date gives:

| Period | Rows | Faulty prevalence | Selected RF RMSE | RF macro-F1 | Faulty recall |
|---|---:|---:|---:|---:|---:|
| Before 14 February | 34,891 | 9.91% | 9.076 | 0.7031 | 0.7679 |
| From 14 February | 331,219 | 8.74% | 9.797 | 0.6018 | 0.5472 |

Before the correction, 48.94% of post-change rows were labelled Faulty and the
RF RMSE rose to 22.603 after the change. With irradiance rescaled after
2023-02-14, the Faulty prevalence is back in line with earlier periods and
the error gap is small. Some decline in classification after the change
remains; one correction factor does not fit every month equally well
(A1, `reports/tables/irradiance_correction.csv`). The exact period scores are in
`tables/a2_rf_sensor_change.csv`.

For true health scores 0–50 in 2023, the selected regressor's MAE is **26.5**
and mean signed error is **+25.9**: it systematically predicts too high. For
90–100, MAE is **2.73** and signed error **-2.51**. The worst sampled cases
include a current score of 100 followed by a future score of zero, for which
the model predicted 99.7 and Healthy (Zone A6, 2023-11-21). Four more examples
predict Healthy for future Faulty readings of zero. These may represent abrupt outages, sensor
events, or other transitions; the available labels do not resolve the cause.
See `tables/a2_rf_error_by_score.csv` and `tables/a2_rf_failure_cases.csv`.

## Explanations

Permutation importance was measured on 2,500 validation rows using the
selected models and the task score. The largest regression RMSE increases
after shuffling were campus peer deviation (**2.28**), its one-day rolling
mean (**1.12**), and current performance ratio (**1.03**). The largest
classifier macro-F1 decreases were one-day rolling campus peer deviation
(**0.0270**), current performance ratio (**0.0216**), and hour sine (**0.0120**).
These are model reliance measures on the validation period, not causal effects.

TreeSHAP on 80 validation rows also ranks performance ratio and campus peer
deviation highly. For Faulty-class probability, the top mean absolute SHAP
values are 0.0718 for performance ratio and 0.0407 for current campus peer
deviation. Local contributions for three validation examples are in
`tables/a2_shap_local.csv`. Figures are `figures/a2_shap_regression.png` and
`figures/a2_shap_classification.png`. The SHAP sample is small and explanatory;
it is not an evaluation estimate.

## Synthetic anomaly check

An Isolation Forest was fitted on 30,000 training rows labeled Healthy by
the project's rule and evaluated with B2's injected 35% power drops and
complete outages. In the test exercise it detected **64 of 137 injected
rows (46.72%)** and flagged **2,042 of 38,842 control rows (5.26%)**.
This is a modest result on synthetic faults, not evidence of real-world fault
detection. A1 reviewed it and extended it in `src/models/iforest_a1.py`
(tuning on validation, a peer-deviation input); see
`reports/Final_Model_Comparison.md`, section 4. The raw counts are in
`tables/a2_iforest_injected_faults.csv`.

## Reproduce and inspect

Run preprocessing first, then `python -m src.models.rf` for all fits and
explanations, `python -m src.models.rf_diagnostics` for the sensor and score
breakdowns, and `python -m src.models.iforest_a2` for the anomaly check.
`python -m src.models.rf --resume` regenerates selected-model explanations
from saved local model files without retraining. The data and fitted models in
`data/processed/a2/` are ignored by Git.

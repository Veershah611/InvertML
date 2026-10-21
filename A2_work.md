# A2 Work Guide: Solar Inverter Health Monitoring

This guide explains your assignment and the work implemented through
21 October 2026. The final model metrics are in `reports/A2_RESULTS.md`.

## 1. Your role

You are **A2: preprocessing, class balancing, and Random Forest modeling**.
You belong to Pair A with A1.

Your main modeling question is:

> Given an inverter's readings now, what will its health score and operational
> status be exactly one hour later?

| Member | Main responsibility | Supervised model family |
|---|---|---|
| A1 | Data preparation and baselines | Ridge and Logistic Regression |
| A2 — you | Preprocessing and class balancing | Random Forest |
| B1 | Target construction and feature engineering | XGBoost |
| B2 | Evaluation framework and time splits | MLP |

Each member also contributes EDA, features, model explanation, and anomaly work.

## 2. Your assignment and current status

| Responsibility | What you must deliver | Current status |
|---|---|---|
| Preprocessing | Cleaning, missing-value handling, encoding, scaling, comparison of imputers | Complete |
| Class balancing | Compare at least two strategies; discuss SMOTE | Compared class weights and undersampling; interpretation below |
| EDA | Temperature effect and feature correlation | Completed; see `reports/A2_EDA.md` |
| Feature engineering | Lag and rolling statistics | Implemented and tested |
| Regression | Random Forest predicting future health score | Complete; see `reports/A2_RESULTS.md` |
| Classification | Random Forest predicting future status | Complete; see `reports/A2_RESULTS.md` |
| Tuning | RandomizedSearch with time-series cross-validation | Completed on training samples |
| Explainability | Permutation importance, SHAP, failure cases | Complete; see `reports/A2_RESULTS.md` |
| Anomaly detection | Isolation Forest on HKUST injected faults with A1 | A2 benchmark run; A1 review pending |
| Write-up | Data-leakage audit and explanations of your decisions | A2 audit written in `reports/A2_LEAKAGE_AUDIT.md` |

The smaller validation runs compare preprocessing choices. The full model run
is a separate experiment using all 260,819 training rows.

## 3. Understand B1's targets

### Health-score regression

B1 calculates a health score from actual power relative to expected power:

```text
expected_power_w = max(irradiance_wm2 * capacity_kwp * 0.78, 100)
health_score = clip(100 * ac_power_w / expected_power_w, 0, 100)
```

The factor `0.78` is the project's assumed derating factor. The 100 W floor
limits division by very small expected-power values.

Example:

- Irradiance: 500 W/m².
- Array capacity: 10 kWp.
- Expected power: 500 × 10 × 0.78 = 3,900 W.
- Actual power: 3,120 W.
- Health score: 100 × 3,120 / 3,900 = 80.

### Operational-status classification

| Health score | Status |
|---|---|
| 90–100 | Healthy |
| 70–less than 90 | Warning |
| Below 70 | Faulty |

The example above receives the label **Warning**.

These are engineered labels, not confirmed maintenance records. A Faulty label
means the observation meets B1's rule; it does not prove a hardware fault.
Health is undefined below 100 W/m² irradiance or when required measurements
are unavailable.

### Your prediction horizon

Your inputs come from time `t`. Your targets come from exactly `t + 1 hour`:

| Column | Meaning |
|---|---|
| `health_score` | Observed health at the current timestamp |
| `status` | Current operational status |
| `health_score_lead4` | Health exactly one hour later |
| `status_lead4` | Status exactly one hour later |
| `target_time` | Timestamp of the future observation |

Four 15-minute intervals make one hour. However, filtering can remove rows, so
the implementation joins by exact timestamp rather than assuming that the fourth
remaining row is one hour later.

At 10:00, the model can use the available 10:00 readings to predict 11:00 health.
It cannot use the 11:00 readings as inputs.

## 4. Task 1: Preprocess the data

Implementation: [src/preprocess.py](src/preprocess.py).

A1 supplies `data/processed/hkust_15min_v1.parquet`, containing 1,588,901 rows
and 45 columns. It combines inverter measurements, weather and metadata.

Your preprocessing makes these observations suitable for modeling.

### Physical validity

| Condition | Current handling |
|---|---|
| Negative power | Exclude from model rows |
| Power above `1.1 * capacity_kwp * 1000` | Exclude from model rows |
| Irradiance outside 0–1,500 W/m² | Mask the invalid measurement; exclude from daylight modeling |
| Missing or nonpositive capacity | Exclude from model rows |
| Invalid weather values | Mask before model-input imputation |
| Extreme values within the chosen bounds | Retain |

The capacity limit is a screening proxy based on array DC capacity. It is not
a separately verified inverter AC rating. A1's metadata concerns remain relevant.

The latest run flagged 3,209 invalid power readings and 218 invalid irradiance
readings. Flag counts can overlap.

### Daylight and missing observations

Cleaning retains observations from 50 W/m² irradiance, following A1's EDA.
B1's labels use a stricter 100 W/m² threshold. A row may therefore appear in
the daylight table without receiving a health label.

Missing power remains missing. The complete cleaned grid preserves 94,441
missing-power/zero-valid-sample events with an `is_offline` flag. This flag
describes unavailable telemetry, not a confirmed inverter failure.

Observed zero power is different from missing power: a real zero reading can
receive a Faulty label when irradiance is sufficient.

The original design suggests interpolation for short gaps. This initial
forecasting implementation leaves source gaps intact because interpolation from
a later observation can leak future information. Missing model inputs are handled
separately with training-fitted imputation. Missing target labels are never filled.

### Imputation, encoding and scaling

- **Imputation:** replace missing model inputs using training means or medians.
- **Encoding:** convert roof azimuth categories into numerical indicator columns.
- **Scaling:** apply StandardScaler where a model benefits from comparable scales.

Random Forest uses the unscaled transformer. A scaled transformer is also saved
for linear or neural models. Inverter and station IDs remain metadata and are
excluded from the model-input allowlist.

Both transformers learn all statistics and categories from training rows only.
Unknown categories at inference are handled without fitting on test data.

## 5. B2's splits and leakage controls

Shared split functions are in [src/evaluate.py](src/evaluate.py).

| Partition | Dates | Forecast-ready rows | Inverters |
|---|---|---:|---:|
| Training | Through 30 June 2022 | 260,819 | 36 |
| Validation | 1 July–31 December 2022 | 182,731 | 38 |
| Test | 1 January–31 December 2023 | 441,205 | 44 |
| Unseen test | Held-out inverters during 2023 | 44,634 | 6 |

Training teaches the model. Validation helps choose approaches and settings.
The test set evaluates the final chosen approach.

B2 lists seven holdout IDs, but `sq14_inverter` is absent from the data. Six IDs
match. The unseen test is a subset of the full test. It is an inverter-level
holdout, not a guaranteed whole-station holdout.

Current safeguards:

- Held-out inverter IDs are excluded from training and validation.
- Forecast labels are joined inside each split and cannot cross its boundary.
- Imputation, scaling and encoding fit training rows only.
- Time-series CV splits unique timestamps, keeping simultaneous inverters together.
- Training examples whose future labels reach the next validation fold are removed.
- For CV, a fresh preprocessor must be fitted inside each model Pipeline.

Example: if a validation fold begins at 10:00, a training example at 09:30 whose
label is measured at 10:30 must be removed from that fold's training set.

## 6. B1 features already integrated

Shared implementation: [src/features.py](src/features.py).

| Feature | Meaning |
|---|---|
| `specific_yield` | Current AC power divided by array capacity |
| `performance_ratio` | Actual power divided by expected power |
| `peer_dev_campus` | Deviation from the simultaneous campus median PR |
| `peer_dev_station` | Deviation from the simultaneous station median PR |
| `self_dev_30d` | Deviation from the inverter's preceding 30-day PR baseline in the same irradiance bin |
| `thermal_stress` | Irradiance × ambient temperature interaction |
| `temp_delta_stc` | Ambient temperature minus 25°C |

Integration restored the B2 functions/constants used by A1/B2 models, preserved
missing physical measurements, and corrected the incoming 200-row baseline to
an actual 30-calendar-day baseline.

Training/validation peer statistics exclude held-out inverters. Test features use
the simultaneous test telemetry; unseen-test rows are sliced from that feature
table. Only preceding observations contribute to self-history. This assumes
sequential predictions with current telemetry available.

These are B1's features. Your separate lag-and-rolling contribution is
implemented in `add_a2_history_features()` and tested for timestamp alignment.
All seven B1 inputs remain in the final model after a validation check of the
39-input and 37-input alternatives.

## 7. Task 2: Answer your EDA questions

### Question A: How does ambient temperature relate to performance?

Examine temperature against normalized performance while accounting for irradiance
and differences between inverters. Sunny conditions can raise both temperature
and power, so temperature versus raw power alone can be misleading.

The completed analysis is in `reports/A2_EDA.md`. Across 36 inverters, 26 have
positive and 10 negative temperature/PR slopes in the filtered daytime sample.
The pattern does not establish a physical temperature effect.

### Question B: Which features are redundant?

Use a correlation matrix on training data to identify strongly related inputs.
Review B1's pruning logic and compare validation performance before and after
selection. Correlation alone does not establish that a feature is useless.

The correlation figure and validation comparison are in `reports/A2_EDA.md`.
Two correlated inputs could be dropped with almost no score change, so all 39
were retained for the main comparison.

## 8. Task 3: Build lag and rolling features

Build the assigned historical features for PR and peer deviation:

| Feature | Meaning |
|---|---|
| Lag `t−1` | Value 15 minutes earlier |
| Lag `t−4` | Value one hour earlier |
| Lag `t−96` | Value 24 hours earlier on a complete 15-minute grid |
| 1-hour rolling mean/std | Recent average and variability |
| 4-hour rolling mean/std | Longer short-term pattern |
| 1-day rolling mean/std | Recent daily pattern |

Use elapsed-time checks: 96 remaining logger rows are not necessarily 24 hours
because overnight observations are absent. Group by inverter, keep windows
backward-looking, and preserve missing history.

Test that changing future observations cannot change an earlier feature.

## 9. Task 4: Study class imbalance

Class counts differ, so high overall accuracy can hide poor detection of Faulty
examples. Compare at least two training strategies:

| Strategy | Approach |
|---|---|
| Class weights | Give less common classes more weight during fitting |
| Random undersampling | Reduce majority-class training examples |
| SMOTE | Generate synthetic minority examples; assess and justify suitability |

Use validation macro-F1 and Faulty recall for the required comparison. Also
inspect precision and the confusion matrix to understand false alarms.

Apply resampling only inside training folds. Keep validation and test distributions
unchanged. Regression needs error analysis across health-score ranges rather than
class balancing.

The comparison is in `reports/tables/a2_balancing.csv`. Class weights gave
Faulty recall 0.6637; undersampling gave 0.6263 on the same validation rows.
Undersampling had higher macro-F1 (0.6774 versus 0.6443; re-run 2026-10-21
on the corrected target). The main classifier
uses class weights because missed Faulty cases are the stated priority. SMOTE
was assessed but not applied: interpolating time-adjacent states can produce
implausible telemetry, and any resampling must stay inside training folds.

## 10. Task 5: Build and tune Random Forest

Train both:

- `RandomForestRegressor` against `health_score_lead4`.
- `RandomForestClassifier` against `status_lead4`.

A forest combines multiple decision trees. Regression averages numerical
predictions; classification combines class predictions/probabilities.

Start with an initial model, then use RandomizedSearch on a manageable training
sample with time-series CV. Candidate settings include number of trees, tree
depth, minimum leaf size and number of features considered per split.

After choosing settings, fit the final pipeline on the full training partition
and evaluate using the team's shared evaluation conventions.

| Evaluation | What it tells you |
|---|---|
| MAE | Average absolute health-score error |
| RMSE | Error measure that penalizes large misses more |
| R² | Performance relative to a constant-mean reference |
| Macro-F1 | Average F1 giving each class equal importance |
| Faulty recall | Fraction of Faulty labels detected |
| PR-AUC | Precision/recall behavior across probability thresholds; state the class/averaging convention |
| Confusion matrix | Which classes are mistaken for others |

Required deliverables include default versus tuned results, train versus test
scores, comparison with A1's baselines, unseen-inverter results and 2–3 failure
cases. Use matching targets and evaluation rows for fair model comparisons.

## 11. Task 6: Explain models and evaluate anomalies

For Random Forest:

- **Permutation importance:** measure the effect of shuffling each feature.
- **SHAP:** explain how features contribute to particular predictions.
- **Failure cases:** investigate large errors with their weather, missingness
  and recent-history context.

For anomaly detection, work with A1 on Isolation Forest using HKUST data with
injected faults. Measure detection performance against the injected scenario
labels. Report that this is a synthetic-fault benchmark.

## 12. Task 7: Write the data-leakage audit

Explain how each safeguard prevents information unavailable at prediction time
from influencing training or evaluation.

Cover chronological splitting, exact forecast horizons, training-only transforms,
fold-specific feature selection/resampling, backward-looking history, exclusion
of future targets from inputs, and held-out inverter isolation.

The completed audit is in `reports/A2_LEAKAGE_AUDIT.md`. It records A2 controls
and code-level follow-ups for other members' pipelines.

## 13. Current validation results

Two fixed Random Forest probes used the same 30,000 training rows and 15,000
validation rows. Test scores were not used to choose imputation.

| Imputation | Validation RMSE | Validation MAE | Macro-F1 | Faulty recall |
|---|---:|---:|---:|---:|
| Median, before A2 history features | 10.350 | 5.874 | 0.6272 | 0.6780 |
| Mean, before A2 history features | 10.351 | 5.878 | 0.6246 | 0.6746 |
| Median, with A2 history features (re-run 2026-10-21) | 10.111 | 5.617 | 0.6382 | 0.6596 |
| Mean, with A2 history features (re-run 2026-10-21) | 10.120 | 5.620 | 0.6406 | 0.6678 |

After adding history features, **mean** was selected using macro-F1, with RMSE
as a tie-breaker. The differences are small and are not a
statistical-significance claim.

Before B1 feature integration, the median probe had RMSE 11.708, macro-F1 0.5558
and Faulty recall 0.4908 on the same sampled rows.

Seventeen automated tests passed after adding history features. Real-data
verification checked saved training
statistics, exact forecast horizons, finite transformed samples and matching
features between unseen-test rows and their counterparts in the full test.

## 14. Files and run instructions

Run these commands from the `InvertML` repository root:

```powershell
# Rebuild preprocessing outputs and compare imputers.
.\.venv\Scripts\python.exe -m src.preprocess

# Recreate EDA, train/evaluate Random Forest, and analyze the 2023 shift.
.\.venv\Scripts\python.exe -m src.eda_a2
.\.venv\Scripts\python.exe -m src.models.rf
.\.venv\Scripts\python.exe -m src.models.rf_diagnostics

# Run the synthetic injected-fault benchmark.
.\.venv\Scripts\python.exe -m src.models.iforest_a2

# Run the existing preprocessing and integration tests.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The commands use the existing local environment. A new checkout needs its Python
environment and A1's prepared data first; see [README.md](README.md).

| Existing file or directory | Purpose |
|---|---|
| [src/preprocess.py](src/preprocess.py) | A2 cleaning, splits, transforms and imputation probes |
| [src/features.py](src/features.py) | Shared B1/B2 feature functions |
| [src/targets.py](src/targets.py) | B1 target definitions |
| [src/evaluate.py](src/evaluate.py) | B2 splitting and evaluation helpers |
| [configs/config.yaml](configs/config.yaml) | Project settings |
| [reports/A2_PREPROCESSING.md](reports/A2_PREPROCESSING.md) | Detailed implementation report |
| [tests/test_preprocess.py](tests/test_preprocess.py) | Preprocessing safeguards |
| [tests/test_feature_integration.py](tests/test_feature_integration.py) | Shared-feature integration safeguards |
| [tests/test_a2_history.py](tests/test_a2_history.py) | Exact-time lag and past-only rolling checks |
| [src/eda_a2.py](src/eda_a2.py) | Temperature/correlation plots and summaries |
| [reports/A2_EDA.md](reports/A2_EDA.md) | EDA results and decisions |
| [reports/A2_LEAKAGE_AUDIT.md](reports/A2_LEAKAGE_AUDIT.md) | A2 leakage review |
| [reports/A2_RESULTS.md](reports/A2_RESULTS.md) | Final RF, baseline, shift, explanation and anomaly results |
| [src/models/rf.py](src/models/rf.py) | A2 balancing, tuning and Random Forest experiments |
| [src/models/rf_diagnostics.py](src/models/rf_diagnostics.py) | Sensor-change and score-range error analysis |
| [src/models/iforest_a2.py](src/models/iforest_a2.py) | HKUST injected-fault anomaly benchmark |
| `reports/tables/a2_preprocessing_*.csv` | Audit, split, CV and comparison tables |
| `data/processed/a2/` | Generated datasets, fitted transformers and manifest |

Generated datasets include `cleaned_grid.parquet`, `daylight.parquet`,
`train.parquet`, `val.parquet`, `test.parquet` and `test_unseen.parquet`.
Model-input columns remain unscaled in the tables so each CV fold can fit its
own preprocessing. Full-training transformers are saved separately.

Data and `.venv` are git-ignored. The preprocessing code was submitted in
[PR #11](https://github.com/Veershah611/InvertML/pull/11); this guide is a separate
local addition.

## 15. Remaining work checklist

- [x] Implement initial preprocessing.
- [x] Use B2's chronological and unseen-inverter split functions.
- [x] Preserve missing target measurements and enforce exact one-hour labels.
- [x] Compare mean and median model-input imputation.
- [x] Integrate B1's physical features and verify compatibility.
- [x] Save reports and test the implementation.
- [x] Complete temperature EDA.
- [x] Complete correlation EDA and evaluate feature selection.
- [x] Implement A2's lag and rolling feature group.
- [x] Compare class weights and undersampling; discuss SMOTE limits.
- [x] Train both full Random Forest models.
- [x] Tune with RandomizedSearch and time-series CV.
- [x] Compare baselines, tuned results and unseen-inverter results.
- [x] Produce permutation importance, SHAP and failure-case analysis.
- [x] Run A2's Isolation Forest injected-fault benchmark; A1 review remains pending.
- [x] Write the A2 leakage audit; other members' pipelines remain outside its scope.

See `reports/A2_EDA.md` for findings. The temperature/PR association is mixed;
the 39-to-37-feature pruning check has nearly tied validation scores across
the two tasks, so all inputs remain in the main model.

The balancing comparison on the same validation rows gave macro-F1 0.6443 and
Faulty recall 0.6637 with class weights, versus 0.6774 and 0.6263 with
undersampling. The main classifier uses class weights because catching Faulty
cases is the project's stated priority. SMOTE is not used: interpolating nearby
time-series rows can produce implausible synthetic operating states and would
need fold-specific resampling.

The A2 Isolation Forest benchmark detected 64 of 137 injected-fault rows and
flagged 2,042 of 38,842 control rows. That is 46.72% detection and a 5.26%
false-alarm rate on the corrected target; it is not strong evidence for
deployment. A1 reviewed the paired benchmark and extended it in
`src/models/iforest_a1.py` (validation tuning, peer-deviation input).

The selected untuned RF regressor has validation RMSE 9.913 and 2023 test RMSE
9.731. The selected untuned classifier has validation macro-F1 0.6534,
Faulty recall 0.6226; on the 2023 test these are 0.6122 and 0.5708.
The tuned variants did not improve the validation criterion used for selection.
On the corrected target the forests beat the persistence and current-status
baselines on 2023 data, and the RF regressor is the project's selected health
forecaster. See `reports/A2_RESULTS.md` and `reports/Final_Model_Comparison.md`.
(Before the 2026-10-21 target fix, test RMSE was 21.845 and the baselines won.)

## 16. Questions to prepare for your viva

**Why use time splits?** The intended prediction uses past information to predict
future observations. Random mixing can give an unrealistically easy evaluation.

**Why fit preprocessing only on training data?** Otherwise future validation/test
distributions influence the values and categories learned during training.

**Why does Random Forest use the unscaled transformer?** Trees split using feature
thresholds and generally do not require comparable feature scales.

**Why mean imputation?** With history features, mean gave higher validation
macro-F1 than median (0.6406 versus 0.6382), so the final A2 models use mean.
The difference is small and limited to this comparison.

**Why leave missing power unlabeled?** Absence of telemetry does not tell us the
actual generated power or prove a hardware failure.

**Why is the target one hour ahead?** It tests forecasting ability and avoids
simply reconstructing a current label from its defining inputs.

**Why keep all inverters at one timestamp in the same CV fold?** It preserves a
clear chronological boundary across the simultaneous campus observations.

**What are the main limitations?** Labels are engineered; missing telemetry is
not evaluated as a supervised forecast target; the holdout is at inverter level;
and A1's documented February 2023 irradiance sensor change is corrected with
one factor estimated from test-year data.

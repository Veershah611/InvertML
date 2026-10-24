# Solar Inverter Health Monitoring: Project Report

Every number in this report comes from one full run of `python -m src.run_all`. The tables are in `reports/tables/`, the figures in `reports/figures/`, and all model scores in `results/results.csv`. Stage numbers follow the 12-stage pipeline of the Innovative Assignment guidelines.

| Stage | Where | Lead |
|---|---|---|
| 1 Problem definition | Section 1 | A1 |
| 2 Data acquisition and understanding | Section 2 | A1 |
| 3 Data preparation | Section 3, `src/prepare_data.py` | A1 |
| 4 Data preprocessing | Section 4, `src/dataset.py`, `src/studies.py` | A2 |
| 5 Data balancing | Section 5, `src/studies.py` | A2 |
| 6 Exploratory data analysis | Section 6, `src/eda.py` | all four |
| 7 Features, target, selection | Section 7, `src/dataset.py`, `src/studies.py` | B1 |
| 8 Baselines | Section 8, `src/models/linear.py` | A1 |
| 9 Models | Section 9, `src/models/` | all four |
| 10 Hyperparameter tuning | Section 10, `src/evaluate.py` | all four |
| 11 Validation and evaluation | Section 11, `src/evaluate.py`, `src/compare.py` | B2 |
| 12 Use case, impact, limitations | Section 12 | B2 |

---

## 1. Problem definition

Inverters are the most failure-prone part of a solar system. When one degrades or trips, the lost energy often goes unnoticed, and low output alone does not show whether a cloud or a fault caused it.

**Goal.** For each inverter every 15 minutes:
1. a **health score (0-100)** forecast one hour ahead,
2. a **status**: Healthy / Warning / Faulty,
3. an **anomaly flag** from an unsupervised detector.

**Users.** The campus operations team, who need a ranked list of inverters to inspect instead of reading yield reports by hand.

**Constraints.** No maintenance or fault records exist for these inverters, so the labels are engineered (Section 7). One weather station serves all rooftops. The data is time-ordered, so evaluation must respect time. Everything runs on a CPU laptop.

## 2. Data acquisition and understanding

| Dataset | Source and licence | Use |
|---|---|---|
| HKUST rooftop PV, 2021-2023 | Lin et al., Scientific Data (2025); Dryad doi:10.5061/dryad.m37pvmd99; public domain (CC0) | everything |
| GPVS-Faults | Bakdi et al., Mendeley doi:10.17632/n76t439f65.1; CC BY 4.0 | real-fault check of the anomaly detector only |

**HKUST.** 60 rooftop stations; inverter-level data every 5 minutes exists only for the **44 inverters at 37 stations** that have panel optimisers, so those are used. Weather comes every minute from one campus station. Station metadata (capacity, tilt, orientation, altitude) is a Brick `.ttl` graph, read with SPARQL.

**What is missing.** DC current/power and module temperature are not recorded. Loggers only record 06:00-17:59, so summer evening output is never seen. Start dates vary from May 2021 to mid-2023.

**GPVS-Faults.** 16 lab recordings of a grid-connected PV system: F0 without fault and F1-F7 with seven fault types, each in two control modes.

## 3. Data preparation (`src/prepare_data.py`)

1. **Metadata:** one SPARQL query turns the graph into one row per inverter. Capacity = sum of (number of panels x panel watts). One syntax error in the raw `.ttl` file (Zone D) is fixed in memory; one file name is matched to its metadata name by hand.
2. **Resampling:** 5-minute AC power and DC voltage, and 1-minute irradiance, temperature, humidity and wind, are averaged to 15 minutes. `n_valid_5min` (0-3) counts the 5-minute readings behind each value.
3. **Full grid:** every 06:00-17:45 bin from each inverter's first day to the end of 2023. Gaps appear as empty rows.
4. **Sensor correction** (see Section 6): irradiance from 2023-02-14 is multiplied by **0.570**; the raw value is kept in `irradiance_raw_wm2`.
5. **Coverage filter:** an inverter is dropped if fewer than 70 % of its daylight bins (>= 50 W/m2) have data. The lowest coverage is 88.6 % (SQ Block P), the median 96.6 %, so **no inverter is dropped**.

| Step | Rows | Columns | Inverters |
|---|---|---|---|
| raw 5-min inverter rows | 4,766,698 | 4 | 44 |
| averaged to 15 min | 1,588,901 | 5 | 44 |
| full 06:00-17:45 grid | 1,588,901 | 5 | 44 |
| merged with weather and metadata | 1,588,901 | 16 | 44 |
| coverage filter (>= 70 %) | 1,588,901 | 17 | 44 |

## 4. Data preprocessing (`src/dataset.py: clean`)

- **Errors become missing, real extremes stay:** power below 0 or above 1.1 x capacity, and irradiance below 0 or above 1,500 W/m2, are sensor errors.
- **Night:** no health score below 100 W/m2 (too dark to judge an inverter). EDA shows 99.8 % of bins produce power from 45-50 W/m2 up, so anything darker is unreliable.
- **Missing power** is an **offline event**, not a fault and not filled in. It feeds the feature `minutes_offline_24h`. Missing feature values (for example no data one day earlier) are filled with the **training median inside the model pipeline**, so validation and test values never influence it.
- **Imputation comparison** (`study_imputation.csv`, Ridge, validation): median RMSE 10.786, mean 10.777. They are equivalent; median is kept because it is robust to outliers.
- **Scaling:** StandardScaler for Ridge, Logistic Regression and the MLP, fitted on training rows only; tree models need none.
- **Encoding:** the inverter ID is **not** a feature, so the models must work for unseen inverters. Orientation is encoded as a 0/1 flag (`multi_orientation`), later dropped (Section 7).

## 5. Data balancing (`src/studies.py`)

Class shares one hour ahead (Healthy / Warning / Faulty): train 85.1 / 8.2 / 6.7 %, validation 79.2 / 11.0 / 9.8 %, test 82.7 / 8.4 / 8.9 %.

| Strategy (Logistic Regression, validation) | Train rows | Macro-F1 | Faulty recall | Faulty precision |
|---|---|---|---|---|
| no balancing | 260,819 | 0.552 | 0.590 | 0.745 |
| **balanced class weights** | 260,819 | 0.569 | **0.723** | 0.466 |
| undersample Healthy (2 per other row) | 116,940 | 0.602 | 0.654 | 0.618 |

**Decision:** balanced class weights. They give the highest Faulty recall, our primary metric, and keep every training row. Every classifier is trained with balanced sample weights. SMOTE was not used: it creates rows by mixing neighbours, which does not make sense for time-ordered measurements. The test set is never resampled.

## 6. Exploratory data analysis (`src/eda.py`, notebook 01)

| Member | Question | Finding | Decision it informs |
|---|---|---|---|
| A1 | How do power and irradiance relate? (`eda_a1_power_vs_irradiance`) | Output rises in a straight line with sunlight; a flat ceiling near 0.75-0.8 kW/kWp is inverter clipping; zero output in full sun marks outages. | Health = actual / expected power; zero in sun is what the score must catch. |
| A1 | (same) Sensor check (`eda_a1_sensor_change`) | Output per unit sunlight drops from about 0.9 to 0.55 for **all** inverters in the week of **2023-02-14**: the irradiance sensor changed. Monthly 2023/2022 ratios 0.51-0.72, median **0.570**. | Rescale irradiance after that date (Section 3). Without it, half of 2023 was labelled Faulty. |
| A1 | (same) Night threshold (`eda_a1_night_threshold`) | 12 % of bins produce at 0-5 W/m2, 78 % at 20-25, **99.8 % from 45-50 W/m2**. | 50 W/m2 = daylight for coverage; 100 W/m2 for a reliable health score. |
| A1 | Where are the gaps? (`eda_a1_gap_heatmap`, `eda_a1_missing_by_hour`) | About 4 % background; long outages at LSK South (Sep-Oct 2023, up to 97 %), Zone J2 (Aug 2022, 92 %), S H Ho (Oct 2023, 86 %), Library (Dec 2023, 72 %), often at all inverters of a station at once. 06:00 is 28 % missing (inverters asleep) vs 3.8 % in other hours. | Outages are events (`minutes_offline_24h`), not gaps to fill; count only daylight bins. |
| A2 | How does temperature affect output? (`eda_a2_temperature_effect`) | Performance ratio in strong sun is 0.99-1.01 at 16-24 C and 0.93-0.96 at 28-38 C, about 5 % lower when hot. | Keep temperature as a feature. |
| A2 | Are features redundant? (`eda_a2_feature_correlation`) | Only `tilt_deg` and `multi_orientation` are almost identical (r = 0.97). | Drop one (Section 7). |
| B1 | Do inverters degrade? (`eda_b1_degradation`) | Median peak-sun performance ratio: 1.002 (2021, May-Dec), 0.962 (2022), 0.977 (2023). No clear yearly decline. | No degradation feature; three years is too short to separate slow ageing from weather and the sensor correction. |
| B1 | Which inverters underperform? (`eda_b1_underperformers`) | Nine inverters are more than 10 % below the campus median at peak sun, worst UG Hall2 2F (-44 %), Zone D 1 (-43 %), S H Ho 2 (-35 %). Zone D 2 is +53 %, which is physically impossible: the two Zone D capacities in the metadata are inconsistent. | Peer deviation as a feature; Zone D capacity listed as a limitation. |
| B2 | How strong is seasonality? (`eda_b2_seasonality`) | Average daylight output per kWp ranges from about 0.20 to 0.38 between months and differs strongly between years (April 2022: 0.37, April 2023: 0.23). | Chronological split (never random); day-of-year features. |
| B2 | How are the classes distributed? (`eda_b2_class_distribution`) | Faulty is 6-10 % depending on season (Winter 10.1 %, Spring 6.0 %, Summer 7.0 %, Autumn 10.3 %). | Imbalance handling (Section 5); judge by Faulty recall and macro-F1, not accuracy. |

## 7. Target, features and selection (`src/dataset.py`, `src/studies.py`)

**Target (B1).**
Health(t) = clip(100 x actual AC power / max(irradiance x kWp x 0.78, 100 W), 0, 100), only when irradiance >= 100 W/m2 and power was measured. The model predicts **Health(t + 1 h)** for the same inverter, matched on the exact timestamp (Option A of the plan). Predicting Health(t) from power and irradiance at t would only relearn the formula (target leakage).

Status: Healthy >= 90, Warning 70-90, Faulty < 70. Sensitivity (`study_thresholds.csv`, training health scores): the Faulty share is 4.4 % for a cut at 60, **7.3 % at 70** and 11.4 % at 80; the Healthy share is 86.0 / **83.2** / 80.0 % for cuts at 85 / **90** / 95. 70 and 90 keep the Faulty class large enough to learn while still meaning "clearly below expected".

**Features (20, all known at time t).**

| Group (owner) | Features |
|---|---|
| Current state, outages (A1) | `health_now`, `n_valid_5min`, `minutes_offline_24h` |
| Weather | `irradiance_wm2`, `temp_c`, `rh_pct`, `wind_speed_ms` |
| Time and metadata (B2) | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `capacity_kwp`, `tilt_deg`, `altitude_m` |
| Physics (B1) | `peer_dev` = (PR - campus median PR) / campus median PR at the same time; `self_dev_30d` = health now - own median health over the past 30 days |
| History (A2) | `health_lag_15m`, `health_lag_1d`, `health_mean_4h`, `peer_dev_mean_4h` |

All rolling windows exclude the current row and use only the past.

**Selection.** Correlation filter on training data (`study_correlated_features.csv`): only `tilt_deg` / `multi_orientation` exceed |r| = 0.85 (r = 0.974), so `multi_orientation` is dropped. Permutation importance (Section 9) confirms which features matter. Adding the two short-history features (`health_lag_15m`, `peer_dev_mean_4h`) lowered a fixed XGBoost's validation RMSE from 10.92 to 10.73, so they were kept.

**Rows.** Train 260,819 (May 2021-June 2022), validation 182,731 (July-December 2022), test 366,110 (2023; 34,891 before and 331,219 after 14 February), of which 37,060 belong to the six unseen inverters.

## 8. Baselines (`src/models/linear.py`)

| Baseline | Rule |
|---|---|
| Persistence (regression) | Health(t + 1 h) = Health(t) |
| Majority class | always "Healthy" |
| Health rule | status in one hour = status now |
| Peer rule | status of 100 x (1 + peer deviation): the inverter judged against the campus at the same moment |

None of them learns anything; every model must beat them. Their scores are in the comparison tables (Section 11).

## 9. Models (`src/models/`)

All models read the same table, use the same 20 features and the same recipe (`run_model` in `src/evaluate.py`): median imputation (+ scaling where needed) inside a pipeline, balanced sample weights for classification, tuning (Section 10), and scores on every evaluation set.

| Track | Model | Owner | Why it fits |
|---|---|---|---|
| T1 | Ridge / Logistic Regression | A1 | interpretable; tests whether the relation is mostly linear |
| T2 | Random Forest (100 trees) | A2 | bagging; non-linear, robust, no scaling |
| T3 | XGBoost (300 trees, `hist`) | B1 | boosting; usually strongest on tables; fast on CPU |
| T4 | MLP (Adam, ReLU, early stopping, batch 256, max 200 epochs) | B2 | tests whether a neural network adds value; needs scaled inputs |
| T5 | Isolation Forest | pairs | unsupervised; catches unusual behaviour without labels |

**What the models rely on** (permutation importance on 5,000 validation rows; bigger = more important):

| Model | Most important features |
|---|---|
| Ridge | health_now (9.6), self_dev_30d (2.0), health_lag_15m (0.8) |
| Random Forest (regression) | peer_dev (4.5), health_now (2.8), self_dev_30d (0.7) |
| XGBoost (regression) | peer_dev (3.6), health_now (3.1), day_cos (0.8) |
| MLP (regression) | health_now (7.8), hour_cos (6.8), hour_sin (4.6) |
| Logistic Regression | health_now (0.18 macro-F1), self_dev_30d, peer_dev_mean_4h |

Every model leans on the current health score: one hour ahead, health mostly persists. The tree models also make strong use of peer deviation, the linear model of the 30-day self deviation. The XGBoost SHAP summary (`xgb_shap_summary.png`) tells the same story: a low peer deviation (doing worse than the rest of the campus right now) pulls the forecast down by up to 55 points, the strongest single effect, and an unusually high 30-day self deviation pulls it down slightly (values return to normal).

## 10. Hyperparameter tuning (`src/evaluate.py`)

GridSearchCV with a **3-fold TimeSeriesSplit** on the training period (each fold trains on the past and validates on the following block). Random Forest and MLP are tuned on a 50,000-row sample of the training period to keep laptop runtimes reasonable, then refitted on all 260,819 rows.

| Model | Search space | Chosen | Validation: default -> tuned |
|---|---|---|---|
| Ridge | alpha 0.01, 1, 100, 10000 | alpha 100 | RMSE 10.678 -> 10.680 |
| Logistic Regression | C 0.01, 1, 100 | C 1 | F1 0.574 -> 0.574, Faulty recall 0.724 -> 0.724 |
| Random Forest (reg.) | max_depth 10, 20; min_samples_leaf 5, 20 | depth 10, leaf 5 | RMSE 10.903 -> **10.361** |
| Random Forest (class.) | same | depth 20, leaf 5 | F1 0.658 -> 0.672, recall 0.602 -> 0.650 |
| XGBoost (reg.) | max_depth 4, 6, 8; learning_rate 0.05, 0.1 | depth 6, lr 0.05 | RMSE 12.046 -> **10.732** |
| XGBoost (class.) | same | depth 4, lr 0.1 | F1 0.633 -> 0.623, recall 0.563 -> 0.630 |
| MLP (reg.) | hidden (32), (64, 32); alpha 0.0001, 0.01 | (64, 32), 0.01 | RMSE 11.188 -> 11.404 |
| MLP (class.) | same | (64, 32), 0.0001 | F1 0.597 -> 0.608, recall 0.677 -> 0.681 |

Tuning helped the tree models most. The linear models barely change: with 260,000 rows they have almost nothing to overfit. For the MLP regressor, the settings chosen on the 50,000-row sample were slightly worse on the full data: the cost of tuning on a sample.

## 11. Validation and evaluation (`src/evaluate.py`, `src/compare.py`)

**Protocol.** Train <= 30 June 2022, validation July-December 2022, test 2023 (all seasons). The six inverters that start in 2023 (SQ Block P, Shaw Auditorium, UG Hall 8, UG Hall 9, Zone A2 inverters 2 and 3) are never in train or validation; "unseen" = their 2023 rows. The test year is also split at the sensor change. Selection rule: **choose on validation, confirm once on test**; regression by lowest RMSE, classification by highest Faulty recall (a missed fault is the costly error). Random seed 42 everywhere.

### Health forecast (regression), RMSE in health points

| Model | Train | Val | Test | Test before / after 14 Feb | Unseen |
|---|---|---|---|---|---|
| Persistence (baseline) | 12.13 | 13.07 | 12.35 | 10.91 / 12.49 | 10.73 |
| Ridge | 10.00 | 10.68 | **10.17** | 9.75 / 10.22 | **9.07** |
| **Random Forest** | 8.22 | **10.36** | 10.45 | 9.49 / 10.54 | 9.13 |
| XGBoost | 7.64 | 10.73 | 10.69 | 9.97 / 10.76 | 9.69 |
| MLP | 8.11 | 11.40 | 10.84 | 12.32 / 10.67 | 11.67 |

- **Every learned model beats persistence** on validation and test (for example Random Forest 10.36 vs 13.07 on validation, a 21 % lower error).
- **Selected: Random Forest** (lowest validation RMSE). On test it is confirmed as clearly better than persistence (10.45 vs 12.35), but Ridge is slightly better on test (10.17) and on unseen inverters (9.07). The difference between the top three models is small; the simple Ridge is a strong, honest alternative.
- **Before vs after 14 February** differs by about 1 point or less for Ridge, Random Forest and XGBoost, so the sensor correction removed the label shift. The early-2023 rows are winter data, which is harder for the MLP (12.32 before vs 10.67 after).
- **Over- and underfitting:** the tree models fit training much better than validation (XGBoost 7.6 vs 10.7), a sign of some overfitting that tuning limited; the linear model scores alike everywhere.

### Status one hour ahead (classification)

| Model | Val F1 / Faulty recall | Test F1 / Faulty recall | Test before 14 Feb | Test after 14 Feb | Unseen F1 / recall |
|---|---|---|---|---|---|
| Majority class | 0.30 / 0.00 | 0.30 / 0.00 | 0.30 / 0.00 | 0.30 / 0.00 | 0.32 / 0.00 |
| Health rule | 0.60 / 0.65 | 0.59 / 0.60 | 0.67 / 0.72 | 0.58 / 0.59 | 0.47 / 0.37 |
| Peer rule | 0.58 / 0.66 | 0.54 / 0.71 | 0.61 / 0.81 | 0.53 / 0.70 | 0.46 / 0.36 |
| **Logistic Regression** | 0.57 / **0.72** | 0.57 / **0.71** | 0.66 / 0.80 | 0.56 / 0.70 | 0.45 / **0.39** |
| Random Forest | **0.67** / 0.65 | **0.63** / 0.58 | **0.71** / 0.74 | **0.62** / 0.56 | **0.49** / 0.34 |
| XGBoost | 0.62 / 0.63 | 0.58 / 0.58 | 0.68 / 0.74 | 0.57 / 0.56 | 0.43 / 0.34 |
| MLP | 0.61 / 0.68 | 0.56 / 0.59 | 0.65 / 0.74 | 0.55 / 0.58 | 0.42 / 0.32 |

- **Selected: Logistic Regression** (highest validation Faulty recall, 0.72, confirmed on test, 0.71). It catches the most faults at the cost of more false alarms (Faulty precision 0.50 on test).
- **Random Forest has the best overall balance** (macro-F1 0.63 on test). If false alarms are expensive, it is the better choice.
- **Simple rules are hard to beat.** The peer rule reaches the same test Faulty recall as Logistic Regression (0.71). One hour ahead, the status mostly persists.
- **Unseen inverters are the weakest set for every model** (Faulty recall 0.32-0.39). Generalising to brand-new rooftops is the main open problem.

### Anomaly detection (Isolation Forest)

| Benchmark | Setting | Detection rate | False-alarm rate | Precision / F1 |
|---|---|---|---|---|
| HKUST, injected faults (A1, A2) | contamination 0.01, 100 trees (chosen on validation) | **76.8 %** of 306 fault rows | **1.1 %** | 0.056 / 0.105 |
| GPVS-Faults, real lab faults (B1, B2) | contamination 0.05, 200 trees | 39.0 % of 847 fault windows | **0 %** of 86 normal windows | 1.00 / 0.56 |

- On HKUST, injected faults (a 35 % drop for 7 days and a 5-day outage) are mostly caught with few false alarms. Precision is low because the injected rows are under 0.1 % of the test year and the detector also flags real unusual rows, which count as "false" here.
- On GPVS-Faults it never raised a false alarm. It catches fault type F5 always, F1, F2 (MPPT mode) and F4 (IPPT mode) mostly, and F6 and F7 almost never (`iforest_gpvs_by_scenario.csv`). Faults that change the measured signals only a little are invisible to an unsupervised detector.

### Failure cases

The largest Random Forest errors (`random_forest_regression_failures.csv`) all have the same shape: health 100 now and **0 one hour later** with measured power, for example Library inverter 2 on 28 and 29 December 2023 and Zone A2 inverter 1 on 1 December 2023. These are sudden trips to zero output. Nothing in the data at time t announces them, so no model can forecast them; they limit every model. Library in December 2023 is also one of the outage months in the gap heatmap.

## 12. Use case, impact and limitations

**Use.** Every 15 minutes each inverter is scored. The operations team sees inverters ranked by forecast health; a Faulty forecast (Logistic Regression, high recall), an offline event or an anomaly flag opens an inspection ticket.

**Impact.** Faster detection of underperforming or offline inverters, less lost energy, and inspection effort pointed at the right roofs. The peer comparison already identifies nine inverters that run more than 10 % below the campus at peak sun.

**Limitations.**
- **Engineered labels.** "Faulty" means health below 70, not a confirmed fault; GPVS-Faults only partly makes up for this.
- **Sensor correction uses 2023 data.** The 0.570 factor compares 2023 months with 2022 months. No model input or training row depends on it, but the test labels do. One factor slightly over-corrects late 2023 (Oct-Dec ratios 0.68-0.72).
- **One weather station** for 37 rooftops: local shading and orientation show up as noise.
- **No data after 18:00**, no DC current, no module temperature.
- **Six unseen inverters only**, so the unseen scores are noisy.
- **Zone D capacities** in the metadata are inconsistent with its output and are not corrected.
- **Sudden trips** cannot be forecast one hour ahead (Section 11).
- **Single subtropical campus:** transfer to other climates or plants is untested.

**Future work.** Real maintenance logs as labels, one irradiance sensor per roof, panel-level optimiser data, and automatic retraining when sensors change.

---

## Changes from the original plan

| Plan | Implemented | Reason |
|---|---|---|
| Separate preprocessing, feature and target modules, each member with their own pipeline | One `dataset.py` that builds one table for every model | identical rows and features make the model comparison fair and the code much shorter |
| RandomizedSearch / Optuna for some models | GridSearchCV + TimeSeriesSplit for all models | small grids are enough and the same recipe is easier to explain |
| Per-model balancing studies | One study, balanced weights for every classifier | the same decision applies to all models |
| SHAP for RF and XGBoost, partial dependence for MLP | Permutation importance for every model, SHAP for XGBoost | one comparable method for all, plus one SHAP example |
| 60 stations | 44 inverters at 37 stations | only these have inverter-level data |
| Lag t-1, t-4, t-96 and many rolling windows | 15-min and 1-day lags, 4-hour means, 30-day self deviation | the few features that carry most of the information |

## Reproducibility

- Python 3.12 with pinned `requirements.txt`; seed 42; all dates and thresholds in `configs/config.yaml`.
- `python -m src.run_all` rebuilds everything from the raw files in about 25 minutes on a laptop CPU; `python -m unittest discover -s tests` runs 7 checks (health formula, status thresholds, exact one-hour target, past-only windows, unseen inverters kept out of training, cleaning rules, metrics).
- The raw files are never modified.

## Team contributions

| Member | Led | Model | EDA | Feature group | Anomaly benchmark |
|---|---|---|---|---|---|
| A1 | Stages 1-3, 8 | Ridge, Logistic Regression, baselines | power vs irradiance, gaps | outage features | HKUST injected faults |
| A2 | Stages 4-5 | Random Forest | temperature, correlation | history (lags, rolling means) | HKUST injected faults |
| B1 | Stage 7 | XGBoost (+ SHAP) | degradation, underperformers | peer and self deviation | GPVS-Faults |
| B2 | Stages 11-12 | MLP | seasonality, class distribution | time and metadata | GPVS-Faults |

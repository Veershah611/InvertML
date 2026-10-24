# Solar Inverter Health Monitoring & Predictive Maintenance System

An end-to-end machine learning project that predicts inverter health, classifies operational status, and flags anomalous behaviour, using **up to three years of inverter-level telemetry from 44 inverters at 37 rooftop PV stations** (out of the dataset's 60) plus on-site weather data. A team of four builds it. The work follows the 12-stage pipeline in the Innovative Assignment guidelines, and every major decision is justified and validated.

> This document is the project **plan**. What was actually built, with every result, is in [REPORT.md](REPORT.md); where the final implementation differs from this plan, REPORT.md lists the difference and the reason.

---

## 0. Pipeline Coverage Checklist (maps to IA Guidelines §2)

| # | Stage | Section here | Lead (all four contribute; see §13) |
|---|-------|--------------|-----------------|
| 1 | Problem Definition & Use Case | §1 | A1 |
| 2 | Data Acquisition & Understanding | §2 | A1 |
| 3 | Data Preparation | §3 | A1 |
| 4 | Data Preprocessing | §4 | A2 |
| 5 | Data Balancing | §5 | A2 |
| 6 | Exploratory Data Analysis | §6 | All four (2 EDA questions each) |
| 7 | Feature Engineering / Selection | §7 | B1 (all four build one feature group each) |
| 8 | Baseline Model | §8 | A1 (every student compares their model against it) |
| 9 | ML / Advanced ML Modelling | §9 | All four, one model family each |
| 10 | Hyperparameter Tuning | §10 | All four, on their own model |
| 11 | Validation & Evaluation | §11 | B2 owns the framework; all four apply it |
| 12 | Use Case, Impact & Limitations | §14 | B2 |

Team: **Pair A** = A1, A2 · **Pair B** = B1, B2.

Every stage is either implemented or explicitly marked "not applicable" with a reason, as the guidelines require.

---

## 1. Problem Definition & Use Case (Stage 1)

**Problem.** Inverters are the most failure-prone component in a PV system. When one degrades, trips, or loses communication, the lost energy often goes unnoticed until someone reviews yield reports by hand. Raw output depends heavily on the weather, so "low output" by itself does not tell you whether the cause is a cloud or a fault.

**Stakeholders / users.**
* Facility / sustainability office O&M teams managing fleets of rooftop PV systems (a campus, a commercial portfolio): they decide which system to inspect first.
* Asset owners: they track energy loss and the cost of downtime.
* Field technicians: they receive the alerts.

**Objective.** Separate weather-driven drops in output from system-driven underperformance. Forecast short-term health so maintenance can be scheduled before a fault grows into a breakdown.

**Inputs.** 5-minute inverter-level generation data, 1-minute on-site weather data, and station metadata (rated capacity, orientation, etc.).

**Outputs.**
1. Continuous **Health Score (0–100)** per station per time step, plus a short-horizon forecast of it.
2. Status class: `Healthy` / `Warning` / `Faulty`.
3. Anomaly flag from an unsupervised detector, for failure patterns the labels do not cover.

**Constraints.**
* No ground-truth maintenance or fault logs are available for these stations, so labels must be engineered (§7.2). Real fault labels come only from the secondary dataset (§2.2).
* A single weather station serves all 37 rooftops.
* Data is time-ordered, so any evaluation must respect chronology.
* Everything must train on CPU-only laptops (§12).

**Expected impact.** Faster detection of underperforming systems, less energy lost while faults go unnoticed, and a ranked inspection list instead of manual report review.

---

## 2. Data Acquisition & Understanding (Stage 2)

### 2.1 Primary dataset: HKUST Rooftop PV (2021–2023)
* **Source:** Lin et al., *A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics*, Scientific Data (2025). Data on Dryad: https://doi.org/10.5061/dryad.m37pvmd99
* **Ownership / licence:** Published on Dryad by the Hong Kong University of Science and Technology team (published 2 Apr 2024, updated 8 Sep 2024). **Licence: public domain (CC0)**, confirmed on the Dryad page by A1. Paper: Lin et al., Scientific Data (2025), https://doi.org/10.1038/s41597-025-04397-y.
* **Collection method:** 60 grid-connected rooftop PV stations on the HKUST campus. Inverter-level data every 5 minutes exists only for the 37 stations with panel-level optimizers; the other 23 have hourly site-level data only. Weather comes from **one on-site station every 1 minute**. Data was collected over Wi-Fi and RS232/RS485 loggers into a central database.
* **Size:** about 296 MB zipped. Inverter-level files: **4,766,698 rows** across 44 CSVs (633 MB unzipped), about 1.6M rows after 15-min resampling.
* **Metadata:** stored as a **Brick schema model (`.ttl`)**, queried with SPARQL (Python `rdflib`). It holds station specifications such as capacity and orientation.
* **Panel-level optimizer data:** 37 of the 60 stations have panel-level optimizers, but the dataset package contains **no panel-level files**, only inverter- and site-level files for those stations.

**Columns** (verified by A1 from the dataset files):

| Column | Type | Unit | Available? |
|---|---|---|---|
| `Time` | datetime | HKT (assumed; README does not state it) | Yes, all files |
| Station / inverter ID | categorical | — | Yes, from the file name |
| `totalActivePower` (AC power) | float | **W** | Yes, inverter files |
| Per-phase L1/L2/L3: AC current, AC voltage, AC frequency, active power, reactive power | float | A, V, Hz, W | Yes, inverter files |
| `dcVoltage` | float | V | Yes, inverter files |
| DC current / DC power | — | — | No, not provided |
| `generation` (energy), `power` | float | kWh, W | Yes, site-level files only |
| Irradiance (single sensor; GHI vs. plane-of-array not stated) | float | W/m² | Yes, 1-min |
| Ambient temperature | float | °C | Yes, 1-min |
| Module temperature | — | — | No, not provided |
| Wind speed / direction, relative humidity, rainfall, sea-level pressure, visibility | float | m/s, °, %, mm, hPa, km | Yes, 1-min |

The feature design (§7) works with **AC power + irradiance + ambient temperature + capacity** alone. Both *conditional* features (AC/DC conversion efficiency and module − ambient temperature) are **not possible** with this dataset.

**Scope decision (A1): inverter-level data only.** That means **44 inverters at 37 stations**, the stations with panel-level optimizers. The other 23 stations have only hourly site-level data and are excluded. Inverter data (5-min) is resampled to 15-min.

**Dataset findings (A1, from inspecting the files):**
* Folder layout: `PV generation dataset/` holds `PV stations with panel level optimizer/` (44 inverter-level CSVs at 5-min plus 37 site-level CSVs at 15-min) and `PV stations without panel level optimizer/` (23 site-level CSVs at **1-hour**). The README's "5-minute for all stations" holds only for the inverter files.
* `Meteorological dataset/` has 7 variables, one file per year (2021–2023), at 1-min resolution.
* Start dates vary by inverter: most start between May and Oct 2021, and none start in Jan 2021. Six inverters start only in 2023 (SQ Block P, Shaw Auditorium, UG Hall8, UG Hall9, Zone A2 Inverter 2 and 3), so they cannot appear in the training period. **B2: this affects the chronological split and the unseen-station choice (§11.1).**
* Inverter files contain daytime rows only (about 06:00–17:55). Missing night rows are expected and are not gaps.
* Two timestamp formats are mixed across files (`2021-05-04 07:00:00` and `2021/5/1 6:00`). Weather files use the second format.
* `Rainfall_2021` is `.xlsx`; the other years are `.csv`.
* Site-level file quirks (not used, but noted): `Zone L1 (Station 1)` has a duplicate first timestamp and runs into 2024. `SQ14` starts in 2023 but has as many rows as files that start in 2021, so its start date is probably wrong.
* Licence: the dataset README says "Licenses: N/A" and calls the paper "under review". Both are outdated: the Dryad page lists the licence as public domain (CC0), and the paper is published (see above).

**Target variable.** None is provided. The health score and status labels are **engineered** (§7.2), and the report must say so plainly.

**Known limitations to document.**
* One weather station for 37 rooftops. Local shading and orientation differences show up as station-level offsets.
* Urban rooftop systems are smaller and more heterogeneous than utility-scale farms.
* Gaps from Wi-Fi and communication failures. These matter: they are themselves a real "inverter offline" signal.
* No real fault labels for these stations.

### 2.2 Secondary dataset: GPVS-Faults (real fault labels)
* **Source:** Bakdi et al., Mendeley Data, DOI 10.17632/n76t439f65.1.
* **What it is:** lab experiments on a grid-connected PV microgrid. There are 16 scenario files (`F0` fault-free, `F1–F7` fault types, under MPPT and IPPT modes), covering PV array faults, inverter faults, grid anomalies, sensor faults and MPPT controller faults.
* **Why we use it:** the HKUST labels are engineered. GPVS-Faults lets us check that our anomaly detector and fault-classification approach catch **real** faults. It is used **only for validation**, with its own small feature pipeline (window statistics over current and voltage signals), and is not merged with HKUST.

### 2.3 Fallback
If the HKUST README shows that DC power or module temperature is missing and the team wants those features, the **NIST Campus PV dataset** is a fallback. It has DC and AC power, module temperatures and inverter fault codes for 3 arrays, but only a small number of systems.

---

## 3. Data Preparation (Stage 3)

* Unzip the data and parse the Brick metadata (`rdflib` + SPARQL) into a station table: ID, capacity, orientation and so on.
* Load the **inverter-level files only**. Convert them to **Parquet with `float32` dtypes** right away (roughly halves memory and loads much faster).
* Parse timestamps in a single timezone (HKT). Check the daylight/clock alignment between the generation and weather data.
* **Resample:** generation 5-min → **15-min**; weather 1-min → 15-min mean. Justification: this cuts the row count about 3×, reduces noise, and still captures dawn and dusk.
* Merge generation and weather on timestamp. Join station metadata on station ID.
* **Duplicate analysis:** count and remove duplicate `(station, timestamp)` rows.
* **Missing-interval analysis:** reindex each station to a full 15-minute grid; compute gaps per station per month.
* **Validity filter:** drop stations with very poor coverage (for example < 70% of daylight intervals). Report which stations were dropped and why.
* Record **rows and features before and after** each step in a summary table (mandatory check).

---

## 4. Data Preprocessing (Stage 4)

* **Night-time handling:** remove rows with irradiance below about 20–50 W/m², because they carry no health information. Justify the threshold from EDA.
* **Missing values:**
  * Short gaps (≤ 1 hour): linear time interpolation.
  * Long gaps while it is daylight: **do not impute**. Keep them as an `is_offline` event, because a communications or inverter outage is itself a health signal.
  * Compare at least two imputation choices and report the effect on validation metrics.
* **Outliers:** physically impossible values (negative power, AC power > 1.1 × rated capacity, irradiance > 1500 W/m²) are errors, so remove them. Extreme but physically valid values may be real faults, so **keep them**. Document how the two cases are told apart.
* **Categorical encoding:** station ID. Decide between leaving it out (better generalisation to unseen stations, **preferred**), target encoding fitted on training data only, or one-hot encoding. Orientation and other metadata are one-hot encoded.
* **Scaling:** tree models need none. Ridge/Logistic Regression, the MLP and the Isolation Forest comparisons get `StandardScaler`, **fitted on training data only**.
* **Leakage check:** every scaler, imputer and encoder is fitted after the split, on training data only. Rolling features look backward only.

---

## 5. Data Balancing (Stage 5)

* Plot the distribution of the `Healthy` / `Warning` / `Faulty` classes, overall and per season. `Faulty` is expected to be rare.
* Candidate strategies:
  * `class_weight='balanced'` (RF) / sample weights (XGBoost). **This is the preferred option.**
  * Random undersampling of `Healthy` within the training set. With millions of rows this is cheap and also speeds up training.
  * SMOTE: only on the training fold, inside an `imblearn.Pipeline`. It is questionable for time-series rows, so justify it or reject it.
* Compare at least two of these on validation macro-F1 and `Faulty` recall.
* The test set is **never** resampled.
* For the regression target: look at its distribution and its error per score range (no balancing).

---

## 6. Exploratory Data Analysis (Stage 6)

EDA should answer specific questions, and each finding should lead to a decision:

| Question | Analysis | Decision it informs |
|---|---|---|
| How do AC power and irradiance relate, per station? | Scatter, normalised by capacity | Expected-power model, night-time threshold |
| How strong is seasonality across 3 years? | Monthly specific yield (kWh/kWp) | Chronological split, seasonal features |
| Do stations degrade over time? | Performance-ratio trend per station per year | Degradation feature, limitations |
| Which stations underperform consistently? | Station PR vs. campus median | Sanity check of health labels |
| How does temperature affect output? | PR vs. ambient (or module) temperature | Temperature features |
| Where are the gaps / offline events? | Missing-interval heatmap (station × month) | Imputation vs. `is_offline` |
| Are features redundant? | Correlation matrix | Feature selection |
| How are classes distributed? | Class counts per season and per station | Balancing, split sanity |

Include univariate, bivariate and target-distribution analysis. Leave out purely decorative plots.

---

## 7. Feature Engineering / Selection (Stage 7)

### 7.1 Features
**Core (needs only AC power, irradiance, temperature and capacity):**
* **Performance Ratio (PR)** = AC energy / (irradiance × rated capacity), clipped to physical limits.
* **Specific yield** = AC power / rated capacity (kW/kWp), so stations of different sizes are comparable.
* **Self-relative deviation** = current PR / that station's rolling 30-day median PR in similar irradiance conditions.
* **Peer deviation** = station PR / campus median PR at the same timestamp. This cancels out the weather, since all stations share the same sky.
* **Temperature effect** = ambient temperature, and an irradiance × temperature interaction.
* **Lag & rolling features:** 1-hour, 4-hour and 1-day rolling mean/std of PR and peer deviation; lags t−1, t−4, t−96.
* **Outage features:** `is_offline`, minutes offline in the last 24 hours.
* **Time features:** sin/cos of hour of day and of day of year (seasonality over 3 years).
* **Station metadata:** capacity, orientation and tilt, if available.

**Conditional (only if the README confirms the columns):**
* Conversion efficiency = AC / DC.
* Temperature delta = module − ambient temperature; thermal-stress proxy = temperature delta per unit irradiance.

### 7.2 Target construction (and avoiding circular leakage)
**Warning:** If the health score is computed from features at time *t* and the model predicts it from those same features at time *t*, the model only learns the formula back. Evaluators will treat that as target leakage.

Use **one** of the following designs and justify it in the report:

* **Option A — Forecasting (recommended).** Health(t) is computed by a rule (penalties on self-relative deviation, peer deviation and outage time). The model predicts **Health(t + h)** (for example h = 1 hour or next day) using only features up to time *t*.
* **Option B — Normal-behaviour model.** A regressor learns expected AC power from weather, time and station metadata on healthy periods. Health = actual / expected power. The target never appears among the inputs.

Status labels: thresholds on the health score (for example ≥ 90 Healthy, 70–90 Warning, < 70 Faulty), **set from EDA**, with a sensitivity analysis.

### 7.3 Selection
* Random Forest and XGBoost feature importances, **plus permutation importance on validation data**.
* Correlation filter for redundant features.
* Report the metric before and after selection to show that the selection helped.

---

## 8. Baseline Model (Stage 8)

* **Regression:** (a) persistence (Health(t+h) = Health(t)), (b) Ridge regression.
* **Classification:** (a) majority class, (b) rule-based threshold on peer deviation, (c) Logistic Regression.

Every advanced model is reported as an improvement over these baselines.

---

## 9. Modelling (Stage 9)

There are four supervised model families from four different ML paradigms (linear, bagging, boosting, neural network), plus one unsupervised layer:

| Track | Model | Task | Owner | Why it fits |
|---|---|---|---|---|
| T1 | Ridge / Logistic Regression | Regression + classification | A1 | Interpretable baseline; tests whether the relationship is mostly linear |
| T2 | Random Forest | Regression + classification | A2 | Bagging: captures non-linearity, robust to outliers, gives importances |
| T3 | XGBoost (`tree_method="hist"`) | Regression + classification (production candidate) | B1 | Boosting: usually the strongest on tabular data; fast on CPU; handles NaNs natively |
| T4 | MLP (`sklearn` MLPRegressor / MLPClassifier) | Regression + classification | B2 | Neural network: tests whether a non-tree learner adds value; needs scaled inputs |
| T5 | Isolation Forest | Unsupervised anomaly detection | Pair A on HKUST, Pair B on GPVS-Faults | Catches failure patterns the engineered labels miss; validated on real faults |

For each model, document its assumptions, why it suits this problem, and where it would fail.

**MLP documentation (required by the guidelines for neural networks):** architecture (hidden layers and units), activation (ReLU), optimizer (Adam), loss (squared error / log-loss), epochs (`max_iter`), batch size, and regularisation (L2 `alpha`, early stopping). Train it on a 500k-row time-ordered sample to keep CPU runtime reasonable.

**Classification design choice to compare:** threshold the regression output into classes, or train a dedicated classifier.

---

## 10. Hyperparameter Tuning (Stage 10)

* Validation method: `TimeSeriesSplit` (expanding window) on the training period only. **No shuffled K-fold.**
* **Tune on a 10–20% time-ordered sample** of the training data, then refit the best configuration on the full training set. This is needed to keep laptop runtimes reasonable, and the report should say so.
* Search: `RandomizedSearchCV` or Optuna for RF and XGBoost; small `GridSearchCV` for Ridge/Logistic Regression; `RandomizedSearchCV` over hidden layers, `alpha`, learning rate and batch size for the MLP; contamination and `n_estimators` for Isolation Forest.
* XGBoost: early stopping on the validation fold.
* Report the search space, the best parameters, and the default-vs-tuned comparison for every model.

---

## 11. Validation & Evaluation (Stage 11)

### 11.1 Split strategy
* **Chronological split:**
  * Train: Jan 2021 – Jun 2022 (18 months)
  * Validation: Jul 2022 – Dec 2022 (6 months)
  * Test: **all of 2023** (12 months, so every season is in the test set)
* **Unseen-station test:** hold out about 20% of stations (**≈ 7 of 37**) completely. Train on the other stations, then evaluate on the held-out stations in 2023. This answers the guidelines' question "how would the solution work with new/unseen data?" Stations that start only in 2023 have no training data, so they are natural candidates for this set.
* **Real-fault check:** GPVS-Faults (§2.2).
* Fixed random seeds (`random_state=42`) everywhere.

### 11.2 Metrics (and why)
* **Regression:** MAE (easy to read, in health points), RMSE (penalises large misses), R². Add residual plots, actual-vs-predicted plots, and error broken down by season and by station.
* **Classification:** macro-F1, per-class precision/recall, a confusion matrix and PR-AUC. **`Faulty` recall is the primary metric.** Accuracy alone is not reported as evidence, since the classes are imbalanced.
* **Anomaly detection:**
  * (a) Agreement with the engineered `Faulty` class on HKUST.
  * (b) **Synthetic fault injection** into HKUST test data (for example a sudden 30% PR drop, or an offline period, on one station): detection rate and false-alarm rate.
  * (c) **Real faults on GPVS-Faults:** precision, recall and F1 for fault vs. fault-free.
* Report training and test scores side by side to show over- or underfitting.

### 11.3 Error analysis & explainability
* SHAP TreeExplainer for RF and XGBoost, computed on a 5–10k-row sample (global summary plus local plots for the lowest health scores).
* Failure-case review: which seasons, weather conditions or stations does each model miss?
* Final model selection is justified on validation results and confirmed **once** on the test set.

---

## 12. Compute Plan (CPU-only laptops)

No GPU is needed: every model, including the small MLP, trains on CPU.

| Step | Single run (est.) | With tuning (est.) |
|---|---|---|
| Load + clean + resample (once, saved as Parquet) | 5–15 min | — |
| Ridge / Logistic Regression | seconds | < 5 min |
| XGBoost (`hist`, ~500 trees) | 2–5 min | 1–3 hrs (on the tuning sample) |
| Random Forest (300 trees) | 15–30 min | tune on the sample only |
| MLP (500k-row sample) | 5–15 min | 1–3 hrs (on the tuning sample) |
| Isolation Forest | 1–3 min | < 30 min |
| SHAP (5–10k rows) | 2–10 min | — |

Assumes about 3M rows × 30 features on a 4–8 core, 16 GB laptop. **Benchmark one XGBoost fit in week 1** and adjust.

* `n_jobs=-1`, `float32`, and Parquet everywhere.
* On 8 GB laptops: load stations in chunks or use Polars, or start with a subset of stations.
* Run long tuning jobs overnight. Google Colab or Kaggle notebooks are a free fallback.

---

## 13. Team Work Breakdown: Two Pairs, Equal ML Exposure

### 13.1 Structure
* **Pair A — Data Foundation:** A1, A2
* **Pair B — Features & Evaluation:** B1, B2

Each pair **leads** half of the pipeline. On top of that, every student:
1. **Owns one supervised model family** and runs it on **both** tasks: regression (health score) and classification (status). Every student therefore does model fitting, class-imbalance handling, hyperparameter tuning, regression metrics, classification metrics and explainability.
2. **Answers 2 EDA questions** from §6, so everyone does analytical EDA.
3. **Builds 1 feature group** from §7, so everyone does feature engineering.
4. **Runs Isolation Forest with their partner** on one anomaly benchmark, so everyone does unsupervised learning.

Leading a stage means designing it, writing the shared code, and defending the decisions in the report. It does not mean being the only person who touches it.

### 13.2 Per-student assignments

| | **A1** | **A2** | **B1** | **B2** |
|---|---|---|---|---|
| **Leads (pipeline stages)** | Stages 1–3: problem definition, licence/citation, README column table, Brick metadata parsing, Parquet conversion, resampling, merging, duplicate/gap analysis, before/after table | Stages 4–5: night-time threshold, gap/offline policy, imputation comparison, outlier rules, encoding/scaling pipelines, **imbalance study** (class weights vs. undersampling vs. SMOTE) | Stage 7: target construction (Option A/B), status thresholds + sensitivity analysis, feature selection (permutation importance, correlation filter) | Stages 11–12: chronological + unseen-station splits, shared `evaluate.py`, synthetic fault injection, GPVS-Faults loader, final comparison table, impact & limitations |
| **EDA questions (§6)** | Power vs. irradiance; gaps/offline heatmap | Temperature effect; feature correlation | Degradation over 3 years; consistently underperforming stations | Seasonality; class distribution |
| **Feature group (§7)** | Outage features (`is_offline`, minutes offline) | Lag & rolling statistics | PR, peer deviation, self-relative deviation | Time (sin/cos) + station metadata |
| **Model family (both tasks)** | **Linear:** persistence + Ridge; majority + rule-based + Logistic (these are the team baselines) | **Bagging:** Random Forest | **Boosting:** XGBoost | **Neural network:** MLP |
| **Tuning method** | GridSearch + TimeSeriesSplit (shared `run_model`) | Same, on a 50k-row sample | Same | Same, on a 50k-row sample |
| **Explainability** | Coefficients + permutation importance | Permutation importance + SHAP | SHAP (global + local) | Permutation importance + partial dependence |
| **Anomaly (with partner)** | Isolation Forest on HKUST with injected faults | Same, together with A1 | Isolation Forest on GPVS-Faults (real faults) | Same, together with B1 |
| **Shared write-up** | Reproducibility section | Data-leakage audit | Feature/target justification | AI-usage & resources log |

**Load balance:** Pair A's pipeline stages are heavier in week 1 (data volume). Pair B's are heavier in weeks 2–3 (target design, evaluation framework, two anomaly datasets). Over the whole project the two loads even out.

### 13.3 Concept coverage matrix
Legend: **L** = leads, **H** = hands-on, **R** = reviews and must be able to explain it in the viva.

| ML concept | A1 | A2 | B1 | B2 |
|---|---|---|---|---|
| Data acquisition, cleaning, merging | L | R | R | R |
| Preprocessing, scaling, leakage control | R | L | R | H (MLP needs scaling) |
| Class-imbalance handling | H | L | H | H |
| Analytical EDA | H | H | H | H |
| Feature engineering | H | H | L | H |
| Target design & feature selection | R | R | L | R |
| Baselines | L | H (compares) | H (compares) | H (compares) |
| Supervised regression model | H | H | H | H |
| Supervised classification model | H | H | H | H |
| Hyperparameter tuning with time-series CV | H | H | H | H |
| Regression metrics & residual analysis | H | H | H | L |
| Classification metrics (F1, PR-AUC, confusion matrix) | H | H | H | L |
| Unsupervised anomaly detection | H | H | H | H |
| Explainability (SHAP / permutation) | H | H | H | H |
| Generalisation (unseen-station test) | H | H | H | L |

Every student is hands-on (H or L) in **at least 12 of the 15 concepts** and reviews the rest.

### 13.4 Working rules
* **Within a pair:** partners review each other's code before merging, and each explains the other's model at the weekly sync. In the viva, every student should be able to explain their partner's model as well as their own.
* **Across pairs:** Pair A reviews Pair B's target design and evaluation code; Pair B reviews Pair A's data and preprocessing pipeline. This catches leakage the authors themselves miss.
* **Shared contracts:** one `prepare_data.py` → Parquet output, one `dataset.py` (target, features, splits) that every model reads, one `evaluate.py` with a single training-and-scoring recipe. Nobody computes metrics their own way.
* **Every model deliverable includes:** a default-vs-tuned table, train-vs-test scores, a baseline comparison, results on the unseen-station test, and 2–3 failure cases.

### 13.5 Timeline (4 weeks)

| Week | Pair A (A1, A2) | Pair B (B1, B2) | Everyone |
|---|---|---|---|
| 1 | A1: download, README columns, metadata, Parquet. A2: preprocessing rules drafted on 5–10 stations | B1: draft target + status thresholds on the subset. B2: splits, `evaluate.py`, GPVS loader, fault-injection code | Benchmark one XGBoost fit; start your 2 EDA questions |
| 2 | A1: full-dataset preparation + baselines. A2: imbalance study; preprocessing frozen | B1: target + features frozen; feature selection. B2: evaluation framework frozen | Build your feature group; finish EDA; first untuned model run |
| 3 | Tune RF + linear models; Isolation Forest on HKUST injected faults | Tune XGBoost + MLP; Isolation Forest on GPVS-Faults | SHAP/permutation analysis; unseen-station test; cross-pair review |
| 4 | Write data and preprocessing sections | Write evaluation, comparison and impact sections | Final model choice; peer explanation drill; viva rehearsal |

### 13.6 How to collaborate

**One GitHub repo, with code in `.py` files and notebooks as thin drivers.** Notebooks cause merge conflicts, so shared logic lives in `src/` and each notebook only imports and calls it.

```
InvertML/
├── data/                # NOT in git (.gitignore); raw data + processed Parquet
├── src/
│   ├── prepare_data.py  # Stage 3 (A1)
│   ├── dataset.py       # Stages 4 + 7: cleaning, target, all feature groups, splits
│   ├── eda.py           # Stage 6: all eight EDA questions
│   ├── studies.py       # Stages 4, 5, 7: imputation, balancing, thresholds, correlation
│   ├── evaluate.py      # Stage 11: metrics and the shared run_model() recipe (B2)
│   ├── compare.py       # Stage 11: final comparison
│   └── models/          # linear.py (A1), rf.py (A2), xgb.py (B1), mlp.py (B2), iforest.py (pairs)
├── notebooks/           # 01_data_and_eda, 02_target_and_features, 03_models, 04_evaluation
├── configs/config.yaml  # paths, seeds, split dates, thresholds: one source of truth
├── results/results.csv  # one block of rows per model and task
├── reports/figures/, reports/tables/
└── requirements.txt     # pinned versions
```

**Git workflow**
* `main` is protected: no direct pushes.
* One branch per task (`a2/random-forest`, `b1/target`). Open a pull request and have **your partner** review it before merging. This is the within-pair review from §13.4.
* Pull `main` every morning before starting work.

**Data sharing**
* Raw and processed data never go into git. A1 uploads the cleaned Parquet files to one shared Google Drive / OneDrive folder (`hkust_15min.parquet`). Everyone downloads the same file.
* If the processed data changes, bump the version and announce it in the team chat.

**Same environment on four laptops**
* Pinned `requirements.txt` (or `environment.yml`) plus a `make setup` / one-line install command in the README.
* All split dates, seeds and thresholds are read from `configs/config.yaml`, never hard-coded, so everyone's numbers are comparable.

**Tracking experiments**
* Each model run replaces its own rows in `results/results.csv`: owner, model, task, evaluation set and metrics. `python -m src.compare` turns this into the comparison table in the report.

**Planning & communication**
* GitHub Projects board with columns To do / In progress / Review / Done. Every row in §13.2 becomes a card.
* One team chat for quick questions. One fixed **30-minute weekly sync**: each pair demos progress, then the cross-pair review.
* Report written in one shared Google Doc or Overleaf, with one section per lead and a final edit pass by everyone.

**Backend API & dashboard (optional, outside the ML split).** A FastAPI endpoint and a small React dashboard can be built with GenAI assistance as a team effort once the model is frozen. Only claim deployment that is actually demonstrated (guidelines §9).

---

## 14. Use Case, Impact & Limitations (Stage 12)

**How it would be used.** Every 15 minutes, the latest telemetry for each station is scored. The O&M team sees stations ranked by forecast health, and `Faulty` predictions, offline events or anomaly flags trigger an inspection ticket.

**Impact.** Shorter time-to-detection for underperforming or offline inverters, less energy lost, inspection effort directed where it matters, and year-over-year degradation tracking from the 3-year history.

**Risks & limitations.**
* The HKUST labels are engineered, not real maintenance records. GPVS-Faults validates the approach on real faults, but in a lab setting with different signals.
* One weather station for all rooftops: local shading and microclimate are not captured.
* Single campus in a subtropical climate: transfer to other climates or to utility-scale plants is untested.
* If DC power or module temperature is missing, inverter-internal losses cannot be separated from array-side losses.
* False alarms cost technician time, and missed faults cost energy. The threshold choice is a business trade-off.

**Future work.** Real maintenance logs as labels, per-rooftop irradiance sensors, panel-optimizer data for finer fault localisation, temporal deep learning (LSTM/TCN) with GPU access, and automatic retraining when data drifts.

---

## 15. Reproducibility & Academic Integrity

* Fixed seeds; pinned `requirements.txt`; notebooks run top to bottom; download instructions for both datasets; the Parquet build script is included.
* Log of datasets (with citations), libraries, external code and GenAI usage (what was generated and how it was verified).
* No fabricated results: every number in the report traces back to a notebook cell.

---

## 16. Technology Stack

* **Core:** Python, Pandas, PyArrow/Parquet, NumPy, Scikit-Learn, rdflib (Brick metadata)
* **Modelling:** XGBoost, Scikit-Learn (Ridge, Logistic Regression, Random Forest, MLP, Isolation Forest)
* **Explainability & plots:** permutation importance, SHAP, Matplotlib
* **Optional serving:** FastAPI, Pydantic, Uvicorn, React, Tailwind CSS, Recharts

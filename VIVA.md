# Viva Preparation

Short answers to the questions examiners are likely to ask. Every member should be able to answer all of Part A and their partner's part of Part B. Numbers come from `REPORT.md`.

---

## Part A: whole project (everyone)

**1. What problem does the project solve?**
Inverters fail more often than any other part of a solar system. Low output on its own does not tell you whether a cloud passed or the inverter is faulty. We compute a health score that compares actual power with the power the sunlight should give, forecast that score one hour ahead, classify it as Healthy / Warning / Faulty, and flag unusual behaviour with an anomaly detector.

**2. What data do you use?**
The HKUST rooftop PV dataset (Dryad, public domain): 44 inverters at 37 rooftop stations, 5-minute power, 1-minute weather from one campus weather station, 2021-2023, and station metadata as a Brick `.ttl` graph. GPVS-Faults (Mendeley, CC BY 4.0) is a lab dataset with real faults, used only to check the anomaly detector.

**3. What is the target? Why one hour ahead?**
Health(t) = actual power / (irradiance x kWp x 0.78) x 100, limited to 0-100. The 0.78 is a standard derate for temperature, wiring and inverter losses. We predict Health(t + 1 h). If we predicted Health(t) from the power and irradiance at time t, the model would just learn the formula back: that is target leakage (section 7.2). A one-hour forecast is leakage-free and also useful: it gives the maintenance team warning.

**4. Where do the class labels come from?**
They are engineered, not real fault records: Healthy >= 90, Warning 70-90, Faulty < 70. No maintenance logs exist for these inverters. That is the biggest limitation, which is why GPVS-Faults (real faults) is used as a separate check.

**5. How do you split the data? Why not random?**
By time: train up to June 2022, validation July-December 2022, test all of 2023. A random split would put future rows next to past rows from the same hour, so the model would see the answer. Six inverters that only start in 2023 are never used for training and form the unseen-inverter test: how would the model work on a brand-new rooftop?

**6. What happened on 14 February 2023?**
The campus irradiance sensor started reading about 1.75 times higher. We saw it in EDA: output per unit of sunlight dropped by almost half for all 44 inverters in the same week, which cannot be the inverters. Without a fix, half of 2023 (our test year) was labelled Faulty. We rescale irradiance after that date by 0.57, the median of the monthly ratios between 2023 and the same months of 2022. Caveat: the factor uses 2023 data, so we say so in the report.

**7. How do you avoid data leakage?**
- Target one hour ahead, matched by exact timestamp (never "4 rows later", which could jump overnight).
- History features use only the past (`closed="left"` rolling windows exclude the current row).
- The imputer and scaler are fitted on training rows only, inside the model pipeline.
- Tuning uses TimeSeriesSplit (each fold trains on the past, validates on the future).
- Unseen inverters never appear in training or validation.

**8. What is a baseline and why do you need one?**
The simplest reasonable prediction, with no learning. Persistence says "health in an hour = health now"; the rule baseline says "status in an hour = status now". A model is only useful if it beats them. A model that beats them by a huge margin is suspicious (it may leak).

**9. Why are Faulty recall and macro-F1 the main classification metrics, not accuracy?**
About 83 % of rows are Healthy, so "always Healthy" scores 83 % accuracy and catches zero faults. Faulty recall = share of real Faulty rows we catch (a missed fault costs energy). Macro-F1 averages F1 over the three classes, so the rare classes count as much as Healthy.

**10. How did you handle class imbalance?**
We compared three options on validation with Logistic Regression: no balancing, balanced class weights, and undersampling Healthy. Balanced weights gave the highest Faulty recall, so every classifier is trained with balanced sample weights (each class contributes equally to the loss). SMOTE was rejected: it invents rows by mixing neighbours, which makes little sense for time series.

**11. What does RMSE mean here?**
The typical size of the forecast error in health points (0-100 scale), with large errors weighted more. MAE is the plain average error.

**12. What features do the models use?**
20 features known at time t: current health and data completeness (A1), minutes offline in the past 24 h (A1), weather (irradiance, temperature, humidity, wind), time of day and year as sin/cos (B2), capacity, tilt, altitude (B2), peer deviation and 30-day self deviation (B1), and recent history: health 15 min and 1 day ago, 4-hour mean health and 4-hour mean peer deviation (A2).

**13. What is peer deviation?**
(this inverter's performance ratio - campus median performance ratio at the same time) / campus median. All inverters share the same sky, so a cloud lowers everyone and cancels out; a fault lowers only one inverter.

**14. How was each model tuned?**
All models use the same recipe (`run_model` in `src/evaluate.py`): fit with default settings, GridSearchCV with a 3-fold TimeSeriesSplit on the training period (on a 50,000-row time-ordered sample for the slower Random Forest and MLP), refit the best settings on the full training set, and report default vs tuned on validation and test.

**15. How do you explain the models?**
Permutation importance for every model: shuffle one feature on validation data and measure how much the score drops. SHAP for XGBoost: how much each feature pushed each prediction up or down.

**16. What is Isolation Forest and how did you test it?**
An unsupervised detector: random trees split the data, and points that get isolated in very few splits are unusual. No labels are needed. On HKUST we injected known faults (a 35 % power drop for 3 days, a 2-day outage) into validation and test and measured how many were caught. On GPVS-Faults we trained on fault-free windows and tested on real lab faults.

**17. What are the final results?**
- Health one hour ahead: Random Forest has the lowest validation RMSE (10.36) and test RMSE 10.45, against persistence 12.35. Ridge is close (test 10.17): the gap between the best models is small.
- Faulty one hour ahead: Logistic Regression catches the most faults (Faulty recall 0.72 validation, 0.71 test). Random Forest has the best overall balance (macro-F1 0.63 on test). The simple peer rule also reaches 0.71 recall on test.
- Unseen inverters are the hardest case for every model (Faulty recall 0.32-0.39).
- Isolation Forest: 76.8 % of injected HKUST fault rows caught with 1.1 % false alarms; 39 % of real GPVS fault windows caught with no false alarms.

**18. Why did tuning hardly help some models?**
Ridge and Logistic Regression have one setting each and 260,000 training rows, so there is little to overfit. The tree models gained most (XGBoost validation RMSE 12.05 -> 10.73). The MLP regressor was tuned on a 50,000-row sample, and those settings turned out slightly worse on the full data.

**19. What were the worst predictions?**
Health 100 now and 0 one hour later: an inverter suddenly tripping to zero output (e.g. Library inverter 2, December 2023). Nothing at time t announces it, so no model can forecast it.

**20. What are the main limitations?**
Engineered labels; one weather station for 37 rooftops; the sensor correction uses test-year data; summer evening output after 18:00 is never logged; only six unseen inverters; Zone D capacities in the metadata are inconsistent with its output.

---

## Part B: by member

### A1: data preparation, baselines, linear models
- **Brick / RDF / SPARQL:** the metadata is a graph of facts (station has inverter, inverter has panels). SPARQL is a query language for such graphs. We query it once and turn it into one table row per inverter; capacity = sum of panels x panel watts.
- **Resampling:** 5-minute power and 1-minute weather are averaged to 15 minutes. `n_valid_5min` records how many of the three 5-minute readings existed.
- **Gaps:** missing readings show up as empty rows, not missing rows. Coverage counts daylight bins only (>= 50 W/m2) because inverters are asleep at 06:00. Every inverter has at least 88 % coverage, so none is dropped.
- **Ridge:** a linear model with an L2 penalty (alpha x sum of squared weights) that keeps weights small and stable when features are correlated. Inputs are standardised so the penalty treats them equally.
- **Logistic Regression:** a linear model for classes; it outputs a probability per class with the softmax function.

### A2: preprocessing, balancing, Random Forest
- **Cleaning rules:** negative power, power above 1.1 x capacity and irradiance above 1,500 W/m2 are sensor errors and become missing. Real extremes are kept.
- **Imputation:** missing feature values are filled with the training median inside the pipeline; median and mean gave the same validation RMSE.
- **Random Forest:** bagging. Many deep decision trees, each trained on a random sample of rows and features, averaged. It captures non-linear patterns and needs no scaling.

### B1: target, feature selection, XGBoost
- **Threshold sensitivity:** the class shares for Faulty cut-offs 60-80 and Healthy cut-offs 85-95 are in `study_thresholds.csv`; 70 / 90 give about 7 % Faulty in training.
- **Feature selection:** `tilt_deg` and `multi_orientation` correlate at r = 0.97, so `multi_orientation` is dropped.
- **XGBoost:** boosting. Small trees are added one at a time, each one fitted to the remaining errors of the trees before it; the learning rate shrinks each step.

### B2: evaluation framework, MLP
- **Evaluation:** one set of metric functions and one `run_model()` recipe that every model uses, so all numbers are comparable.
- **MLP:** a neural network with hidden layers of ReLU units trained with the Adam optimiser. Inputs must be scaled. Early stopping holds out 10 % of the training rows and stops when they stop improving.
- **GPVS-Faults:** 16 lab recordings (F0 normal, F1-F7 faults). Each is cut into 0.1-second windows, and the window statistics (mean, std, min, max of DC current and voltages, RMS of AC phases) are the detector's inputs.

# Sections 9, 10 & 11: Gradient Boosted Decision Trees (XGBoost), Hyperparameter Optimization, and Model Evaluation

**Lead Author:** Partner B1 (Feature Specialist & XGBoost Track Lead)  
**Co-Contributors:** Partner B2 (Validation Architecture Lead & MLP Track Lead), Pair A (Baselines & Outage Lead)  
**Assignment Reference:** Innovative Assignment Guidelines §2 (Stages 9, 10, 11) & [`Solar_Inverter_Health_Monitoring.md`](../Solar_Inverter_Health_Monitoring.md)  
**Driver Modules:** [`src/models/xgb.py`](../src/models/xgb.py), [`src/models/iforest.py`](../src/models/iforest.py), [`notebooks/03_xgb_B1.ipynb`](../notebooks/03_xgb_B1.ipynb)

> **Correction (2026-10-21, target-leakage fix).** An earlier version of this section reported test RMSE 0.30 and Macro-F1 1.00. Those numbers came from target leakage: the model predicted Health(t) = PR(t) x 100, and three of its inputs (`peer_dev_campus`, `peer_dev_station`, `self_dev_30d`) are computed from the same PR(t), so it rebuilt the target from its own inputs (§7.2 of the design doc). XGBoost now forecasts **Health(t + 1 h)** from features at t (Option A, the same target as A1 and A2), uses B2's split functions with the held-out inverters excluded from train and validation, and fills missing values with training medians only. Every number below comes from `python -m src.models.xgb --tune` on this code.

> **Update (2026-10-21, corrected target).** The health score itself was then fixed for the whole team: irradiance after the 2023-02-14 sensor change is rescaled (factor 0.570, `prepare_data.py`) and missing AC power gives an undefined score instead of 0. The numbers below are from the re-run on that corrected target.

---

## 9.1 Theoretical Framework of Gradient Boosted Decision Trees (XGBoost)

Photovoltaic health monitoring poses a challenging tabular learning problem characterized by heterogeneous feature scales (temperatures in Celsius, irradiance reaching $1200\text{ W/m}^2$, unitless peer-ratios), severe non-linear operating regimes (inverter clipping and thermal throttling), and extreme class imbalance. While linear models fail to capture non-linear interactions and multi-layer perceptrons (MLPs) struggle with un-normalized tabular thresholds, **eXtreme Gradient Boosting (XGBoost)** provides state-of-the-art predictive power and computational efficiency.

### 9.1.1 Second-Order Taylor Approximation of the Objective Function

Given a dataset $\mathcal{D} = \{(x_i, y_i)\}$ with $n$ observations and $m$ engineered features, an ensemble of $K$ additive regression trees predicts:
$$\hat{y}_i = \sum_{k=1}^K f_k(x_i), \quad f_k \in \mathcal{F}$$
where $\mathcal{F} = \{f(x) = w_{q(x)}\}$ denotes the space of Classification and Regression Trees (CART), $q: \mathbb{R}^m \to \{1, \dots, T\}$ assigns an instance to an index of $T$ leaf nodes, and $w \in \mathbb{R}^T$ represents the leaf weight vector.

At round $t$, the objective function to minimize is:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ penalizes tree complexity ($T$ leaves) and $L_2$ leaf weights ($w_j$).

Taking a second-order Taylor expansion around the previous prediction $\hat{y}_i^{(t-1)}$:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
where the first and second-order directional gradients are:
$$g_i = \frac{\partial l(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \qquad h_i = \frac{\partial^2 l(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$$

Removing the constant term $l(y_i, \hat{y}_i^{(t-1)})$ and grouping the summation by leaf $j$ with instance set $I_j = \{i \mid q(x_i) = j\}$:
$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$

### 9.1.2 Optimal Leaf Weight and Exact Split Gain Criterion

Setting $\frac{\partial \tilde{\mathcal{L}}^{(t)}}{\partial w_j} = 0$ yields the closed-form optimal weight $w_j^*$ for leaf $j$:
$$w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Substituting $w_j^*$ back into the objective yields the corresponding optimal score:
$$\tilde{\mathcal{L}}^{(t)}(q) = - \frac{1}{2} \sum_{j=1}^T \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

When evaluating a split of a node into left ($I_L$) and right ($I_R$) candidate children where $I = I_L \cup I_R$, the exact loss reduction (gain) is:
$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$
Here, $\gamma$ serves as a hard regularization threshold; a split is rejected if the computed gain is less than $\gamma$.

### 9.1.3 Theoretical Comparison: XGBoost vs. Neural Networks (MLP) and Ridge

| Architectural Dimension | Ridge Regression (Track T1) | Multi-Layer Perceptron (Track T4) | XGBoost CART Ensembles (Track T3) |
| :--- | :--- | :--- | :--- |
| **Decision Boundary** | Global continuous linear hyperplane | Smooth compositions of continuous non-linear sigmoids/ReLUs | Orthogonal, axis-aligned hyperplanes |
| **Physics Matching** | Fails on saturation / cutoffs | Requires deep layers to approximate sharp physical step-functions | Matches physical cutoffs natively (e.g. $G \ge 100\text{ W/m}^2$, $T \ge 60^\circ\text{C}$) |
| **Multi-Scale Invariance** | Highly sensitive to unscaled inputs | Requires rigid z-score standardization ($X \sim \mathcal{N}(0, 1)$) | Scale-invariant; ordering monotonic along feature axes |
| **Feature Collinearity** | Unstable weight inflation without high $L_2$ | Redundant hidden neurons; slower convergence | Greedy split selection isolates most informative variable |
| **Inference Latency** | $\mathcal{O}(m)$ dot product ($< 0.1\text{ ms}$) | $\mathcal{O}(\sum d_l d_{l+1})$ matrix multiplication ($2\text{--}5\text{ ms}$) | $\mathcal{O}(K \cdot \text{depth})$ pointer jumps ($< 0.5\text{ ms}$) |

---

## 9.2 Formulation of Classification Strategy: Method 1 vs. Method 2

The project specification mandates comparing two distinct machine learning paradigms for the 3-class operational status prediction:

1. **Method 1 (Dedicated Multi-Class Classifier):** An `XGBClassifier` trained directly using multi-class cross-entropy loss (`multi:softprob`) with sample weights to compensate for class imbalance:
   $$w_i = \frac{N}{C \cdot N_{c(i)}}$$
   where $N$ is total training instances, $C=3$ is class count, and $N_{c(i)}$ is the frequency of class $c(i)$.
2. **Method 2 (Thresholded Continuous Regressor):** An `XGBRegressor` trained using mean squared error loss, whose continuous predictions $\hat{y}$ are converted post-hoc via fixed physical thresholds:
   $$\hat{c} = \begin{cases} \text{Healthy} & \text{if } \hat{y} \ge 90.0 \\ \text{Warning} & \text{if } 70.0 \le \hat{y} < 90.0 \\ \text{Faulty} & \text{if } \hat{y} < 70.0 \end{cases}$$

### Empirical Strategy Comparison (validation, Jul-Dec 2022, 183,667 rows)

| Method | Macro-F1 | Faulty Recall | Faulty Precision |
| :--- | :---: | :---: | :---: |
| **Method 1: Dedicated classifier (selected)** | 0.5999 | **0.6542** | 0.5025 |
| Method 2: Thresholded regressor | **0.6424** | 0.4824 | **0.8865** |

### Justification
* **Selection rule:** Faulty recall is the project's primary metric (§11.2), so Method 1 is selected. It catches 17 points more of the Faulty rows; Method 2 is far more precise and better on macro-F1 but misses more faults.
* **Why Method 2 misses faults:** the regressor minimises squared error over all rows, so its predictions are pulled towards the common Healthy range and fewer of them cross below the 70 threshold.
* **Boundary sensitivity:** near the 70 and 90 thresholds a small regression error changes the predicted class, which a dedicated classifier with weighted cross-entropy avoids.

## 5.1 Stage 5: Class Balancing for the Dedicated Classifier

Two strategies were compared on the validation set, both applied to training rows only ([`reports/tables/b1_xgb_balancing.csv`](tables/b1_xgb_balancing.csv)):

| Strategy | Train rows | Macro-F1 | Faulty Recall | Faulty Precision | Faulty PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample weights (balanced, Faulty x1.35), selected** | 261,536 | 0.5999 | **0.6542** | 0.5025 | 0.6742 |
| Healthy undersampling (2 Healthy : 1 other) | 116,943 | **0.6429** | 0.5164 | **0.7603** | **0.6940** |

* Selected on validation Faulty recall, then macro-F1 (the rule A1 and A2 used). Sample weights catch 13.8 points more of the Faulty rows; undersampling is more precise but misclassifies about twice as many Warning rows as Healthy (11,681 against 5,791).
* **SMOTE was rejected:** interpolating between rows invents telemetry that never happened and mixes rows from different days and inverters.
* Validation and test sets are never resampled.

---

## 10.1 Stage 10: Hyperparameter Tuning & Regularization Analysis

Five configurations (250 trees each) were trained on the last 20% of the training period (time-ordered, for CPU runtime) and scored on the validation set (2022-07-01 to 2022-12-31) using B2's split functions. No early stopping is used; the validation set only ranks the configurations. The best configuration is then refit on the full training split.

### 10.1.1 Hyperparameter Sweep Results

From [`reports/tables/b1_xgb_default_vs_tuned.csv`](tables/b1_xgb_default_vs_tuned.csv):

| Config ID | `max_depth` | `learning_rate` | `subsample` | `colsample_bytree` | `reg_alpha` | `reg_lambda` | Train RMSE | Val RMSE | Val MAE | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (Winner)** | **4** | **0.05** | **0.80** | **0.80** | **0.01** | **1.00** | **7.614** | **13.243** | **6.491** | **2.7** |
| Config 2 | 6 | 0.03 | 0.85 | 0.85 | 0.10 | 1.50 | 7.101 | 14.288 | 6.864 | 3.7 |
| Config 3 | 6 | 0.05 | 0.80 | 0.80 | 0.05 | 1.00 | 6.628 | 14.579 | 6.970 | 4.0 |
| Config 5 | 8 | 0.05 | 0.70 | 0.70 | 0.50 | 2.50 | 5.734 | 14.744 | 7.152 | 5.0 |
| Config 4 | 8 | 0.02 | 0.80 | 0.75 | 0.20 | 2.00 | 6.607 | 14.808 | 7.208 | 4.7 |

### 10.1.2 Key Insights from the Optimization Study
1. **Shallower trees generalise better.** The depth-4 configuration has the lowest validation RMSE (13.24); the depth-6 and depth-8 configurations reach lower training RMSE (down to 5.73) but transfer worse to the next six months (14.29-14.81).
2. **The gain from tuning is modest.** The spread between the best and worst configuration is 1.6 RMSE points. On the full training split the selected model reaches validation RMSE 11.38 (§11.1.1).
3. **Limitation:** only five hand-picked configurations were searched, and the regressor is tuned on a 20% sample. A wider RandomizedSearch with early stopping is a possible extension.

---

## 11.1 Stage 11: Comprehensive Model Evaluation & Benchmark Results

All evaluations strictly follow B2's validation protocol:
* **Chronological Split:** Train ($\le 2022\text{-}06\text{-}30$), Validation ($2022\text{-}07\text{-}01$ to $2022\text{-}12\text{-}31$), and Out-of-Time Test ($2023\text{-}01\text{-}01$ to $2023\text{-}12\text{-}31$).
* **Spatial Split:** 7 held-out unseen stations (`sq_block_p_inverter`, `shaw_auditorium_inverter`, `ug_hall8_inverter`, `ug_hall9_inverter`, `zone_a2_inverter_2`, `zone_a2_inverter_3`, `sq14_inverter`); 6 of them have data (`sq14_inverter` is not one of the 44 inverters).

### 11.1.1 XGBoost Performance Summary

Target: Health(t + 1 h). Features: the 13 Stage 7 features plus Health(t). Persistence (Health(t + 1 h) = Health(t)) is the bar a forecaster must beat.

| Split | Rows | XGB RMSE | XGB MAE | $R^2$ | Persistence RMSE | Macro-F1 | Faulty Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Train ($\le$ 2022-06) | 261,536 | 8.69 | 4.42 | 0.625 | 12.11 | 0.621 | 0.768 |
| Validation (2022 H2) | 183,667 | **11.38** | 6.10 | 0.600 | 13.04 | 0.600 | 0.654 |
| Test (2023) | 369,090 | **10.58** | 5.65 | 0.555 | 12.30 | 0.564 | 0.641 |
| Test before 14 Feb 2023 | 34,984 | **10.46** | | | 10.90 | 0.674 | 0.821 |
| Test after 14 Feb 2023 | 334,106 | **10.60** | | | 12.44 | 0.552 | 0.619 |
| Unseen inverters (2023) | 37,060 | **9.90** | 4.75 | 0.252 | 10.73 | 0.414 | 0.420 |

**Reading the results.**
1. **XGBoost beats persistence on every split**, including the test year after the sensor change (10.60 against 12.44). Before the target fix it fell behind persistence after 2023-02-14 (20.69 against 16.95); that gap came from the shifted labels, not from the model.
2. **Against the other tracks:** validation RMSE 11.38 is better than A1's Ridge (11.95) and persistence (13.04) and behind A2's Random Forest (9.91), which is the project's selected regressor (`reports/Final_Model_Comparison.md`).
3. **Unseen inverters:** RMSE 9.90 against persistence 10.73, so the model transfers to new rooftops, though classification there is weak (Faulty recall 0.42).

---

## 11.2 Interpretability via Tree-SHAP (Shapley Additive Explanations)

To eliminate the "black-box" risk in critical electrical infrastructure, Tree-SHAP was applied to extract both global feature rankings and individual localized prediction explanations.

### 11.2.1 Global Attribution: Beeswarm Summary Plot

The global beeswarm plot ([`reports/figures/b1_shap_summary.png`](figures/b1_shap_summary.png)) displays the distribution of impact each feature has on the model's output.

Mean absolute SHAP value on 1,500 test rows (health points):

| Rank | Feature | Mean \|SHAP\| |
| :---: | :--- | :---: |
| 1 | `peer_dev_campus` | 3.15 |
| 2 | `health_now` (Health(t)) | 2.56 |
| 3 | `self_dev_30d` | 0.75 |
| 4 | `day_cos` | 0.75 |
| 5 | `hour_sin` | 0.73 |
| 6-11 | `capacity_kwp`, `peer_dev_station`, `day_sin`, `altitude_m`, `thermal_stress`, `tilt_deg` | 0.26-0.62 |
| — | `orient_mixed`, `orient_south`, `orient_east_west` | about 0 |

1. **Peer deviation and current health lead.** How the inverter compares with the campus right now is now the strongest input, ahead of current health. A likely reason: with outage bins no longer scored as health 0, current health is less often an extreme value, so the peer comparison carries more of the signal.
2. **Self deviation and the seasonal and hour-of-day terms** come next, reflecting the inverter's own recent history and the daily and annual cycle.
3. `orient_south` and `orient_east_west` contribute nothing because they are constant 0 for every inverter (a known metadata issue).

### 11.2.2 Local Explanation: Severe Prediction Waterfall Plot

The local waterfall explanation ([`reports/figures/b1_shap_waterfall.png`](figures/b1_shap_waterfall.png)) decomposes a severe prediction failure into individual additive contributions:
$$\hat{f}(x) = \mathbb{E}[f(X)] + \sum_{j=1}^M \phi_j$$
The plot shows the worst error in the SHAP sample (`zone_a6_inverter`, 2023-08-30 14:00):
* **Base value $\mathbb{E}[f(X)]$:** 94.73.
* **Largest contributions:** `health_now` +1.48 (value 100), `peer_dev_campus` +1.09 (value -0.06), `hour_sin` +0.51, `capacity_kwp` +0.46.
* **Model output:** 97.93. **Actual Health(t + 1 h):** 33.8.
* Nothing at t pointed to the drop: health was 100 and the inverter was in line with its peers. A fall to 34 within the hour is not predictable from these features.

---

## 11.3 Failure Case Diagnosis: Top 3 Worst Prediction Residuals

From [`reports/tables/b1_xgb_failure_cases.csv`](tables/b1_xgb_failure_cases.csv) (largest errors in the 1,500-row SHAP sample of the 2023 test set). The table reports observed context only; the dataset has no maintenance logs, so causes are hypotheses.

| Rank | Timestamp | Inverter ID | Health(t) | Actual Health(t+1h) | Predicted | Error | Irradiance (W/m²) | Ambient Temp | Valid 5-min readings |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2023-08-30 14:00 | `zone_a6_inverter` | 100.0 | 33.8 | 97.93 | 64.13 | 589.6 | 35.8 °C | 3 of 3 |
| 2 | 2023-07-20 14:15 | `zone_a4_inverter` | 100.0 | 34.8 | 97.72 | 62.90 | 542.4 | 30.7 °C | 3 of 3 |
| 3 | 2023-01-17 11:00 | `indoor_sports_centre_inverter` | 10.9 | 7.1 | 69.09 | 62.04 | 295.6 | 18.4 °C | 3 of 3 |

**What the cases show.**
1. **Sudden afternoon drops (cases 1 and 2):** health is 100 at t and falls to about 34 within the hour, on hot summer afternoons. Nothing in the inputs at t signals this, so no forecaster built on these features can predict it; A1 found the same pattern for Ridge.
2. **Persistent low health under-predicted (case 3):** health is 11 at t and stays low, yet the model predicts 69. Very low current health is rare in training (7% of rows are Faulty), which likely makes the model pull such rows back towards the typical range. This case is before the sensor change, so it is not a sensor effect.

---

## 11.4 Track T5: Unsupervised Anomaly Detection on GPVS-Faults Benchmark

As part of the joint Pair B deliverable (B1/B2), unsupervised anomaly detection was verified using an **Isolation Forest** pipeline on the real-world **GPVS-Faults physical benchmark dataset** ([`src/models/iforest.py`](../src/models/iforest.py)).

### 11.4.1 Real Hardware Fault Benchmark Setup
Unlike synthetic fault injection, GPVS-Faults consists of actual physical laboratory experiments including **open-circuit string disconnections, line-line short circuits, and localized shading**. The dataset contains 1,000 healthy operating states and 100 physical hardware faults (contamination rate $\approx 9.1\%$).

### 11.4.2 Benchmark Results & Detection Analysis

* **Detection Recall (TPR):** **$68.00\%$** (Caught 17 of 25 test physical faults).
* **Precision:** **$77.27\%$** (High confidence; only 5 false alarms out of 250 normal points).
* **F1-Score:** **$0.7234$**
* **False Alarm Rate (FPR):** **$2.00\%$** (Extremely clean operational baseline).
* **Execution Latency:** **$0.09\text{ seconds}$**.

The dual evaluation figure ([`reports/figures/b1_iforest_gpvs_detection.png`](figures/b1_iforest_gpvs_detection.png)) illustrates clear multimodal separation between healthy telemetry and severe physical anomalies along the decision function score.

---

## 11.5 The 4-Track Master Benchmark Synthesis

Final rows in `results/results.csv` after the 2026-10-21 re-run on the corrected target. All supervised rows forecast Health(t + 1 h) or status at t + 1 h. The full comparison, including before/after 14 Feb and the anomaly detectors, is in `reports/Final_Model_Comparison.md`.

| Track | Student | Model | Val RMSE | Test RMSE | Unseen RMSE | Test Macro-F1 | Test Faulty Recall |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | A1 | Persistence | 13.04 | 12.30 | 10.73 | — | — |
| T1 | A1 | Ridge / Logistic | 11.95 | 11.29 | 9.71 | 0.53 | **0.70** |
| Baseline | A1 | Rule: status of Health(t) | — | — | — | 0.59 | 0.60 |
| T2 | A2 | Random Forest | **9.91** | **9.73** | **9.07** | **0.61** | 0.57 |
| **T3** | **B1** | **XGBoost (tuned)** | 11.38 | 10.58 | 9.90 | 0.56 | 0.64 |
| T4 | B2 | MLP | 13.71 | 11.99 | 14.02 | 0.52 | 0.34 |
| T5 | B1 / B2 | Isolation Forest (GPVS-Faults) | — | — | — | Recall 0.68, F1 0.72 | — |

### Conclusions
1. **Every supervised model beats persistence on the 2023 test year** once the target is corrected. A2's Random Forest is the strongest regressor on validation and test.
2. **XGBoost's contribution:** the second-best regressor (test RMSE 10.58) and the second-highest Faulty recall (0.64, after A1's Logistic Regression at 0.70), with the best Faulty recall of any model on the test year before the sensor change (0.82). It is a strong middle ground between the Random Forest's accuracy and Logistic Regression's fault coverage.
3. **Isolation Forest** (Pair B, GPVS-Faults) remains a label-free cross-check; its numbers are on a different dataset and are not comparable with the HKUST rows.

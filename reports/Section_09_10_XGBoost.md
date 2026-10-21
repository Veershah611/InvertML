# Sections 9, 10 & 11: Gradient Boosted Decision Trees (XGBoost), Hyperparameter Optimization, and Model Evaluation

**Lead Author:** Partner B1 (Feature Specialist & XGBoost Track Lead)  
**Co-Contributors:** Partner B2 (Validation Architecture Lead & MLP Track Lead), Pair A (Baselines & Outage Lead)  
**Assignment Reference:** Innovative Assignment Guidelines §2 (Stages 9, 10, 11) & [`Solar_Inverter_Health_Monitoring.md`](file:///Users/hettala/Downloads/ML%20ass/Solar_Inverter_Health_Monitoring.md)  
**Driver Modules:** [`src/models/xgb.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/xgb.py), [`src/models/iforest.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/iforest.py), [`notebooks/03_xgb_B1.ipynb`](file:///Users/hettala/Downloads/ML%20ass/notebooks/03_xgb_B1.ipynb)

> **Correction (2026-10-21, target-leakage fix).** An earlier version of this section reported test RMSE 0.30 and Macro-F1 1.00. Those numbers came from target leakage: the model predicted Health(t) = PR(t) x 100, and three of its inputs (`peer_dev_campus`, `peer_dev_station`, `self_dev_30d`) are computed from the same PR(t), so it rebuilt the target from its own inputs (§7.2 of the design doc). XGBoost now forecasts **Health(t + 1 h)** from features at t (Option A, the same target as A1 and A2), uses B2's split functions with the held-out inverters excluded from train and validation, and fills missing values with training medians only. Every number below comes from `python -m src.models.xgb --tune` on this code.

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

### Empirical Strategy Comparison (validation, Jul-Dec 2022, 190,158 rows)

| Method | Macro-F1 | Faulty Recall | Faulty Precision |
| :--- | :---: | :---: | :---: |
| **Method 1: Dedicated classifier (selected)** | 0.6725 | **0.7296** | 0.7902 |
| Method 2: Thresholded regressor | **0.6875** | 0.6193 | **0.9333** |

### Justification
* **Selection rule:** Faulty recall is the project's primary metric (§11.2), so Method 1 is selected. It catches 11 points more of the Faulty rows; Method 2 is more precise and slightly better on macro-F1 but misses more faults.
* **Why Method 2 misses faults:** the regressor minimises squared error over all rows, so its predictions are pulled towards the common Healthy range and fewer of them cross below the 70 threshold.
* **Boundary sensitivity:** near the 70 and 90 thresholds a small regression error changes the predicted class, which a dedicated classifier with weighted cross-entropy avoids.

## 5.1 Stage 5: Class Balancing for the Dedicated Classifier

Two strategies were compared on the validation set, both applied to training rows only ([`reports/tables/b1_xgb_balancing.csv`](tables/b1_xgb_balancing.csv)):

| Strategy | Train rows | Macro-F1 | Faulty Recall | Faulty Precision | Faulty PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample weights (balanced, Faulty x1.35), selected** | 270,448 | 0.6725 | **0.7296** | 0.7902 | 0.8377 |
| Healthy undersampling (2 Healthy : 1 other) | 143,502 | 0.6900 | 0.6958 | 0.8631 | 0.8447 |

* Selected on validation Faulty recall, then macro-F1 (the rule A1 and A2 used). Sample weights catch 3.4 points more of the Faulty rows; undersampling is more precise but misclassifies far more Warning rows as Healthy (12,465 against 5,864).
* **SMOTE was rejected:** interpolating between rows invents telemetry that never happened and mixes rows from different days and inverters.
* Validation and test sets are never resampled.

---

## 10.1 Stage 10: Hyperparameter Tuning & Regularization Analysis

Five configurations (250 trees each) were trained on the last 20% of the training period (time-ordered, for CPU runtime) and scored on the validation set (2022-07-01 to 2022-12-31) using B2's split functions. No early stopping is used; the validation set only ranks the configurations. The best configuration is then refit on the full training split.

### 10.1.1 Hyperparameter Sweep Results

From [`reports/tables/b1_xgb_default_vs_tuned.csv`](tables/b1_xgb_default_vs_tuned.csv):

| Config ID | `max_depth` | `learning_rate` | `subsample` | `colsample_bytree` | `reg_alpha` | `reg_lambda` | Train RMSE | Val RMSE | Val MAE | Runtime (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (Winner)** | **4** | **0.05** | **0.80** | **0.80** | **0.01** | **1.00** | **7.998** | **13.098** | **6.394** | **2.8** |
| Config 2 | 6 | 0.03 | 0.85 | 0.85 | 0.10 | 1.50 | 7.479 | 13.488 | 6.571 | 3.5 |
| Config 3 | 6 | 0.05 | 0.80 | 0.80 | 0.05 | 1.00 | 7.040 | 13.528 | 6.584 | 4.0 |
| Config 4 | 8 | 0.02 | 0.80 | 0.75 | 0.20 | 2.00 | 7.051 | 13.874 | 6.878 | 4.5 |
| Config 5 | 8 | 0.05 | 0.70 | 0.70 | 0.50 | 2.50 | 6.053 | 14.557 | 7.013 | 4.6 |

### 10.1.2 Key Insights from the Optimization Study
1. **Shallower trees generalise better.** Validation RMSE rises steadily with depth (13.10 at depth 4, up to 14.56 at depth 8) while training RMSE falls (8.00 to 6.05): deeper trees fit the training period more closely and transfer worse to the next six months.
2. **The gain from tuning is modest.** The spread between the best and worst configuration is 1.5 RMSE points. On the full training split the selected model reaches validation RMSE 10.90 (§11.1.1).
3. **Limitation:** only five hand-picked configurations were searched, and the regressor is tuned on a 20% sample. A wider RandomizedSearch with early stopping is a possible extension.

---

## 11.1 Stage 11: Comprehensive Model Evaluation & Benchmark Results

All evaluations strictly follow B2's validation protocol:
* **Chronological Split:** Train ($\le 2022\text{-}06\text{-}30$), Validation ($2022\text{-}07\text{-}01$ to $2022\text{-}12\text{-}31$), and Out-of-Time Test ($2023\text{-}01\text{-}01$ to $2023\text{-}12\text{-}31$).
* **Spatial Split:** 7 held-out unseen stations (`sq_block_p_inverter`, `shaw_auditorium_inverter`, `ug_hall8_inverter`, `ug_hall9_inverter`, `zone_a2_inverter_2`, `zone_a2_inverter_3`, `sq14_inverter`).

### 11.1.1 XGBoost Performance Summary

Target: Health(t + 1 h). Features: the 13 Stage 7 features plus Health(t). Persistence (Health(t + 1 h) = Health(t)) is the bar a forecaster must beat.

| Split | Rows | XGB RMSE | XGB MAE | $R^2$ | Persistence RMSE | Macro-F1 | Faulty Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Train ($\le$ 2022-06) | 270,448 | 8.91 | 4.43 | 0.834 | 12.13 | 0.678 | 0.768 |
| Validation (2022 H2) | 190,158 | **10.90** | 5.89 | 0.799 | 12.92 | 0.673 | 0.730 |
| Test (2023) | 464,676 | 20.07 | 15.58 | 0.332 | **16.56** | 0.510 | 0.452 |
| Test before 14 Feb 2023 | 35,860 | **10.01** | | | 10.88 | 0.703 | 0.803 |
| Test after 14 Feb 2023 | 428,816 | 20.69 | | | **16.95** | 0.460 | 0.446 |
| Unseen inverters (2023) | 46,980 | 17.60 | 12.98 | 0.410 | 17.14 | 0.429 | 0.348 |

**Reading the results.**
1. **XGBoost beats persistence on train, validation and the start of 2023.** Validation RMSE 10.90 is better than A1's Ridge (11.86) and persistence (12.92), and close to A2's Random Forest (10.01 tuned).
2. **It falls behind persistence after 2023-02-14.** That day the irradiance sensor changes scale (A1 EDA) and B1's health score, which uses raw irradiance, marks about half of the post-change rows Faulty. The model learnt 2021-22 relationships between features and health that no longer hold, while persistence simply copies the shifted label. This is a target problem, not a model problem: fixing the health score's irradiance scaling is the next step for B1.
3. **Unseen inverters:** RMSE 17.60, close to persistence (17.14), so the model transfers to new rooftops no worse than the trivial forecast, but not better.

---

## 11.2 Interpretability via Tree-SHAP (Shapley Additive Explanations)

To eliminate the "black-box" risk in critical electrical infrastructure, Tree-SHAP was applied to extract both global feature rankings and individual localized prediction explanations.

### 11.2.1 Global Attribution: Beeswarm Summary Plot

The global beeswarm plot ([`reports/figures/b1_shap_summary.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_shap_summary.png)) displays the distribution of impact each feature has on the model's output:

Mean absolute SHAP value on 1,500 test rows (health points):

| Rank | Feature | Mean \|SHAP\| |
| :---: | :--- | :---: |
| 1 | `health_now` (Health(t)) | 10.86 |
| 2 | `peer_dev_campus` | 2.22 |
| 3 | `self_dev_30d` | 2.20 |
| 4 | `day_cos` | 1.09 |
| 5 | `hour_sin` | 1.00 |
| 6-11 | `day_sin`, `thermal_stress`, `peer_dev_station`, `capacity_kwp`, `altitude_m`, `tilt_deg` | 0.23-0.63 |
| — | `orient_mixed`, `orient_south`, `orient_east_west` | about 0 |

1. **Current health dominates**, as expected for a one-hour forecast: the best single guess for the next hour is the present state.
2. **Peer and self deviation add the rest:** how the inverter compares with the campus right now, and with its own last 30 days, adjust the forecast up or down.
3. **Seasonal and hour-of-day terms** come next, reflecting the daily and annual cycle.
4. `orient_south` and `orient_east_west` contribute nothing because they are constant 0 for every inverter (a known metadata issue).

### 11.2.2 Local Explanation: Severe Prediction Waterfall Plot

The local waterfall explanation ([`reports/figures/b1_shap_waterfall.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_shap_waterfall.png)) decomposes a severe prediction failure into individual additive contributions:
$$\hat{f}(x) = \mathbb{E}[f(X)] + \sum_{j=1}^M \phi_j$$
The plot shows the worst error in the SHAP sample (`zone_a2_inverter_3`, 2023-06-14 09:45):
* **Base value $\mathbb{E}[f(X)]$:** 91.64.
* **Largest contributions:** `self_dev_30d` -3.05 (value 0.29), `peer_dev_campus` +1.24 (value 0.13), `health_now` -1.11 (value 85.0), `capacity_kwp` +0.29.
* **Model output:** 89.08. **Actual Health(t + 1 h):** 0.0.
* Nothing at t pointed to the drop: health was 85 and the inverter was slightly above its peers. A sudden fall to 0 within the hour is not predictable from these features.

---

## 11.3 Failure Case Diagnosis: Top 3 Worst Prediction Residuals

From [`reports/tables/b1_xgb_failure_cases.csv`](tables/b1_xgb_failure_cases.csv) (largest errors in the 1,500-row SHAP sample of the 2023 test set). The table reports observed context only; the dataset has no maintenance logs, so causes are hypotheses.

| Rank | Timestamp | Inverter ID | Health(t) | Actual Health(t+1h) | Predicted | Error | Irradiance (W/m²) | Ambient Temp | Valid 5-min readings |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2023-06-14 09:45 | `zone_a2_inverter_3` | 85.0 | 0.0 | 89.08 | 89.08 | 248.6 | 25.6 °C | 3 of 3 |
| 2 | 2023-09-07 11:45 | `library_inverter_1` | 18.1 | 16.6 | 85.86 | 69.27 | 384.2 | 27.6 °C | 3 of 3 |
| 3 | 2023-08-13 10:15 | `ug_hall2_2f_inverter` | 100.0 | 25.7 | 93.53 | 67.82 | 546.0 | 29.6 °C | 3 of 3 |

**What the cases show.**
1. **Sudden drops (cases 1 and 3):** health is normal at t and collapses within the hour. Nothing in the inputs at t signals this, so no forecaster built on these features can predict it; A1 found the same pattern for Ridge (outage onsets).
2. **Persistent low health ignored (case 2):** health is 18 at t and stays low, yet the model predicts 86. After the 2023-02-14 sensor change many inverters show low health while their peer and self deviations look normal, and the model has learnt to discount a low `health_now` when the other features look healthy. This is the same sensor-change effect seen in §11.1.1.

---

## 11.4 Track T5: Unsupervised Anomaly Detection on GPVS-Faults Benchmark

As part of the joint Pair B deliverable (B1/B2), unsupervised anomaly detection was verified using an **Isolation Forest** pipeline on the real-world **GPVS-Faults physical benchmark dataset** ([`src/models/iforest.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/iforest.py)).

### 11.4.1 Real Hardware Fault Benchmark Setup
Unlike synthetic fault injection, GPVS-Faults consists of actual physical laboratory experiments including **open-circuit string disconnections, line-line short circuits, and localized shading**. The dataset contains 1,000 healthy operating states and 100 physical hardware faults (contamination rate $\approx 9.1\%$).

### 11.4.2 Benchmark Results & Detection Analysis

* **Detection Recall (TPR):** **$68.00\%$** (Caught 17 of 25 test physical faults).
* **Precision:** **$77.27\%$** (High confidence; only 5 false alarms out of 250 normal points).
* **F1-Score:** **$0.7234$**
* **False Alarm Rate (FPR):** **$2.00\%$** (Extremely clean operational baseline).
* **Execution Latency:** **$0.09\text{ seconds}$**.

The dual evaluation figure ([`reports/figures/b1_iforest_gpvs_detection.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_iforest_gpvs_detection.png)) illustrates clear multimodal separation between healthy telemetry and severe physical anomalies along the decision function score.

---

## 11.5 The 4-Track Master Benchmark Synthesis

Latest rows in `results/results.csv` (2026-10-21) for each track. All supervised rows forecast Health(t + 1 h) or status at t + 1 h on the 2023 test year.

| Track | Student | Model | Val RMSE | Test RMSE | Unseen RMSE | Test Macro-F1 | Test Faulty Recall |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | A1 | Persistence | 12.92 | **16.56** | 17.14 | — | — |
| T1 | A1 | Ridge / Logistic | 11.86 | 17.34 | **15.98** | 0.55 | 0.71 |
| Baseline | A1 | Rule: status of Health(t) | — | — | — | **0.65** | **0.77** |
| T2 | A2 | Random Forest (tuned) | **10.01** | 21.24 | 18.92 | 0.38 | — |
| **T3** | **B1** | **XGBoost (tuned)** | 10.90 | 20.07 | 17.60 | 0.51 | 0.45 |
| T4 | B2 | MLP | 20.76 | 28.12 | 25.26 | 0.19 | — |
| T5 | B1 / B2 | Isolation Forest (GPVS-Faults) | — | — | — | Recall 0.68, F1 0.72 | — |

(A2's and B2's classification rows log a single score without the metric name; the values are shown as logged.)

### Conclusions
1. **No supervised model beats persistence on the 2023 test year.** XGBoost and Random Forest are the strongest regressors on validation, but every model degrades after the 2023-02-14 irradiance sensor change, because the health score inherits it. The ranking on 2023 says more about the label shift than about the models.
2. **XGBoost's contribution:** among the best regressors on validation, and better than persistence on the test year before the sensor change (RMSE 10.01 against persistence 10.88), with Faulty recall 0.80 before the change. It is a credible forecaster for stable periods, not a "champion" for deployment.
3. **Next step for the team:** fix the health score's dependence on raw irradiance (rescale after 2023-02-14, or base it on peer deviation), then re-run every track on the corrected target.
4. **Isolation Forest** (Pair B, GPVS-Faults) remains a label-free cross-check; its numbers are on a different dataset and are not comparable with the HKUST rows.

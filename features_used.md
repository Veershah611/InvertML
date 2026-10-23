# Features Used Across Models in InvertML

This document provides a comprehensive catalog of all features used in each machine learning model and baseline across the **InvertML** solar inverter health monitoring pipeline. For each model family, the exact feature set, feature count, target variable, and physical rationale are documented.

---

## 1. Quick Reference: Model-to-Feature Summary

| Track | Model Family | Task | Owner | # Features | Feature Set Summary | Primary Source Files |
|---|---|---|---|:---:|---|---|
| **Baselines** | Persistence Regressor | Regression | A1 | **1** | Current health score `health_now` | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py) |
| **Baselines** | Majority Class Classifier | Classification | A1 | **0** | No feature inputs (predicts prior dominant class) | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py) |
| **Baselines** | Status of Health(t) Rule | Classification | A1 | **1** | Current health score `health_now` evaluated against fixed thresholds | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py) |
| **Baselines** | Peer Deviation Rule | Classification | A1 | **2** | Campus peer deviation `peer_dev_campus` + outage status `is_offline` | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py) |
| **Track T1** | Ridge Regression | Regression | A1 | **18** | Current health, sample counts, 4 weather signals, 4 cyclical time, 6 station metadata, 2 outage features | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py) |
| **Track T1** | Logistic Regression | Classification | A1 | **18** | Same 18 features as Ridge; fitted with balanced class weights | [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py), [`src/models/linear_balancing.py`](file:///home/jemil/InvertML/src/models/linear_balancing.py) |
| **Track T2** | Random Forest (Regressor & Classifier) | Regression & Classification | A2 | **39** | 9 base telemetry/weather, 4 cyclical time, 1 categorical azimuth, 7 B1 core physical features, 6 exact-time lags, 12 past-only rolling statistics | [`src/models/rf.py`](file:///home/jemil/InvertML/src/models/rf.py), [`src/preprocess.py`](file:///home/jemil/InvertML/src/preprocess.py) |
| **Track T3** | XGBoost (`tree_method="hist"`) | Regression & Classification | B1 | **14** | `health_now` + 13 Stage 7 collinearity-pruned features (3 cyclical time, 6 metadata, 2 peer deviations, 1 self deviation, 1 thermal stress) | [`src/models/xgb.py`](file:///home/jemil/InvertML/src/models/xgb.py), [`src/features.py`](file:///home/jemil/InvertML/src/features.py) |
| **Track T4** | Multi-Layer Perceptron (MLP) | Regression & Classification | B2 | **14** | Current health, 3 physical telemetry signals, 4 cyclical time encodings, 6 station metadata features | [`src/models/mlp.py`](file:///home/jemil/InvertML/src/models/mlp.py), [`src/models/mlp_balancing.py`](file:///home/jemil/InvertML/src/models/mlp_balancing.py) |
| **Track T5** | Isolation Forest (HKUST Inverter Telemetry) | Unsupervised Anomaly Detection | Pair A (A1/A2) | **3 or 4** | Baseline (3): irradiance, temperature, specific yield. Extended (4, selected): + `peer_dev_campus` | [`src/models/iforest_a1.py`](file:///home/jemil/InvertML/src/models/iforest_a1.py), [`src/models/iforest_a2.py`](file:///home/jemil/InvertML/src/models/iforest_a2.py) |
| **Track T5** | Isolation Forest (GPVS-Faults Lab Benchmark) | Unsupervised Anomaly Detection | Pair B (B1/B2) | **18** | High-frequency 0.1 s window stats: 12 DC metrics (mean/std/min/max of Ipv, Vpv, Vdc) + 6 AC RMS phase metrics (ia, ib, ic, va, vb, vc) | [`src/models/iforest.py`](file:///home/jemil/InvertML/src/models/iforest.py), [`src/evaluate.py`](file:///home/jemil/InvertML/src/evaluate.py) |

---

## 2. Features Used by Model Family

### 2.1 Baseline Models (Track T1 Baselines — Partner A1)

Baseline models provide the mandatory non-trivial performance benchmarks against which all supervised learning models are evaluated.

* **Target:** Health forecast $\text{Health}(t + 1\text{ h})$ (Regression) or Operational Status at $t + 1\text{ h}$ (Classification: `Healthy` $\ge 90$, `Warning` $70\text{--}90$, `Faulty` $< 70$).
* **Implementation:** [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py)

#### 1. Persistence Model (Regression)
* **Features Used (1):**
  1. `health_now`: Continuous health score of the inverter at current time $t$.
* **Explanation:** Assumes the health state of the inverter is static across the 1-hour horizon: $\widehat{\text{Health}}(t + 1\text{ h}) = \text{Health}(t)$. This serves as the primary benchmark that any predictive regressor must outperform.

#### 2. Majority Class Baseline (Classification)
* **Features Used (0):** None.
* **Explanation:** Uses `sklearn.dummy.DummyClassifier(strategy="most_frequent")` to unconditionally predict the dominant class observed in the training split (`Healthy`).

#### 3. Status of Health(t) Rule (Classification)
* **Features Used (1):**
  1. `health_now`: Current continuous health score at time $t$.
* **Explanation:** Maps current health into operational status via standard thresholds:
  $$\text{Status}(t) = \begin{cases} \text{Healthy} & \text{if } \text{Health}(t) \ge 90 \\ \text{Warning} & \text{if } 70 \le \text{Health}(t) < 90 \\ \text{Faulty} & \text{if } \text{Health}(t) < 70 \end{cases}$$
  Assumes status persists unchanged into $t + 1\text{ h}$.

#### 4. Campus Peer Deviation Rule (Classification)
* **Features Used (2):**
  1. `peer_dev_campus`: Normalized deviation of the inverter's performance ratio relative to the campus-wide peer median at time $t$.
  2. `is_offline`: Indicator flag for missing power or zero valid telemetry readings.
* **Explanation:** Employs empirical thresholds tuned on the training partition: if an inverter is offline, it is flagged as `Faulty`; otherwise, if `peer_dev_campus` $< -0.40 \implies \text{Faulty}$, if $< -0.225 \implies \text{Warning}$, else $\text{Healthy}$.

---

### 2.2 Linear Models (Track T1 — Partner A1)

Ridge Regression and Logistic Regression provide transparent, interpretable linear models that test whether degradation dynamics can be captured by linear hyperplanes.

* **Target:** $\text{Health}(t + 1\text{ h})$ (Ridge) and $\text{Status}(t + 1\text{ h})$ (Logistic Regression).
* **Pipeline:** Training-fitted mean imputation $\to$ [`StandardScaler()`](file:///home/jemil/InvertML/src/models/linear.py#L30) $\to$ Model (`Ridge` with $\alpha=0.01$ or `LogisticRegression` with $C=10$, `class_weight="balanced"`).
* **Implementation:** [`src/models/linear.py`](file:///home/jemil/InvertML/src/models/linear.py), [`src/models/linear_balancing.py`](file:///home/jemil/InvertML/src/models/linear_balancing.py)
* **Total Features Used (18):**

| # | Feature Name | Category | Brief Explanation |
|:---:|:---|:---|:---|
| 1 | `health_now` | Target History | Current instantaneous health score at time $t$ ($0\text{--}100$). Strongest autoregressive baseline. |
| 2 | `n_valid_5min` | Data Quality | Number of valid 5-minute telemetry intervals in the 15-minute resampled bin ($0, 1, 2, 3$). |
| 3 | `irradiance_wm2` | Weather | On-site horizontal plane solar irradiance ($\text{W/m}^2$). Primary driver of theoretical PV output. |
| 4 | `temp_c` | Weather | Ambient air temperature ($^\circ\text{C}$). Drives thermal degradation and panel efficiency drop. |
| 5 | `rh_pct` | Weather | Relative humidity percentage ($0\text{--}100\%$). Correlates with cloudiness and atmospheric scattering. |
| 6 | `wind_speed_ms` | Weather | Wind speed in meters per second ($\text{m/s}$). Governs convective cooling of rooftop modules. |
| 7 | `hour_sin` | Cyclical Time | $\sin(2\pi \cdot t_{\text{hour}} / 24)$. Continuous circular projection of diurnal solar angle. |
| 8 | `hour_cos` | Cyclical Time | $\cos(2\pi \cdot t_{\text{hour}} / 24)$. Captures diurnal solar zenith elevation without midnight boundary discontinuity. |
| 9 | `day_sin` | Cyclical Time | $\sin(2\pi \cdot t_{\text{day}} / 365.25)$. Continuous circular encoding of annual seasonal progression. |
| 10 | `day_cos` | Cyclical Time | $\cos(2\pi \cdot t_{\text{day}} / 365.25)$. Captures seasonal winter/summer solar declination. |
| 11 | `capacity_kwp` | Station Metadata | Nameplate DC array peak rated power ($\text{kWp}$), standardizing expected array scale. |
| 12 | `tilt_deg` | Station Metadata | Rooftop panel tilt inclination slope in degrees (defaults to campus median $15.0^\circ$). |
| 13 | `altitude_m` | Station Metadata | Installation height above sea level in meters ($20\text{--}120\text{ m}$). |
| 14 | `orient_south` | Station Metadata | Binary indicator ($1$ if azimuth is south-facing $135^\circ\text{--}225^\circ$, else $0$). Peak midday exposure. |
| 15 | `orient_east_west` | Station Metadata | Binary indicator ($1$ if azimuth is east/west-facing $45^\circ\text{--}135^\circ$ or $225^\circ\text{--}315^\circ$, else $0$). |
| 16 | `orient_mixed` | Station Metadata | Binary indicator ($1$ if multi-string rooftop contains mixed orientations/flat roofs, else $0$). |
| 17 | `is_offline` | Outage Status | Binary flag indicating communication dropout or missing generation during the current bin. |
| 18 | `minutes_offline_24h`| Outage History | Cumulative minutes an inverter was offline during daylight ($G \ge 50\text{ W/m}^2$) over the preceding 24 hours. |

---

### 2.3 Random Forest (Track T2 — Partner A2)

Random Forest represents the bagging paradigm of decision tree ensembles, capable of modelling complex non-linear interactions, handling outliers gracefully, and providing robust feature importances and TreeSHAP explanations.

* **Target:** `health_score_lead4` (Health at $t + 1\text{ h}$) and `status_lead4` (Status at $t + 1\text{ h}$).
* **Pipeline:** [`build_preprocessor()`](file:///home/jemil/InvertML/src/preprocess.py#L225) (Mean imputation with missingness indicators + OneHotEncoder for categorical `azimuth`) $\to$ `RandomForestRegressor` / `RandomForestClassifier` (`n_estimators=80`, `max_depth=16`, `min_samples_leaf=10`, `max_features=0.7`, `class_weight="balanced"`).
* **Implementation:** [`src/models/rf.py`](file:///home/jemil/InvertML/src/models/rf.py), [`src/preprocess.py`](file:///home/jemil/InvertML/src/preprocess.py)
* **Total Features Used (39):** 38 numerical features + 1 categorical feature (`azimuth`).

#### Feature Breakdown:

##### 1. Base Measurements & Telemetry (9 features)
1. `ac_power_w`: Instantaneous AC active power output at time $t$ in Watts ($\text{W}$). Valid input because the target is 1 hour in the future.
2. `irradiance_wm2`: Solar irradiance in $\text{W/m}^2$.
3. `temp_c`: Ambient temperature in $^\circ\text{C}$.
4. `wind_speed_ms`: Wind speed in $\text{m/s}$.
5. `rh_pct`: Relative humidity in $\%$.
6. `capacity_kwp`: DC array peak rated capacity in $\text{kWp}$.
7. `tilt_deg`: Panel installation tilt angle in degrees.
8. `altitude_m`: Building altitude above sea level in meters.
9. `n_valid_5min`: Count of valid 5-min intervals recorded in the 15-min aggregation window.

##### 2. Cyclical Temporal Encodings (4 features)
10. `hour_sin`: Sine encoding of fractional hour ($0\text{--}24$).
11. `hour_cos`: Cosine encoding of fractional hour ($0\text{--}24$).
12. `day_sin`: Sine encoding of continuous day of year ($1\text{--}366$).
13. `day_cos`: Cosine encoding of continuous day of year ($1\text{--}366$).

##### 3. Station Categorical Orientation (1 feature)
14. `azimuth`: Compass orientation heading of the rooftop installation (e.g., `south`, `east_west`, `mixed`, `other`), one-hot encoded by the pipeline preprocessor.

##### 4. B1 Core Physical & Domain Features (7 features)
15. `specific_yield`: Standardized power output: $Y_{\text{spec}} = (P_{\text{AC}} / 1000) / \text{capacity\_kwp}$ ($\text{kW/kWp}$).
16. `performance_ratio`: Instantaneous efficiency metric comparing measured AC power against theoretical derated clear-sky power ($\text{PR} \in [0.0, 2.5]$).
17. `peer_dev_campus`: Normalized deviation of inverter PR from the instantaneous campus-wide median PR. Cancels common weather transients.
18. `peer_dev_station`: Normalized deviation of inverter PR from the instantaneous median of inverters on the *exact same building rooftop*.
19. `self_dev_30d`: Deviation of current PR from the inverter's own rolling 30-day historical baseline under identical solar irradiance conditions.
20. `thermal_stress`: Sunlight-heat interaction proxy: $(G / 1000) \times T_{\text{ambient}}$.
21. `temp_delta_stc`: Departure of ambient temperature from Standard Test Conditions: $T_{\text{ambient}} - 25.0^\circ\text{C}$.

##### 5. A2 Autoregressive Lagged Signals (6 features)
22. `performance_ratio_lag_15m`: Performance ratio lagged by 1 step ($t - 15\text{ min}$).
23. `performance_ratio_lag_1h`: Performance ratio lagged by 4 steps ($t - 1\text{ hour}$).
24. `performance_ratio_lag_1d`: Performance ratio lagged by 1 day ($t - 24\text{ hours}$).
25. `peer_dev_campus_lag_15m`: Campus peer deviation lagged by 1 step ($t - 15\text{ min}$).
26. `peer_dev_campus_lag_1h`: Campus peer deviation lagged by 4 steps ($t - 1\text{ hour}$).
27. `peer_dev_campus_lag_1d`: Campus peer deviation lagged by 1 day ($t - 24\text{ hours}$).

##### 6. A2 Past-Only Rolling Window Statistics (12 features)
*All rolling windows are strictly closed on the left (`closed="left"`), excluding the current timestamp $t$ to prevent data leakage:*
28. `performance_ratio_roll_1h_mean`: Trailing 1-hour rolling mean of PR.
29. `performance_ratio_roll_1h_std`: Trailing 1-hour rolling standard deviation of PR (captures short-term output instability).
30. `performance_ratio_roll_4h_mean`: Trailing 4-hour rolling mean of PR.
31. `performance_ratio_roll_4h_std`: Trailing 4-hour rolling standard deviation of PR.
32. `performance_ratio_roll_1d_mean`: Trailing 1-day rolling mean of PR.
33. `performance_ratio_roll_1d_std`: Trailing 1-day rolling standard deviation of PR.
34. `peer_dev_campus_roll_1h_mean`: Trailing 1-hour rolling mean of campus peer deviation.
35. `peer_dev_campus_roll_1h_std`: Trailing 1-hour rolling standard deviation of campus peer deviation.
36. `peer_dev_campus_roll_4h_mean`: Trailing 4-hour rolling mean of campus peer deviation.
37. `peer_dev_campus_roll_4h_std`: Trailing 4-hour rolling standard deviation of campus peer deviation.
38. `peer_dev_campus_roll_1d_mean`: Trailing 1-day rolling mean of campus peer deviation.
39. `peer_dev_campus_roll_1d_std`: Trailing 1-day rolling standard deviation of campus peer deviation.

> **Note on Feature Pruning Comparison:** In [`src/models/rf.py`](file:///home/jemil/InvertML/src/models/rf.py#L186-L227), a 37-feature pruned subset (removing `irradiance_wm2` and `temp_delta_stc` due to $|r| > 0.85$ collinearity) was tested against all 39 features. Performance on validation was virtually tied (RMSE 10.055 vs 10.064; Macro-F1 0.6386 vs 0.6372), so the full 39-feature set was retained for final modeling.

---

### 2.4 XGBoost (Track T3 — Partner B1)

XGBoost is the primary production candidate model from the gradient boosting tree paradigm, utilizing histogram-based splits (`tree_method="hist"`) and exact split-gain regularized pruning.

* **Target:** $\text{Health}(t + 1\text{ h})$ (Continuous $0\text{--}100$) and $\text{Status}(t + 1\text{ h})$ (3-class softmax: `Healthy`, `Warning`, `Faulty`).
* **Design Philosophy:** XGBoost operates on the **Stage 7 Collinearity-Pruned Feature Set** plus `health_now`. Highly collinear features were systematically removed on the training set to prevent tree split variance and preserve clean TreeSHAP attribution.
* **Implementation:** [`src/models/xgb.py`](file:///home/jemil/InvertML/src/models/xgb.py), [`src/features.py`](file:///home/jemil/InvertML/src/features.py)
* **Total Features Used (14):**

| # | Feature Name | Origin / Group | Mathematical Formulation | Physical Explanation |
|:---:|:---|:---|:---|:---|
| 1 | `health_now` | Target History | $\text{Health}(t) = \min(100, \max(0, \text{PR} \times 100))$ | Instantaneous health score at time $t$. Baseline signal for 1-hour ahead forecasting. |
| 2 | `hour_sin` | B2 Cyclical Time | $\sin(2\pi \cdot t_{\text{hour}} / 24)$ | Continuous diurnal solar angle phase. |
| 3 | `day_sin` | B2 Cyclical Time | $\sin(2\pi \cdot t_{\text{day}} / 365.25)$ | Continuous annual seasonal phase. |
| 4 | `day_cos` | B2 Cyclical Time | $\cos(2\pi \cdot t_{\text{day}} / 365.25)$ | Continuous annual seasonal solar declination amplitude. |
| 5 | `capacity_kwp` | B2 Metadata | Rated DC array capacity ($\text{kWp}$) | Array peak capacity; normalizes scaling differences across rooftops. |
| 6 | `tilt_deg` | B2 Metadata | Cleaned rooftop inclination angle ($^\circ$) | Affects solar incidence angle across summer/winter. |
| 7 | `altitude_m` | B2 Metadata | Elevation above sea level ($\text{m}$) | Roof height; correlates with wind exposure and microclimate variations. |
| 8 | `orient_south` | B2 Metadata | $\mathbb{I}(135^\circ \le \text{azimuth} \le 225^\circ)$ | Binary flag for south-facing arrays experiencing peak midday irradiance. |
| 9 | `orient_east_west`| B2 Metadata | $\mathbb{I}(\text{azimuth} \in \text{East or West})$ | Binary flag for arrays with asymmetric morning/afternoon generation profiles. |
| 10 | `orient_mixed` | B2 Metadata | $\mathbb{I}(\text{azimuth contains 'mixed'})$ | Binary flag for complex, multi-string, or flat rooftop installations. |
| 11 | `peer_dev_campus`| B1 Peer Group | $\frac{\text{PR}_i(t) - \text{Median}_{\text{campus}}(\text{PR}(t))}{\text{Median}_{\text{campus}}(\text{PR}(t)) + 0.05}$ | **Core Signal:** Instantaneous campus peer deviation. Common cloud shadows cancel out, isolating true inverter hardware faults. |
| 12 | `peer_dev_station`| B1 Peer Group| $\frac{\text{PR}_i(t) - \text{Median}_{\text{station}}(\text{PR}(t))}{\text{Median}_{\text{station}}(\text{PR}(t)) + 0.05}$ | **Local Signal:** Instantaneous peer deviation among inverters mounted on the *same physical building rooftop*. |
| 13 | `self_dev_30d` | B1 Relative Dev | $\frac{\text{PR}_i(t) - \text{Baseline}_{30\text{d}}(G_{\text{bin}})}{\text{Baseline}_{30\text{d}}(G_{\text{bin}}) + 0.05}$ | **Degradation Signal:** Inverter PR compared to its own 30-day historical trailing median under the same irradiance tier. Detects progressive soiling/string failure. |
| 14 | `thermal_stress`| B1 Interaction | $(G / 1000.0) \times T_{\text{ambient}}$ | **Thermal Signal:** Sunlight-heat compounding stress proxy. Drives thermal throttling and inverter heat dissipation limits. |

#### Stage 7 Pruning Decisions (Why 14 features instead of 18 candidates):
Candidate features were evaluated for collinearity on the training split ($\le 2022\text{-}06\text{-}30$):
* `specific_yield` was pruned ($|r| = 0.92$ with `hour_cos`; redundant with normalized `peer_dev_campus`).
* `hour_cos` was pruned ($|r| = 0.96$ with `thermal_stress`; thermal stress already captures diurnal heating and solar zenith).
* `temp_delta_stc` was pruned ($|r| = 0.94$ with diurnal heating; redundant with ambient temperature embedded in `thermal_stress`).
* `performance_ratio` was pruned ($|r| = 1.00$ with `peer_dev_campus` on the baseline sky; peer deviation provides noise-free relative signal).

---

### 2.5 Multi-Layer Perceptron (Track T4 — Partner B2)

The Multi-Layer Perceptron (MLP) tests whether feedforward artificial neural networks can discover non-linear combinations of normalized tabular telemetry that tree models might miss.

* **Target:** $\text{Health}(t + 1\text{ h})$ (continuous $0\text{--}100$) and $\text{Status}(t + 1\text{ h})$ (3-class discrete status).
* **Architecture:** 2 hidden layers with $(100, 50)$ neurons, ReLU activation, Adam optimizer, initial learning rate $\eta = 0.005$, batch size $256$, $L_2$ penalty $\alpha = 0.0001$, early stopping enabled.
* **Pipeline:** Training-fitted mean imputation $\to$ [`StandardScaler()`](file:///home/jemil/InvertML/src/models/mlp.py#L82) $\to$ `MLPRegressor` / `MLPClassifier`. Rigid z-score standardization is mandatory for neural network gradient stability.
* **Implementation:** [`src/models/mlp.py`](file:///home/jemil/InvertML/src/models/mlp.py), [`src/models/mlp_balancing.py`](file:///home/jemil/InvertML/src/models/mlp_balancing.py)
* **Total Features Used (14):**

| # | Feature Name | Category | Brief Explanation |
|:---:|:---|:---|:---|
| 1 | `health_now` | Target History | Current instantaneous health score at time $t$ ($0\text{--}100$). Strong autoregressive anchor. |
| 2 | `ac_power_w` | Telemetry | Instantaneous AC generation at time $t$ in Watts ($\text{W}$). Causal predictor of $t + 1\text{ h}$ state. |
| 3 | `irradiance_wm2` | Weather | Solar irradiance ($\text{W/m}^2$). Primary physical energy input. |
| 4 | `temp_c` | Weather | Ambient air temperature ($^\circ\text{C}$). Drives thermal silicon losses. |
| 5 | `hour_sin` | Cyclical Time | $\sin(2\pi \cdot t_{\text{hour}} / 24)$. Continuous circular projection of diurnal time. |
| 6 | `hour_cos` | Cyclical Time | $\cos(2\pi \cdot t_{\text{hour}} / 24)$. Continuous diurnal elevation coordinate. |
| 7 | `day_sin` | Cyclical Time | $\sin(2\pi \cdot t_{\text{day}} / 365.25)$. Continuous annual seasonal cycle coordinate. |
| 8 | `day_cos` | Cyclical Time | $\cos(2\pi \cdot t_{\text{day}} / 365.25)$. Continuous annual solar declination coordinate. |
| 9 | `capacity_kwp` | Station Metadata | Nameplate DC array peak power capacity ($\text{kWp}$). |
| 10 | `tilt_deg` | Station Metadata | Installation slope angle in degrees. |
| 11 | `altitude_m` | Station Metadata | Rooftop installation elevation above sea level ($\text{m}$). |
| 12 | `orient_south` | Station Metadata | Binary indicator for south-facing arrays ($135^\circ\text{--}225^\circ$). |
| 13 | `orient_east_west` | Station Metadata | Binary indicator for east- or west-facing arrays. |
| 14 | `orient_mixed` | Station Metadata | Binary indicator for multi-string or flat rooftop orientations. |

> **Note on Training Protocol:** Because training deep networks on 1.5 million rows exceeds laptop CPU budget, MLP training is executed on a seeded time-ordered sample of 60,000 training rows across all inverters, while validation and test sets are evaluated in full.

---

### 2.6 Isolation Forest (Track T5 — Unsupervised Anomaly Detection)

Isolation Forest provides an unsupervised safety layer that detects anomalous operational patterns without requiring pre-engineered health labels. Two distinct implementations address two complementary validation objectives:

#### A. HKUST Inverter Telemetry Benchmark (Pair A: A1 & A2)
* **Goal:** Detect synthetic operational faults (power dips, thermal anomalies) injected into healthy HKUST inverter telemetry ($\text{Health} \ge 90$).
* **Implementation:** [`src/models/iforest_a1.py`](file:///home/jemil/InvertML/src/models/iforest_a1.py), [`src/models/iforest_a2.py`](file:///home/jemil/InvertML/src/models/iforest_a2.py)
* **Features Evaluated:**

| Configuration | # Features | Features Used | Explanation |
|---|:---:|---|---|
| **A2 Baseline Inputs** | **3** | `irradiance_wm2`, `temp_c`, `specific_yield` | Evaluates whether current specific generation $P_{\text{AC}} / (\text{capacity} \times 1000)$ is anomalous given current sunlight and temperature. |
| **A1 Extended Inputs (Selected Winner)** | **4** | `irradiance_wm2`, `temp_c`, `specific_yield`, `peer_dev_campus` | Adding `peer_dev_campus` normalizes against the instantaneous campus median, canceling cloud transients and reducing false alarm rate from $5.3\%$ down to **$0.6\%$** on the test split. |

#### B. GPVS-Faults Benchmark on Real Laboratory Physical Faults (Pair B: B1 & B2)
* **Goal:** Validate that unsupervised Isolation Forest detects **real, physically induced laboratory faults** (PV array short/open circuit, inverter bridge degradation, MPPT failure, grid voltage sags).
* **Dataset:** GPVS-Faults benchmark (Bakdi et al., Mendeley Data, DOI 10.17632/n76t439f65.1). High-frequency electrical data aggregated into $0.1$-second windows ($5$ grid cycles at $50\text{ Hz}$, $\sim 1,000$ raw samples per window).
* **Implementation:** [`src/models/iforest.py`](file:///home/jemil/InvertML/src/models/iforest.py), [`src/evaluate.py`](file:///home/jemil/InvertML/src/evaluate.py#L260-L330)
* **Total Features Used (18 Window Statistics):**

| # | Feature Name | Signal Type | Aggregation Metric | Physical Significance |
|:---:|:---|:---|:---|:---|
| 1 | `Ipv_mean` | DC Array Current | Window Mean | Identifies string disconnection or array shading drops. |
| 2 | `Ipv_std` | DC Array Current | Window Std Dev | Detects rapid MPPT current oscillations or arcing. |
| 3 | `Ipv_min` | DC Array Current | Window Minimum | Captures transient current dropouts. |
| 4 | `Ipv_max` | DC Array Current | Window Maximum | Detects DC surge spikes. |
| 5 | `Vpv_mean` | DC Array Voltage | Window Mean | Detects module open/short-circuit conditions. |
| 6 | `Vpv_std` | DC Array Voltage | Window Std Dev | Reflects voltage instability under variable load. |
| 7 | `Vpv_min` | DC Array Voltage | Window Minimum | Identifies severe voltage sags. |
| 8 | `Vpv_max` | DC Array Voltage | Window Maximum | Identifies overvoltage conditions. |
| 9 | `Vdc_mean` | DC-Link Bus Voltage | Window Mean | Critical inverter internal DC-link capacitor voltage level. |
| 10 | `Vdc_std` | DC-Link Bus Voltage | Window Std Dev | Detects DC-link ripple and capacitor aging. |
| 11 | `Vdc_min` | DC-Link Bus Voltage | Window Minimum | Catches bridge under-voltage dropouts. |
| 12 | `Vdc_max` | DC-Link Bus Voltage | Window Maximum | Catches regenerative overvoltage surges. |
| 13 | `ia_rms` | Phase A AC Current | Root Mean Square (RMS) | Phase A current loading and magnitude. |
| 14 | `ib_rms` | Phase B AC Current | Root Mean Square (RMS) | Phase B current loading and magnitude. |
| 15 | `ic_rms` | Phase C AC Current | Root Mean Square (RMS) | Phase C current loading and magnitude; imbalance across (ia, ib, ic) reveals bridge faults. |
| 16 | `va_rms` | Phase A AC Voltage | Root Mean Square (RMS) | Phase A grid coupling voltage magnitude. |
| 17 | `vb_rms` | Phase B AC Voltage | Root Mean Square (RMS) | Phase B grid coupling voltage magnitude. |
| 18 | `vc_rms` | Phase C AC Voltage | Root Mean Square (RMS) | Phase C grid coupling voltage magnitude; voltage dips indicate grid sags. |

---

## 3. Comprehensive Feature Comparison Matrix

The table below maps every engineered feature against all models in the project, demonstrating feature reuse, specialization, and design rationale across tracks.

| Feature Name | Category | Baselines (A1) | Ridge / Logistic (A1) | Random Forest (A2) | XGBoost (B1) | MLP (B2) | IForest HKUST (Pair A) | IForest GPVS (Pair B) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `health_now` | Target History | ✓ | ✓ | — | ✓ | ✓ | — | — |
| `ac_power_w` | Telemetry | — | — | ✓ | — | ✓ | — | — |
| `n_valid_5min` | Telemetry Quality | — | ✓ | ✓ | — | — | — | — |
| `irradiance_wm2` | Weather | — | ✓ | ✓ | — | ✓ | ✓ | — |
| `temp_c` | Weather | — | ✓ | ✓ | — | ✓ | ✓ | — |
| `rh_pct` | Weather | — | ✓ | ✓ | — | — | — | — |
| `wind_speed_ms` | Weather | — | ✓ | ✓ | — | — | — | — |
| `hour_sin` | Cyclical Time | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `hour_cos` | Cyclical Time | — | ✓ | ✓ | — *(pruned)* | ✓ | — | — |
| `day_sin` | Cyclical Time | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `day_cos` | Cyclical Time | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `capacity_kwp` | Station Metadata | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `tilt_deg` | Station Metadata | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `altitude_m` | Station Metadata | — | ✓ | ✓ | ✓ | ✓ | — | — |
| `orient_south` | Station Metadata | — | ✓ | — | ✓ | ✓ | — | — |
| `orient_east_west` | Station Metadata | — | ✓ | — | ✓ | ✓ | — | — |
| `orient_mixed` | Station Metadata | — | ✓ | — | ✓ | ✓ | — | — |
| `azimuth` *(raw)* | Station Metadata | — | — | ✓ *(one-hot)* | — | — | — | — |
| `is_offline` | Outage Status | ✓ *(rule)* | ✓ | — | — | — | — | — |
| `minutes_offline_24h` | Outage History | — | ✓ | — | — | — | — | — |
| `specific_yield` | B1 Core Physical | — | — | ✓ | — *(pruned)* | — | ✓ | — |
| `performance_ratio` | B1 Core Physical | — | — | ✓ | — *(pruned)* | — | — | — |
| `peer_dev_campus` | B1 Core Physical | ✓ *(rule)* | — | ✓ | ✓ | — | ✓ *(ext)* | — |
| `peer_dev_station` | B1 Core Physical | — | — | ✓ | ✓ | — | — | — |
| `self_dev_30d` | B1 Core Physical | — | — | ✓ | ✓ | — | — | — |
| `thermal_stress` | B1 Core Physical | — | — | ✓ | ✓ | — | — | — |
| `temp_delta_stc` | B1 Core Physical | — | — | ✓ | — *(pruned)* | — | — | — |
| `performance_ratio_lag_*` (3) | A2 History Lags | — | — | ✓ | — | — | — | — |
| `peer_dev_campus_lag_*` (3) | A2 History Lags | — | — | ✓ | — | — | — | — |
| `performance_ratio_roll_*` (6) | A2 Rolling Stats | — | — | ✓ | — | — | — | — |
| `peer_dev_campus_roll_*` (6) | A2 Rolling Stats | — | — | ✓ | — | — | — | — |
| `Ipv_*`, `Vpv_*`, `Vdc_*` (12) | High-Freq DC Signals | — | — | — | — | — | — | ✓ |
| `ia/b/c_rms`, `va/b/c_rms` (6) | High-Freq AC Signals | — | — | — | — | — | — | ✓ |
| **Total Features Count** | | **0 – 2** | **18** | **39** | **14** | **14** | **3 or 4** | **18** |

---

## 4. Key Architectural & Anti-Leakage Rules

### 4.1 Target Formulation and Forward Lead Alignment (Option A)
To prevent circular self-prediction (target leakage), all supervised forecasting models predict the health state **one hour ahead in the future**:
$$\text{Target} = \text{Health}(t + 1\text{ h}) = \text{Health}(t + 4 \times 15\text{ min})$$
* **Why current power $P_{\text{AC}}(t)$ and current health $\text{Health}(t)$ are valid features:** Because all inputs are observed strictly at or before time $t$, supplying current power does not leak the target at $t + 1\text{ h}$. The model must learn temporal transition and degradation dynamics over the intervening 60 minutes.
* **Exact Timestamp Alignment:** Targets are joined strictly on `(inverter_id, time + 1 hour)` within the same calendar day. The last four 15-minute intervals of a day never take targets from the following morning.

### 4.2 Training-Only Imputation, Scaling, and Pruning
* Every `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` is fitted **strictly on the chronological training split** ($\le 2022\text{-}06\text{-}30$).
* Validation ($2022\text{ H2}$) and test splits ($2023$) are transformed using parameters learned on training data alone.
* Held-out stations (`unseen_stations`) are completely excluded from training and validation partitions.
* Collinearity pruning ($|r| > 0.85$) was computed exclusively on training records, preventing optimistic test variance reduction.

---

## 5. Document Revision History

* **2026-10-21:** Final project-wide re-run incorporating the $0.570$ post-sensor-change irradiance correction, undefined health scores for missing power, and resolution of B1 XGBoost target leakage.
* **2026-10-23:** Consolidated master feature catalog created (`features_used.md`).

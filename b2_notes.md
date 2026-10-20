# Student B2 Comprehensive Technical Defense & Viva Master Guide
**Role:** Evaluation Architect & Neural Network Specialist (Pair B)  
**Project:** Solar Inverter Health Monitoring & Anomaly Detection System  
**Codebase:** `InvertML`  
**Author / Committer:** `jp4812 <jemilpatel4812@gmail.com>`

---

## Table of Contents
1. [B2 Core Role, System Architecture & Team Boundaries](#1-b2-core-role-system-architecture--team-boundaries)
2. [Datasets, Physical Assumptions & Mathematical Target Engineering](#2-datasets-physical-assumptions--mathematical-target-engineering)
3. [Stage 11: Master Validation Framework (`src/evaluate.py`)](#3-stage-11-master-validation-framework-srcevaluatepy)
4. [Stage 7: Feature Engineering (`src/features.py`)](#4-stage-7-feature-engineering-srcfeaturespy)
5. [Stage 9: Supervised Modeling — Multi-Layer Perceptron (`src/models/mlp.py`)](#5-stage-9-supervised-modeling--multi-layer-perceptron-srcmodelsmlppy)
6. [Stage 10: Hyperparameter Tuning with TimeSeriesSplit](#6-stage-10-hyperparameter-tuning-with-timeseriessplit)
7. [Stage 11 / Task 7: Model Explainability & Error Diagnostics](#7-stage-11--task-7-model-explainability--error-diagnostics)
8. [Task 5: Unsupervised Anomaly Detection on GPVS-Faults (`src/models/iforest.py`)](#8-task-5-unsupervised-anomaly-detection-on-gpvs-faults-srcmodelsiforestpy)
9. [The Production Inference Pipeline: 2-Tier Cascade](#9-the-production-inference-pipeline-2-tier-cascade)
10. [Master Viva & Oral Exam Defense Drill (12 High-Yield Questions)](#10-master-viva--oral-exam-defense-drill-12-high-yield-questions)

---

## 1. B2 Core Role, System Architecture & Team Boundaries

### A. Team Division & Responsibilities
The 4-student engineering team is structured into two functional pairs:
* **Pair A (Students A1, A2):** Data Ingestion & Preprocessing Foundation (raw telemetry parsing, Brick schema extraction, timestamp alignment, gap imputation, SMOTE vs. class-weight balancing).
* **Pair B (Students B1, B2):** Feature Engineering, Supervised Modeling & Evaluation Architecture.
  * **B1 (Feature Specialist & Tree Lead):** Owns Track T3 (**XGBoost**) and co-owns feature target definitions.
  * **B2 (Evaluation Architect & Neural Network Lead — YOU):**
    * **Pipeline Leadership (Stages 11 & 12):** Constructs the central evaluation framework ([`src/evaluate.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/evaluate.py)), establishes the chronological and spatial unseen-station splits, manages the master leaderboard ([`results/results.csv`](file:///Users/akshayjoshi/Desktop/InvertML/results/results.csv)), and writes the operational impact and limitation analyses.
    * **Modeling Ownership (Track T4):** Builds, scales, tunes, and explains the **Multi-Layer Perceptron (MLP)** for continuous Health Score regression and 3-class status classification.
    * **Anomaly Benchmark (Track T5):** Validates unsupervised **Isolation Forest** on the **GPVS-Faults** laboratory benchmark.

### B. Team Boundary Matrix
| Pipeline Stage | Lead Student | B2 Direct Contribution / Deliverable |
| :--- | :---: | :--- |
| **Stage 6: Exploratory Data Analysis** | All (Individual) | `notebooks/01_eda_B2.ipynb` (3-year seasonality decomposition & class imbalance analysis across seasons). |
| **Stage 7: Feature Engineering** | Pair B | Cyclical time features ($\sin/\cos$) and station metadata encoders in [`src/features.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/features.py). |
| **Stage 9: Supervised Modeling** | B2 (Track T4) | `mlp_regressor` and `mlp_classifier` in [`src/models/mlp.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/models/mlp.py). |
| **Stage 10: Hyperparameter Tuning** | B2 (Track T4) | Exhaustive `GridSearchCV` using `TimeSeriesSplit` across 36 permutations (108 fits). |
| **Stage 11: Validation Engine** | **B2 (Lead)** | Core module [`src/evaluate.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/evaluate.py) imported by all 4 teammates. |
| **Stage 11 / Task 7: Explainability** | B2 (Track T4) | Permutation importance, Partial Dependence Plots, and top 3 failure case diagnostics. |
| **Track T5: Anomaly Detection** | Pair B (with B1) | `IsolationForest` pipeline validated on real electrical physical faults in [`src/models/iforest.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/models/iforest.py). |
| **Stage 12: Deployment & Viva** | **B2 (Lead)** | Operational impact, honest limitations, and defense cheat sheet. |

---

## 2. Datasets, Physical Assumptions & Mathematical Target Engineering

### A. The Two Datasets & Their Rationale

```
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│       PRIMARY: HKUST Rooftop Dataset         │    │       SECONDARY: GPVS-Faults Benchmark       │
├──────────────────────────────────────────────┤    ├──────────────────────────────────────────────┤
│ • 37 campus rooftop inverter installations   │    │ • Physical microgrid laboratory experiment   │
│ • 1.58 million 15-minute rows (2021–2023)    │    │ • Calibrated electrical instrumentation      │
│ • Rich weather context (GHI, Temp, Wind, RH) │    │ • Ground-truth hardware failures (F0 to F7)  │
│ • Labels: Mathematically rule-engineered     │    │ • Labels: Real physical hardware breakdowns  │
│ • Used for: Primary training & evaluation    │    │ • Used for: Real-world reality check         │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

#### Why we cannot rely solely on the HKUST Dataset:
In commercial and academic solar installations, technicians rarely record timestamped, machine-readable failure logs for every minor wire corrosion or diode degradation. Consequently, **the status labels (`Healthy`, `Warning`, `Faulty`) in HKUST are mathematically engineered** using performance ratio equations.  
Testing solely on HKUST invites a fatal viva criticism:  
> *"Did your neural network detect actual physical breakdowns, or did it just reverse-engineer your own mathematical formulas?"*

**The Justification for GPVS-Faults:**  
GPVS-Faults provides true physical hardware faults ($F_1$: open circuits, $F_2$: short circuits, $F_3$: line-to-line faults, $F_5$: partial shading). Validating our anomaly detector on GPVS proves that our algorithms catch physical hardware failures in real life.

---

### B. Physical Derivations & Mathematical Targets

#### 1. The Physics of Solar Inverter Conversion
Solar photovoltaic (PV) modules convert sunlight into Direct Current (DC) electricity. The solar inverter converts DC into 3-phase Alternating Current (AC) synchronized with the building grid.
* Under peak clear-sky conditions, the sun delivers a standardized solar irradiance of:
  $$G_{\text{STC}} = 1000 \text{ W/m}^2 \quad (\text{Standard Test Conditions})$$
* A rooftop array rated at $P_{\text{rated}} = C \text{ kWp}$ (where $C$ is `capacity_kwp`) produces:
  $$P_{\text{DC, theoretical}} = \left(\frac{G}{1000 \text{ W/m}^2}\right) \times (C \times 1000 \text{ W})$$

#### 2. Derivation of the $0.78$ Balance-of-System (BOS) Efficiency Factor
In real rooftop installations, theoretical DC power is never fully realized at the AC output due to continuous physical losses:
1. **Inverter DC-to-AC Inversion Losses:** Efficiency $\approx 96.0\% - 97.5\%$
2. **Thermal Derating:** Silicon panels lose $\sim 0.4\%$ efficiency per $^\circ\text{C}$ above $25^\circ\text{C}$ (summer rooftop temperatures reach $50^\circ\text{C} - 60^\circ\text{C} \implies \sim 10\% - 14\%$ thermal loss).
3. **Soiling & Dust Accumulation:** Light attenuation due to urban particulates $\approx 3\% - 5\%$
4. **Wiring & Ohmic Resistance:** Cable dissipation $\approx 2\%$
5. **MPPT Tracking Efficiency:** Sub-optimal operating point tracking $\approx 1\% - 2\%$

Multiplying these independent efficiencies:
$$\eta_{\text{BOS}} = 0.97 \times 0.88 \times 0.96 \times 0.98 \times 0.98 \approx \mathbf{0.78}$$

Therefore, the **Expected AC Power** is mathematically defined as:
$$\text{Expected Power (W)} = \left(\frac{\text{irradiance\_wm2}}{1000}\right) \times (\text{capacity\_kwp} \times 1000) \times 0.78$$

#### 3. Health Score Formula (Target for `mlp_regressor`)
The continuous target is the **Performance Ratio (PR)** normalized to a 100-point scale:
$$\text{Health Score} = \min\left(100.0, \, \frac{\text{Actual AC Power}}{\text{Expected Power}} \times 100.0\right)$$

```python
# Code Snippet from prepare_mlp_data in src/models/mlp.py
expected_power = (irradiance / 1000.0) * (capacity * 1000.0) * 0.78
expected_power = np.maximum(expected_power, 100.0)  # Numerical stability floor
performance_ratio = ac_power / expected_power
health_score = np.clip(performance_ratio * 100.0, 0.0, 100.0)
```

#### 4. Operational Status Thresholds (Target for `mlp_classifier`)
To triage operational maintenance tickets, continuous health is discretized into 3 operational regimes:
* **$\text{Health Score} \ge 90.0 \implies \textbf{Healthy}$ (Class 0):** Inverter operates within normal manufacturer tolerances ($\ge 90\%$ expected yield).
* **$70.0 \le \text{Health Score} < 90.0 \implies \textbf{Warning}$ (Class 1):** Sub-optimal operation (moderate dust soiling, partial micro-shading, minor string degradation). Non-critical maintenance ticket generated.
* **$\text{Health Score} < 70.0 \implies \textbf{Faulty}$ (Class 2):** Severe operational failure (inverter tripping, string disconnection, bypass diode short-circuit). Immediate technician dispatch required.

#### 5. Night-Time Standby Assumption
At night ($\text{irradiance} < 10\text{ W/m}^2$), solar arrays produce zero generation. Inverters enter standby mode, consuming a few Watts from the grid to power internal telemetry radios.  
**Justification:** Dividing standby power by zero expected solar power produces infinite or undefined ratios. Therefore, samples where `irradiance < 10 W/m²` represent non-operational nighttime standby and are excluded or gated to prevent catastrophic distortion of health metrics.

---

## 3. Stage 11: Master Validation Framework (`src/evaluate.py`)

As the **Evaluation Architect**, B2 wrote [`src/evaluate.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/evaluate.py) to guarantee that all 4 team members test their models under strictly identical, leakage-free conditions.

---

### A. The Chronological Split Setup

```
01-Jan-2021                 30-Jun-2022           31-Dec-2022                 31-Dec-2023
├────────────────────────────────┼─────────────────────┼───────────────────────────┤
│        TRAINING SET            │   VALIDATION SET    │         TEST SET          │
│          18 Months             │      6 Months       │     12 Months (Full Year) │
│       523,908 Samples          │   326,621 Samples   │      738,372 Samples      │
└────────────────────────────────┴─────────────────────┴───────────────────────────┘
```

#### Why standard K-Fold shuffle cross-validation is STRICTLY FORBIDDEN:
1. **Temporal Autocorrelation:** Solar irradiance and ambient temperatures change slowly over 15-minute intervals. If you shuffle data randomly, row $t$ might be in the training set while row $t+1$ (15 minutes later) is in the test set. Because the weather at 12:15 PM is almost identical to 12:00 PM, the model achieves near-zero error simply by memorizing adjacent weather points.
2. **Concept Drift & Degradation:** Solar panels physically degrade at $0.5\% - 0.8\%$ per year, and inverter capacitors age over time. Shuffling leaks future degraded states into past baseline training data.
3. **The Chronological Rule:** The model must **only train on past data (2021 to mid-2022)**, tune hyperparameters on **subsequent data (late 2022)**, and evaluate on a **completely unseen future year (2023)** containing all four distinct meteorological seasons.

---

### B. The 7 "Unseen Station" Spatial Generalization Test

```
                     ALL 37 ROOFTOP STATIONS
┌───────────────────────────────────────┬─────────────────────────────────┐
│     30 Known Rooftops (Training)      │   7 Held-Out Unseen Rooftops    │
│  • SQ Block P excluded                │  • SQ Block P                   │
│  • Shaw Auditorium excluded           │  • Shaw Auditorium              │
│  • UG Halls 8 & 9 excluded            │  • UG Hall 8 & UG Hall 9        │
│  • Used for all model fitting         │  • Zone A2-2, Zone A2-3, SQ14   │
│                                       │  • TESTED ONLY IN 2023 (72,612) │
└───────────────────────────────────────┴─────────────────────────────────┘
```

#### What this proves to the examination committee:
* In a commercial university or corporate microgrid, new solar panels are constantly added to new campus buildings.
* If a model is trained on specific rooftops, it can easily overfit to the static physical obstructions of those specific roofs (e.g., memorizing that a particular air conditioning unit on Shaw Auditorium casts a shadow at 3:00 PM).
* By completely withholding 7 rooftop stations during training and testing on them in 2023, **we evaluate true spatial generalization**:  
  *"Does our neural network learn the fundamental physical relationship between irradiance, tilt, and capacity, or does it merely memorize campus rooftop maps?"*

---

### C. Standard Metrics & Mathematical Justification

#### 1. Regression Metrics
* **MAE (Mean Absolute Error):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
  * *Purpose:* Direct, linear interpretation in health score points (e.g., our tuned model is off by $\pm 2.8$ points on average).
* **RMSE (Root Mean Squared Error) — PRIMARY METRIC:**
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  * *Mathematical Justification:* The $L_2$ squared penalty disproportionately punishes large prediction mistakes. If an inverter has a true health score of 20 (broken) and the model predicts 90 (healthy), the error is 70 points, contributing $70^2 = 4,900$ to the sum! For electrical equipment monitoring, missing a major breakdown is catastrophic; RMSE reflects this penalty.
* **$R^2$ (Coefficient of Determination):**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  * Measures the proportion of variance in health score explained by the model relative to a naive mean baseline.

#### 2. Classification Metrics
* **Faulty Recall (The Single Most Critical Metric in the Project):**
  $$\text{Recall}_{\text{Faulty}} = \frac{\text{True Positives (Faults caught)}}{\text{True Positives} + \text{False Negatives (Missed faults)}} = \frac{TP}{TP + FN}$$
  * *Domain Justification:* Missing an active inverter failure ($FN$) allows thousands of kilowatt-hours to be lost and risks electrical fires. A false alarm ($FP$) merely causes a technician to check a dashboard. Therefore, **Recall on the Faulty class is prioritized over overall Accuracy**.
* **Macro-F1 Score:**
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{Healthy}} + \text{F1}_{\text{Warning}} + \text{F1}_{\text{Faulty}}}{3}$$
  * *Mathematical Justification:* Because Healthy samples account for $\sim 85\%$ of all data, a naive model that predicts "Healthy" 100% of the time achieves an impressive 85% Accuracy while having a disastrous Macro-F1 of $\approx 0.31$. Macro-F1 weights every operational class equally, preventing dominant classes from masking catastrophic failure detection.
* **False Alarm Rate (False Positive Rate):**
  $$\text{FPR} = \frac{FP}{FP + TN}$$
  * Must be maintained below $3.0\%$ to prevent **technician alarm fatigue** (if an automated system cries wolf every day, maintenance crews ignore all alarms).

---

## 4. Stage 7: Feature Engineering (`src/features.py`)

B2 added two critical categories of features to [`src/features.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/features.py):

---

### A. Cyclical 15-Minute Time Features

#### The Problem with Raw Integer Hours:
In 24-hour solar telemetry, 23:45 at night and 00:00 (midnight) are separated by only 15 minutes physically. However, if represented as raw numbers ($23.75$ and $0.0$), a linear model or neural network interprets their numerical distance as:
$$|23.75 - 0.0| = 23.75 \quad (\text{Maximal numerical distance!})$$
This introduces an artificial mathematical discontinuity right at midnight.

#### The Mathematical Solution: Polar Projection onto the Unit Circle
We map time onto orthogonal trigonometric components:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24.0}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24.0}\right)$$
$$\text{day\_sin} = \sin\left(\frac{2\pi \times \text{day\_of\_year}}{365.25}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \times \text{day\_of\_year}}{365.25}\right)$$

```
                  hour_sin (12:00 PM = +1.0)
                             │
                             │
       (06:00 AM)            │            (06:00 PM)
  -1.0 ──────────────────────┼────────────────────── +1.0  hour_cos
                             │
                             │
                             │
                  hour_sin (00:00 AM = -1.0)
```

* **Euclidean Distance at Boundary:**
  The distance between 23:45 and 00:00 in $(\sin, \cos)$ space is:
  $$\Delta d = \sqrt{(\sin(2\pi \cdot \frac{23.75}{24}) - \sin(0))^2 + (\cos(2\pi \cdot \frac{23.75}{24}) - \cos(0))^2} \approx 0.065$$
  The mathematical discontinuity is eliminated; the network perceives smooth, continuous diurnal cycling.
* **Annual Cyclicality (`day_sin`, `day_cos`):** Captures seasonal solar zenith angles (high summer sun vs. low winter sun).

---

### B. Station Metadata Encoders
1. **`capacity_kwp`:** Raw system capacity in kW.
2. **`tilt_deg`:** Panel inclination angle relative to the horizontal roof plane.
3. **`altitude_m`:** Station elevation above sea level.
4. **Azimuth One-Hot Encoding:** Inverter azimuth angles determine sun-tracking orientation:
   * `orient_south`: Maximum midday solar harvest.
   * `orient_east_west`: Dual-peak harvest (morning east, afternoon west).
   * `orient_mixed`: Complex rooftop geometry.

---

## 5. Stage 9: Supervised Modeling — Multi-Layer Perceptron (`src/models/mlp.py`)

B2 owns **Track T4**, testing whether deep neural network representations outperform tree models (Random Forest, XGBoost) and linear baselines.

---

### A. The Neural Network Architecture
The network consists of a feedforward Multi-Layer Perceptron:
* **Input Layer ($d = 13$ features):** Scaled numerical features.
* **Hidden Layer 1:** 100 neurons, $\text{ReLU}(z) = \max(0, z)$ activation function.
* **Hidden Layer 2:** 50 neurons, $\text{ReLU}(z) = \max(0, z)$ activation function.
* **Output Layer:**
  * For `mlp_regressor`: 1 linear unit predicting continuous Health Score ($0 - 100$).
  * For `mlp_classifier`: 3 Softmax units predicting class probabilities:
    $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^3 e^{z_j}} \quad \text{for } c \in \{\text{Healthy}, \text{Warning}, \text{Faulty}\}$$

```
Input Layer (13)        Hidden Layer 1 (100)      Hidden Layer 2 (50)       Output Layer
[ ac_power_w     ] ───► ( Neuron 1: ReLU ) ────► ( Neuron 1: ReLU ) ────► [ Health Score (0-100) ]
[ irradiance_wm2 ] ───► ( Neuron 2: ReLU ) ────► ( Neuron 2: ReLU )         (for mlp_regressor)
[ temp_c         ] ───►        ...                     ...                      OR
[ hour_sin, cos  ] ───►        ...                     ...               [ P(Healthy) ]
[ day_sin, cos   ] ───►        ...                     ...               [ P(Warning) ]
[ capacity_kwp   ] ───► ( Neuron 100: ReLU) ───► ( Neuron 50: ReLU )     [ P(Faulty)  ]
[ tilt, azimuth  ]                                                       (for mlp_classifier)
```

---

### B. Why `StandardScaler` is REQUIRED for Neural Networks (Viva Proof!)

If an examiner asks:
> *"Why did your partner's Random Forest and XGBoost run fine without scaling, while your MLP completely required StandardScaler?"*

#### The Mathematical Proof:
1. **Tree Models (Random Forest, XGBoost):**  
   Decision trees evaluate splits using monotonic rank-order inequalities ($x_j \le \theta$). Multiplying a feature by $10,000$ does not change the ordering of values; split points merely rescale. Hence, trees are completely scale-invariant.
2. **Neural Networks (MLP):**  
   Neural networks compute weighted sums and update weights via **Stochastic Gradient Descent (SGD / Adam)**:
   $$z = \sum_{j=1}^d w_j x_j + b, \quad w_j \leftarrow w_j - \eta \frac{\partial \mathcal{L}}{\partial w_j}$$
   By the chain rule, the gradient with respect to weight $w_j$ is directly proportional to the magnitude of the input feature $x_j$:
   $$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial z} \cdot x_j$$
   * If `ac_power_w` is unscaled, $x_{\text{power}} \approx 50,000\text{ W}$.
   * Meanwhile, $x_{\text{hour\_sin}} \in [-1.0, 1.0]$.
   * The gradient for $w_{\text{power}}$ will be **$50,000\times$ larger** than the gradient for $w_{\text{hour\_sin}}$!
   * The loss surface becomes an extremely elongated, pathological ellipsoid. Gradient descent oscillates violently along the power axis while taking microscopic steps along the time axis, causing gradient explosion or permanent divergence.
   * `StandardScaler` standardizes all features to $\mu = 0, \sigma = 1$, creating spherical loss contours for rapid, stable gradient convergence.

#### The Anti-Leakage Pipeline Architecture:
```python
mlp_regressor = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42))
])
```
* The `Pipeline` guarantees that `scaler.fit_transform()` is executed **only on training data**.
* When predicting on validation, test, or unseen station splits, the scaler executes only `scaler.transform()` using the stored training $\mu$ and $\sigma$, strictly preventing data leakage.

---

### C. Architectural Comparison: Method 1 vs. Method 2

We conducted a formal experiment comparing two distinct architectural approaches for operational status classification:
* **Method 1 (Dedicated Softmax Classifier):** Train `mlp_classifier` directly using cross-entropy loss across the 3 classes.
* **Method 2 (Thresholded Regressor):** Train `mlp_regressor` to predict continuous health score, then apply static threshold rules ($\ge 90 \to \text{Healthy}$, $70-90 \to \text{Warning}$, $< 70 \to \text{Faulty}$).

#### Experimental Comparison Results:
| Method | Operational Status Architecture | Test Macro-F1 | Faulty Recall | Verdict |
| :---: | :--- | :---: | :---: | :---: |
| **Method 1** | **Dedicated Softmax MLPClassifier** | **0.96** | **98.00%** | **WINNER** |
| **Method 2** | Thresholded Output from MLPRegressor | 0.94 | 95.00% | Underperformed on border cases |

#### Deep Justification for Why Method 1 Won:
1. **Loss Function Objective:**
   * Method 2 optimizes **Mean Squared Error (MSE)**:
     $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum (y - \hat{y})^2$$
     MSE treats an error from 85 to 80 (both inside the Warning band) with the exact same penalty as an error from 72 to 67 (which crosses the threshold between Warning and Faulty). It has no concept of class boundaries.
   * Method 1 optimizes **Categorical Cross-Entropy**:
     $$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^3 y_c \log \hat{p}_c$$
     The cross-entropy gradient approaches infinity as the predicted probability for the true class approaches zero:
     $$\lim_{\hat{p}_c \to 0} \frac{\partial \mathcal{L}}{\partial z} \to -\infty$$
     This forces the neural network's decision hyperplane to adjust aggressively to separate the rare `Faulty` cluster from the `Warning` cluster, resulting in **$98\%$ Faulty Recall**!

---

## 6. Stage 10: Hyperparameter Tuning with TimeSeriesSplit

---

### A. The Smart Compute Strategy
* The full training set contains **523,908 rows**.
* Training a 2-layer MLP on 523k rows takes $\sim 37$ seconds per fit.
* Evaluating 36 hyperparameter combinations across 3 cross-validation folds equals **108 fits**.
* Running 108 fits on 523k rows would take:
  $$108 \times 37\text{s} = 3,996\text{s} \approx \mathbf{66.6\text{ minutes}}$$
* **The Smart Engineering Plan:** Extract a **time-ordered 15% sample (~78,000 rows)** representing all weather cycles for cross-validation search (takes only 6 minutes). Once the optimal hyperparameter architecture is identified, **refit the single winning configuration on 100% of the training data (523,908 rows)**.

---

### B. TimeSeriesSplit (Expanding Window Cross-Validation)

```
Split 1: [ Train: Month 1 to 6  ] ──► [ Test: Month 7 to 9  ]
Split 2: [ Train: Month 1 to 9  ] ────────────► [ Test: Month 10 to 12 ]
Split 3: [ Train: Month 1 to 12 ] ──────────────────────► [ Test: Month 13 to 15 ]
```

* Strict temporal progression: training window expands forward; validation window is always strictly in the future.

---

### C. Exhaustive GridSearchCV Across 36 Combinations

```python
param_grid = {
    "mlp__hidden_layer_sizes": [(32, 16), (64, 32), (100, 50)],
    "mlp__alpha": [0.0001, 0.001, 0.01],          # L2 regularization
    "mlp__learning_rate_init": [0.001, 0.005],    # Initial step size
    "mlp__batch_size": [256, 512],                # Mini-batch gradient update
}
```

#### Why we tuned these specific parameters:
1. **`hidden_layer_sizes`:** Tests network capacity. `(32, 16)` tests if a compact network prevents overfitting; `(100, 50)` tests if extra representational capacity captures complex rooftop shading non-linearities.
2. **`alpha` (L2 Weight Decay):** Adds penalty $\frac{1}{2} \alpha \|\mathbf{w}\|_2^2$ to the loss function. Discourages individual weights from growing excessively large, smoothing decision boundaries.
3. **`learning_rate_init`:** Controls Adam optimizer step size. Too high ($> 0.01$) causes unstable oscillation; too low ($< 0.0005$) causes premature convergence to sub-optimal local minima.
4. **`batch_size`:** Mini-batch size affects gradient stochasticity. Size 256 offers a regularizing gradient noise that helps escape saddle points faster than batch size 1024.

---

### D. Tuning Results & Impact Table

* **Winning Configuration:** `hidden_layer_sizes=(100, 50)`, `learning_rate_init=0.005`, `batch_size=256`, `alpha=0.0001`.

| Architecture | Validation RMSE | Test RMSE (2023) | Unseen Station RMSE | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Untuned `(64, 32)`** | 3.54 | 5.70 | 7.96 | 37.2s |
| **GridSearch Tuned `(100, 50)`** | **3.45** | **4.76** | **5.04** | 36.8s |

> **Viva Takeaway:** Hyperparameter tuning produced a **$16.5\%$ error reduction on the chronological test set (5.70 $\to$ 4.76)** and a massive **$36.7\%$ error reduction on the unseen station holdout test (7.96 $\to$ 5.04)**!

---

## 7. Stage 11 / Task 7: Model Explainability & Error Diagnostics

Neural networks are frequently dismissed as "black boxes." In Stage 11, B2 implemented model explainability techniques to prove that the neural network learned true solar physics.

---

### A. Permutation Feature Importance

#### Mathematical Procedure:
1. Measure baseline Test RMSE on untouched data: $\text{RMSE}_{\text{base}} = 4.76$.
2. For each feature $j \in \{1, \dots, 13\}$:
   * Randomly permute (shuffle) the values of feature column $j$ across the test samples while leaving all other 12 columns untouched. This completely destroys the relationship between feature $j$ and the target.
   * Compute new $\text{RMSE}_j$.
   * Calculate importance: $\Delta \text{RMSE}_j = \text{RMSE}_j - \text{RMSE}_{\text{base}}$.
3. Repeat 5 times ($n_{\text{repeats}} = 5$) to calculate mean and standard deviation.

#### Results & Physical Rankings:
```
ac_power_w       : ███████████████████████████████████████ 3.4349 +/- 0.0200
irradiance_wm2   : ██████████████████ 1.6415 +/- 0.0108
capacity_kwp     : ███████████ 0.9356 +/- 0.0073
hour_cos         : █ 0.0512 +/- 0.0015
orient_mixed     : ▏ 0.0312 +/- 0.0009
day_cos          : ▏ 0.0291 +/- 0.0012
temp_c           : ▏ 0.0102 +/- 0.0006
tilt_deg         : ▏ 0.0019 +/- 0.0001
```

* **Physical Insight:** `ac_power_w`, `irradiance_wm2`, and `capacity_kwp` drive $98\%$ of the model's predictive variance. This confirms that the network bases its predictions on fundamental energy conversion physics rather than spurious correlations with time or temperature.
* *Artifact:* Saved to [`reports/figures/b2_permutation_importance.png`](file:///Users/akshayjoshi/Desktop/InvertML/reports/figures/b2_permutation_importance.png).

---

### B. Partial Dependence Plots (PDP) & The Downward Slope Paradox

#### Mathematical Definition:
The partial dependence of the model prediction on feature subset $S$ (e.g., `irradiance_wm2`) is obtained by marginalizing over the distribution of all other complementary features $C$:

$$\hat{f}_S(\mathbf{x}_S) = \mathbb{E}_{\mathbf{x}_C}[\hat{f}(\mathbf{x}_S, \mathbf{x}_C)] = \frac{1}{N} \sum_{i=1}^N \hat{f}(\mathbf{x}_S, \mathbf{x}_C^{(i)})$$

#### The Question Professors WILL Ask:
> *"In your PDP graph below, why does the predicted Health Score curve DOWNWARD as irradiance increases from 50 to 700 W/m²? Shouldn't more sunlight mean better health?"*

```
     PDP Curve: Predicted Health Score vs. Irradiance
     92 | \
        |   \
     60 |     \
     40 |       \______________________
        +──────────────────────────────
        0       200     400     700 W/m²
```

#### The Master Viva Explanation:
* **The Mathematical Reason:**  
  By definition, Partial Dependence varies **one feature** (`irradiance_wm2`) across its entire domain while **holding all other input features fixed at their dataset values**, including `ac_power_w` (which is frozen at its dataset average of $\sim 8\text{ kW}$).
* **The Physical Ratio:**  
  Recall the Performance Ratio formula:
  $$\text{Health Score} \propto \frac{\text{Actual AC Power}}{\textbf{Irradiance} \times \textbf{Capacity}}$$
  **Irradiance is in the denominator!**
* **The Contrast:**
  * **At Low Irradiance ($50\text{ W/m}^2$):** Weak, dim overcast sunlight. If the inverter produces $8\text{ kW}$ under dim light, it is vastly exceeding physical expectations $\implies \textbf{Health Score} \approx \mathbf{92}$ (Exceptional).
  * **At High Irradiance ($700\text{ W/m}^2$):** Blazing, clear-sky midday sun. If the inverter is *still producing only that same $8\text{ kW}$*, it is severely underperforming for that volume of incoming solar energy $\implies \textbf{Health Score} \approx \mathbf{40}$ (Faulty breakdown!).
* **Conclusion:** The downward slope is **proof that the neural network learned the inverse mathematical law of Performance Ratios**.
* *Artifact:* Saved to [`reports/figures/b2_partial_dependence.png`](file:///Users/akshayjoshi/Desktop/InvertML/reports/figures/b2_partial_dependence.png).

---

### C. Top 3 Worst Failure Cases (Domain Error Diagnosis)

We isolated the 3 largest absolute residuals $|y_{\text{true}} - y_{\text{pred}}|$ across all 738,372 test samples:

```
Timestamp            Inverter ID        Actual  Pred   Residual  Irradiance  AC Power
2023-11-08 17:30:00  zone_d_inverter_2   95.00  10.27   84.73     1.72 W/m²   95.0 W
2023-10-26 17:45:00  zone_d_inverter_2  100.00  22.47   77.53     4.24 W/m²  157.3 W
2023-10-13 17:45:00  zone_d_inverter_2   95.19  18.37   76.82     5.96 W/m²  114.0 W
```

#### The Physical Autopsy:
1. **Identical Location & Time:** All 3 failures occurred on the same unit (`zone_d_inverter_2`) in late autumn at **17:30–17:45 (sunset)**.
2. **The Sensor Glitch:** Solar irradiance had plummeted to between $1.72\text{ W/m}^2$ and $5.96\text{ W/m}^2$. AC output was near zero ($95\text{ W} - 157\text{ W}$) on a $50\text{ kW}$ array.
3. **The Misclassification Mechanism:** During midday, an inverter producing only $100\text{ W}$ represents catastrophic electrical failure, so the model predicted an abysmal score of $10 - 22$ points. But at 5:45 PM in November, the sun has already set; producing $100\text{ W}$ is normal inverter sleep transition.
4. **Engineering Fix for Production Deployment:** Apply an **irradiance threshold mask** in the serving layer:
   ```python
   # Production Gating Rule
   if row["irradiance_wm2"] < 10.0:
       status = "Standby_Night"
       health_score = np.nan  # Do not calculate health score during twilight
   ```

---

## 8. Task 5: Unsupervised Anomaly Detection on GPVS-Faults (`src/models/iforest.py`)

---

### A. The Algorithm: `sklearn.ensemble.IsolationForest`

#### How Isolation Forest Isolates Anomalies:
* Builds an ensemble of $100$ randomized decision trees (iTrees).
* In each tree, it picks a feature at random and chooses a random split between $\min$ and $\max$.
* **Normal points** reside in dense multi-dimensional clusters. To isolate one normal point requires an average path length $E(h(x)) \ge 12 - 15$ cuts.
* **Electrical Anomalies** (e.g., short circuit where voltage collapses to $0\text{V}$) sit in sparse, empty feature space. They are isolated near the root of the tree in **$1 - 3$ cuts**!

#### The Anomaly Score Equation:
$$s(x, n) = 2^{ -\frac{E(h(x))}{c(n)} }$$
* If average path length $E(h(x)) \to 0 \implies s \to 1.0$ (Anomaly, label `-1`).
* If average path length $E(h(x)) \to c(n) \implies s \to 0.5$ (Normal, label `+1`).

#### Why `max_samples="auto"` (256 samples) Prevents Masking and Swamping:
* **Swamping:** In a large dataset, normal points near an anomaly can mistakenly be partitioned into the anomaly's leaf.
* **Masking:** When a cluster of anomalies exists (e.g. 10 consecutive short-circuit readings), a large tree sees them as a normal dense cluster! Taking a small random subsample of 256 points dilutes the anomaly cluster into single isolated outliers, making them easily detectable.

---

### B. Empirical Benchmark Results on GPVS Real Hardware Faults

```python
# Evaluated on 1,100 samples (1,000 Normal, 100 Physical Faults)
# Test Set: 275 samples (250 Normal, 25 Physical Faults)
```

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Precision** | **77.27%** | When the system raises an anomaly alarm, >3 out of 4 times it is a real physical hardware breakdown. |
| **Detection Rate (Recall)** | **68.00%** | The model caught 17 out of 25 real physical faults **without ever seeing a single training label**! |
| **F1-Score** | **0.7234** | Harmonic balance between precision and recall. |
| **False Alarm Rate (FPR)** | **2.00%** | Only 5 false alarms out of 250 normal points. Prevents maintenance crew alarm fatigue. |
| **Execution Time** | **0.30s** | Ultra-lightweight for edge IoT microcontrollers. |

#### Confusion Matrix:
$$\begin{bmatrix} TN = 245 & FP = 5 \\ FN = 8 & TP = 17 \end{bmatrix}$$

---

## 9. The Production Inference Pipeline: 2-Tier Cascade

In commercial SCADA and campus monitoring, we combine Task 5 and Task 3 into a **2-Tier Cascaded Architecture**:

```
                       User Uploads Telemetry (CSV / Live IoT Stream)
                                            │
                                            ▼
           ═══════════════════════════════════════════════════════════════
           TIER 1 (GATEKEEPER): Isolation Forest (Task 5)
           ═══════════════════════════════════════════════════════════════
           • Runs fast on raw electrical channels (Voltage, Current, Power)
           • Unsupervised: Works on ANY inverter brand or uploaded schema
           • Decides: "Is this electrical telemetry normal?"
                                            │
                           ┌────────────────┴────────────────┐
                           ▼                                 ▼
                     Score >= 0.0                      Score < 0.0
                    (Normal: +1)                      (Anomaly: -1)
                           │                                 │
                           ▼                                 ▼
                 [ 🟢 All Systems Normal ]     ═════════════════════════════════
                 (Saves 95% server compute)    TIER 2 (DIAGNOSIS): Supervised MLP
                                               ═════════════════════════════════
                                               • mlp_classifier: Faulty or Warning?
                                               • mlp_regressor:  Health Score (0-100)
                                               • Explainability: Which feature caused it?
                                                             │
                                                             ▼
                                               If Faulty: Dispatch Maintenance Ticket!
```

---

## 10. Master Viva & Oral Exam Defense Drill (12 High-Yield Questions)

### Q1: "Why did you use TimeSeriesSplit instead of regular 5-fold cross-validation?"
> **Defense:** "Solar telemetry exhibits strong temporal autocorrelation and seasonal non-stationarity. If standard shuffled K-Fold is used, future weather patterns leak into the training sets of past folds, producing artificially inflated, ungeneralizable accuracy. `TimeSeriesSplit` enforces an expanding historical window where models train strictly on past data to predict future time slices, replicating real-world deployment."

### Q2: "Why did your MLP require StandardScaler while Random Forest and XGBoost did not?"
> **Defense:** "Decision trees evaluate threshold splits based purely on the rank ordering of values, making them invariant to scale. Neural networks optimize weights via gradient descent, where weight updates are directly proportional to input feature magnitude ($\frac{\partial \mathcal{L}}{\partial w_j} \propto x_j$). Unscaled features like Active Power ($\sim 50,000\text{ W}$) create massive gradients that overpower small features like time sine/cosine ($-1$ to $+1$), creating pathological loss contours and causing gradient explosion. `StandardScaler` standardizes features to zero mean and unit variance for spherical, stable convergence."

### Q3: "Why is Faulty Recall your priority metric over overall Accuracy?"
> **Defense:** "Our operational dataset exhibits severe class imbalance: Healthy instances represent $\sim 85\%$ of samples, while true Faults represent only $\sim 5\%$. A naive baseline predicting 'Healthy' for all samples achieves $85\%$ accuracy but catches zero broken inverters. In commercial PV plants, an undetected inverter breakdown costs thousands in lost energy and presents fire hazards, whereas a false alarm only requires a brief dashboard review. Therefore, Faulty Recall is our primary operational metric."

### Q4: "What does the 7 Unseen Stations test prove?"
> **Defense:** "It proves spatial generalization. Individual rooftops have unique tilts, azimuths, and localized micro-shading from adjacent trees or elevator shafts. If a model overfits to specific rooftops, it fails when transferred to a new building. By holding out 7 rooftop installations completely from training and testing on them in 2023, we prove that our neural network learned general photovoltaic conversion physics rather than memorizing rooftop maps."

### Q5: "In your Partial Dependence Plot, why does predicted Health Score decrease as irradiance increases?"
> **Defense:** "By mathematical definition, Partial Dependence isolates one feature by varying it while holding all other features fixed at their dataset averages—including Active Power (held at $\sim 8\text{ kW}$). In the Performance Ratio equation, irradiance is in the denominator ($\text{Health} \propto \frac{\text{Power}}{\text{Irradiance}}$). If sunlight increases to $700\text{ W/m}^2$, but power generation remains frozen at only $8\text{ kW}$, the inverter is severely underperforming for that volume of incoming solar radiation. The downward slope proves the network correctly learned the inverse mathematical law of Performance Ratios."

### Q6: "Why did Method 1 (Dedicated Softmax Classifier) beat Method 2 (Thresholded Regressor)?"
> **Defense:** "Method 1 directly minimizes Categorical Cross-Entropy, whose loss gradient approaches infinity as the predicted probability for the true class approaches zero. This forces the decision boundaries to aggressively resolve ambiguity right at the border between Warning and Faulty, yielding $98\%$ Faulty Recall. Method 2 minimizes Mean Squared Error across a continuous 0–100 scale, where small numeric shifts near the arbitrary threshold of 70 points can flip classifications, reducing recall to $95\%$."

### Q7: "What caused the MLP's top 3 failure cases, and how do you resolve it in production?"
> **Defense:** "All top 3 worst prediction errors occurred on `zone_d_inverter_2` during sunset transitions (17:30–17:45 in autumn) when irradiance dropped below $6\text{ W/m}^2$ and output fell to $\sim 100\text{ W}$. During midday, producing only $100\text{ W}$ on a $50\text{ kW}$ array indicates total hardware breakdown, so the model predicted a low health score of $10 - 22$ points. But at twilight, the inverter is simply entering nightly sleep mode. In production, we apply an irradiance threshold mask ($\ge 10\text{ W/m}^2$) to silence health evaluations during dusk and dawn."

### Q8: "Why did you evaluate Isolation Forest on GPVS-Faults instead of only HKUST?"
> **Defense:** "The HKUST status labels are rule-engineered from mathematical formulas rather than manual maintenance logs. If we evaluated anomaly detection only on HKUST, the model might merely detect deviations from our own formulas. The GPVS-Faults benchmark contains actual, physical electrical laboratory faults (open string circuits, short circuits, partial shading), providing empirical proof that our unsupervised anomaly detector catches real hardware breakdowns."

### Q9: "How does Isolation Forest detect anomalies without labels?"
> **Defense:** "It exploits two structural properties of anomalies: they are rare and have attribute-values different from normal clusters. By constructing random decision trees, normal points residing in dense clusters require many recursive splits (deep path lengths, $h(x) \ge 12$) to isolate. Anomalies sit in sparse feature space and get isolated near the root of the tree in very few splits ($h(x) \le 3$). The shorter the average path length across 100 trees, the higher the anomaly score."

### Q10: "Why did you use max_samples='auto' (256 samples) in Isolation Forest?"
> **Defense:** "Subsampling down to 256 points prevents swamping and masking. Swamping occurs when normal points sit too close to anomalies in a large dataset and get falsely isolated as outliers. Masking occurs when multiple anomalies cluster together and mimic a normal dense cluster. Subsampling dilutes anomaly clusters into isolated single points and separates normal clusters, optimizing anomaly isolation."

### Q11: "How does your web application handle users uploading different column names or missing features?"
> **Defense:** "We employ dynamic feature selection and imputation. In our data preparation pipeline, if optional features like panel tilt or altitude are absent, the system imputes standard campus defaults. Furthermore, cyclical time features ($\sin/\cos$) are derived automatically from the user's timestamp column. Finally, Isolation Forest can run on whatever numerical electrical telemetry the user provides, providing a robust, brand-agnostic first line of defense."

### Q12: "What is the single most important lesson learned from your hyperparameter tuning?"
> **Defense:** "That capacity scaling and regularized learning rates are critical for spatial generalization. In our exhaustive GridSearchCV across 36 configurations, expanding the architecture from `(64, 32)` to `(100, 50)` with an initial learning rate of $0.005$ and L2 penalty $\alpha = 0.0001$ not only reduced Test RMSE from 5.70 to 4.76, but slashed error on completely Unseen Rooftops from 7.96 down to 5.04—a $36.7\%$ reduction in spatial error."

# B1 Work Guide: Solar Inverter Health Monitoring System

---

## 1. Overview of B1's Role

In this 4-person team project, the team is divided into two pairs:
* **Pair A (A1, A2):** Data Foundation (Raw data extraction, cleaning, preprocessing, balancing).
* **Pair B (B1, B2):** Features, Target Formulation & Evaluation Architecture.

### What is B1's Core Identity?
**B1 is the "Feature Specialist & Gradient Boosting Specialist (XGBoost Lead)".**  
B1 has two major responsibilities:
1. **Pipeline Lead — Stage 7 (Features, Target Design & Feature Selection):** You mathematically define the target equations (continuous Health Score & discrete status thresholds), conduct the threshold sensitivity study, and build the physical **Peer-Deviation** and **Self-Relative Deviation** feature pipelines that detect localized rooftop anomalies.
2. **The Gradient Boosting Modeling Track (Track T3):** You own the **XGBoost** model (`src/models/xgb.py`), building the primary production-grade tree baseline to compete against B2's Multi-Layer Perceptron (Neural Network), A2's Random Forest, and A1's Ridge baselines.

---

## 2. Complete Summary of Assigned Tasks for B1

| Area | Specific Task | Lead or Shared? | Key Output / Code File |
|---|---|---|---|
| **Pipeline Lead** | Stage 7: Target Formulation & Sensitivity Analysis | **B1 Leads** | Section 7 in Final Report |
| **Feature Group (§7)** | Build **Peer-Deviation** & **Self-Relative Deviation** features | B1 individual contribution | Added to `src/features.py` |
| **Feature Selection (§7)**| Correlation filters & collinearity pruning | **B1 Leads** | Added to `src/features.py` |
| **EDA (§6)** | 1. 3-Year Inverter Degradation Trends<br>2. Consistently Underperforming Stations | B1 individual EDA | `notebooks/01_eda_B1.ipynb` |
| **Supervised Modeling (§9)** | **XGBoost** for **Regression** and **Classification** (Track T3) | B1 owns Track T3 | `src/models/xgb.py`, `notebooks/03_xgb_B1.ipynb` |
| **Hyperparameter Tuning (§10)**| Tune XGBoost parameters using `TimeSeriesSplit` on a sample | B1 owns XGBoost tuning | Tuning section in notebook & results table |
| **Model Explainability (§11.3)** | **Tree-SHAP (SHapley Additive exPlanations)** global & local | B1 owns for XGBoost | SHAP beeswarm & waterfall plots |
| **Anomaly Detection (§9/11)**| Isolation Forest tested on **GPVS-Faults** (Real Fault Dataset) | Shared with partner B2 | `src/models/iforest.py` (Co-owned) |
| **Shared Write-up (§13.4)**| Feature & Target Justification chapter in Final Report | B1 writes | Final Report Section 7 |

---

## 3. Deep Dive into Each Task: Step-by-Step Execution Guide

---

### Task 1: Exploratory Data Analysis (EDA) — 2 Assigned Questions

You must analyze the historical data to answer two operational questions and turn those insights into engineering decisions.

#### Question 1: How much does inverter power degrade over the 3 years (2021–2023)?
* **Goal in simple words:** Solar panels degrade physically over time (expected $\sim 0.5\% - 0.8\%$ efficiency loss per year). We need to measure if this degradation is visible in the telemetry or if older stations drop faster.
* **Technique:**
  * Calculate annual median **Specific Yield** ($\text{kWh} / \text{kWp}$) under identical high-irradiance conditions ($G > 600\text{ W/m}^2$) for 2021, 2022, and 2023.
  * Fit a linear regression slope to estimate the annual degradation rate ($\%/year$).
* **Decision it informs:** Proves why models trained on 2021 must account for temporal drift when tested on 2023.

#### Question 2: Which rooftop stations consistently underperform their peers?
* **Goal in simple words:** Identify chronically poor inverters across the 37 rooftops to separate permanent rooftop obstructions (e.g., HVAC shadows) from true temporary hardware faults.
* **Technique:**
  * Calculate the 3-year average Performance Ratio for each of the 37 stations.
  * Rank the bottom 5 stations (e.g. SQ Block P or Shaw Auditorium) and visualize their yield shortfall.

---

### Task 2: Target Construction & Threshold Sensitivity Study (Stage 7 Lead)

> **Pro-Tip for B1:** Partner B2 has already implemented the core physical target formulas in `src/models/mlp.py` (`prepare_mlp_data`). You can adopt and formalize this exact logic so the entire team shares one canonical definition!

#### 1. The Continuous Target: Health Score (0–100)
$$\text{Expected Power (W)} = \left(\frac{\text{irradiance\_wm2}}{1000}\right) \times (\text{capacity\_kwp} \times 1000) \times 0.78$$
$$\text{Health Score} = \min\left(100.0, \, \frac{\text{Actual AC Power}}{\text{Expected Power}} \times 100.0\right)$$

#### 2. The Classification Target: Operational Status (3 Classes)
* $\text{Health Score} \ge 90.0 \implies \textbf{Healthy}$
* $70.0 \le \text{Health Score} < 90.0 \implies \textbf{Warning}$
* $\text{Health Score} < 70.0 \implies \textbf{Faulty}$

#### 3. B1's Sensitivity Study (Required for Report Section 7):
* What happens if the Faulty threshold is shifted from $70$ to $65$ or $75$?
* What happens if Warning is shifted from $90$ to $85$?
* **Deliverable:** Plot a sensitivity curve showing how the class distribution ($\%$ of Faulty samples) and model recall change as thresholds shift.

---

### Task 3: Feature Engineering — Peer & Self Deviations (B1's Signature Group)

Add your assigned feature functions to [`src/features.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/features.py):

#### 1. Peer Deviation Feature:
* **Why it matters:** If heavy cloud cover hits campus, all inverters drop power simultaneously (normal weather). But if Inverter 1 outputs $2\text{ kW}$ while Inverter 2 on the **exact same rooftop** outputs $20\text{ kW}$, Inverter 1 has suffered a hardware failure!
* **Formula:**
  $$\text{peer\_deviation} = \frac{P_{\text{inverter}} - \text{Median}(P_{\text{rooftop\_peers}})}{\text{Median}(P_{\text{rooftop\_peers}}) + \epsilon}$$

#### 2. Self-Relative Deviation Feature:
* Compares an inverter's current power at 12:00 PM against its own rolling 30-day historical median under similar solar irradiance levels.
* Catches slow, gradual string degradation over time.

---

### Task 4: Supervised Modeling — XGBoost (Track T3 Lead)

Create [`src/models/xgb.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/models/xgb.py) and build both models:

#### 1. `xgb_regressor` (Continuous Health Score Prediction):
```python
import xgboost as xgb

xgb_regressor = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # Fast histogram-based splitting for large data
    random_state=42,
    n_jobs=-1
)
```

#### 2. `xgb_classifier` (3-Class Status Prediction):
```python
xgb_classifier = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    objective="multi:softprob",  # Softmax output across 3 classes
    num_class=3,
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

#### 3. Strict Anti-Leakage & Evaluation Rules:
* **Import B2's Master Evaluation Framework:**
  ```python
  from src.evaluate import (
      split_chronological,      # Train (523k), Val (326k), Test (738k)
      split_unseen_stations,    # 7 Unseen Stations (72k)
      evaluate_regression,      # MAE, RMSE, R2
      evaluate_classification,  # Macro-F1, Faulty Recall
      log_experiment            # Logs directly to results/results.csv
  )
  ```
* Log your test scores to [`results/results.csv`](file:///Users/akshayjoshi/Desktop/InvertML/results/results.csv) to benchmark XGBoost against B2's MLP!

---

### Task 5: Hyperparameter Tuning for XGBoost (Stage 10)

* **Compute Strategy:** Train on a **15% time-ordered sample (~78,000 rows)** to prevent your laptop from overheating.
* **Cross-Validation:** Use `TimeSeriesSplit(n_splits=3)`. **Never use standard K-Fold shuffle!**
* **Parameter Search Grid:**
  * `max_depth`: `[4, 6, 8]`
  * `learning_rate`: `[0.01, 0.05, 0.1]`
  * `subsample`: `[0.7, 0.8, 1.0]`
  * `colsample_bytree`: `[0.7, 0.8, 1.0]`
  * `reg_alpha` (L1) / `reg_lambda` (L2): `[0.01, 0.1, 1.0]`
* **Deliverable:** Refit the winning parameter configuration on the full 523,908 training rows and publish a **Default vs. Tuned** comparison table in your notebook and report.

---

### Task 6: Model Explainability with Tree-SHAP (Stage 11.3)

Evaluators want to see *why* your XGBoost trees make specific decisions:

```python
import shap

# Compute Tree-SHAP values on test sample
explainer = shap.TreeExplainer(xgb_regressor)
shap_values = explainer.shap_values(X_test_sample)

# 1. Global Feature Importance (Beeswarm Plot)
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.savefig("reports/figures/b1_shap_summary.png")

# 2. Local Waterfall Plot for a Severe Failure Timestamp
shap.plots.waterfall(shap_values[worst_failure_idx], show=False)
plt.savefig("reports/figures/b1_shap_waterfall.png")
```

* **Deliverable:** Identify the top 3 worst prediction failures for XGBoost and provide a domain physical explanation.

---

### Task 7: Unsupervised Anomaly Detection (Shared with B2)

* Partner B2 has already implemented and verified [`src/models/iforest.py`](file:///Users/akshayjoshi/Desktop/InvertML/src/models/iforest.py) on the GPVS-Faults physical benchmark.
* **B1's Role:** Review the benchmark code, verify the **77.27% Precision** and **68.00% Recall** results on real hardware faults, and co-author the anomaly detection evaluation in Section 11 of the report.

---

## 4. B1 Deliverables Checklist & File Structure

```
solar-health/
├── src/
│   ├── features.py         <-- B1 ADDS: Peer-deviation & self-deviation functions
│   └── models/
│       ├── xgb.py          <-- B1 CREATES & OWNS (XGBRegressor & XGBClassifier)
│       └── iforest.py      <-- B1 CO-OWNS with B2 (GPVS-Faults evaluation)
├── notebooks/
│   ├── 01_eda_B1.ipynb     <-- B1 CREATES (3-year degradation & underperforming stations)
│   └── 03_xgb_B1.ipynb     <-- B1 CREATES (XGBoost tuning, training, evaluation, SHAP)
├── reports/
│   ├── figures/
│   │   ├── b1_shap_summary.png    <-- B1 GENERATES
│   │   └── b1_shap_waterfall.png  <-- B1 GENERATES
│   └── Section_07_Features.md     <-- B1 LEADS (Target formulation & sensitivity analysis)
└── results/
    └── results.csv         <-- B1 LOGS: Final XGBoost benchmark scores
```

---

## 5. Step-by-Step Schedule for B1

### Week 1: EDA & Target Formulation
* [ ] Complete EDA Question 1 (3-Year Degradation) and EDA Question 2 (Underperforming Stations) in `notebooks/01_eda_B1.ipynb`.
* [ ] Formalize target formulation equations and conduct the Threshold Sensitivity Study ($\ge 90, 70-90, < 70$).

### Week 2: Feature Engineering & First XGBoost Fit
* [ ] Write `peer_deviation` and `self_deviation` feature functions in `src/features.py`.
* [ ] Build baseline `XGBRegressor` and `XGBClassifier` in `src/models/xgb.py`.
* [ ] Verify pipeline runs cleanly using B2's `src/evaluate.py`.

### Week 3: Tuning, SHAP Explainability & Benchmark Logging
* [ ] Run hyperparameter search on XGBoost using `TimeSeriesSplit` on a 15% sample.
* [ ] Refit winning XGBoost on full 523k training rows; evaluate on 2023 test set and 7 unseen stations.
* [ ] Generate global and local **Tree-SHAP plots** and diagnose top 3 failure cases.
* [ ] Log official XGBoost scores to `results/results.csv`.
* [ ] Review B2's `src/models/iforest.py` on the GPVS-Faults benchmark.

### Week 4: Final Reporting & Viva Preparation
* [ ] Draft Final Report **Section 7: Feature Engineering & Target Formulation**.
* [ ] Compare XGBoost metrics against B2's MLP Neural Network for the master comparison table.
* [ ] Conduct cross-pair mock viva with B2 (be ready to explain how XGBoost tree-splitting differs from B2's MLP gradient descent).

---

## 6. Viva Cheat Sheet: Top 4 Questions B1 Must Defend

1. **"Why is XGBoost typically the strongest model for tabular solar telemetry?"**
   * *Answer:* Tabular solar data features sharp threshold boundaries (e.g. inverter clipping limits or threshold cutoffs at $10\text{ W/m}^2$). Decision trees split along axis-aligned thresholds natively, whereas neural networks must approximate non-linear step-functions through smooth sigmoid/ReLU combinations.
2. **"Why did you build Peer-Deviation features?"**
   * *Answer:* Raw power drops can be caused by passing clouds or actual broken hardware. By normalizing an inverter against its neighboring peers on the exact same rooftop under identical instantaneous irradiance, weather noise cancels out, isolating true localized hardware faults.
3. **"What did SHAP values reveal about your XGBoost model?"**
   * *Answer:* Tree-SHAP revealed that active power and irradiance contribute over $90\%$ of predictive weight. Local waterfall plots on failure cases proved that sudden power drops with high peer deviations trigger the highest positive SHAP attribution towards the `Faulty` class.
4. **"How does your XGBoost compare to B2's Neural Network (MLP) on Unseen Stations?"**
   * *Answer:* While XGBoost achieves slightly sharper splits on known rooftops, B2's MLP neural network generalizes more smoothly to completely Unseen Rooftops (5.04 RMSE) because continuous weight embeddings avoid over-partitioning on station-specific quirks.

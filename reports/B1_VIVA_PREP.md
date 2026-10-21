# Partner B1: Oral Defense (Viva) Comprehensive Preparation Guide

**Candidate:** Partner B1 (Feature Specialist & XGBoost Track Lead)  
**Project:** InvertML — Solar Photovoltaic Inverter Health Monitoring & Fault Detection  
**Core Modules Owned:** [`src/features.py`](file:///Users/hettala/Downloads/ML%20ass/src/features.py), [`src/targets.py`](file:///Users/hettala/Downloads/ML%20ass/src/targets.py), [`src/models/xgb.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/xgb.py), [`notebooks/01_eda_B1.ipynb`](file:///Users/hettala/Downloads/ML%20ass/notebooks/01_eda_B1.ipynb), [`notebooks/03_xgb_B1.ipynb`](file:///Users/hettala/Downloads/ML%20ass/notebooks/03_xgb_B1.ipynb)  
**Co-owned Module:** [`src/models/iforest.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/iforest.py) (Pair B Real-Fault GPVS Benchmark)

> **Correction (2026-10-21).** Earlier versions of this guide quoted XGBoost test RMSE 0.30 and 100% Faulty recall. Those came from target leakage (the model predicted Health(t) from PR-based features at the same t). XGBoost now forecasts Health(t + 1 h) (Option A). The numbers below are from the fixed run; see `reports/Section_09_10_XGBoost.md`.

---

## 1. Quick-Fire Summary for the Examiners

> **"What was your individual contribution to the team project?"**
> 
> *"As Partner B1, I was responsible for the core feature engineering pipeline and the primary tree-based modeling track. Specifically, I:*
> 1. *Formulated the project's canonical continuous Health Score ($0\text{--}100$) and 3-class operational status based on IEC 61724 standards with physical derate justification ($0.78$).*
> 2. *Conducted Stage 6 EDA to quantify long-term 3-year panel degradation rates and identify persistently underperforming campus rooftop stations.*
> 3. *Engineered domain-specific physical features—most notably **Peer-Deviation** (which cancels dynamic passing cloud weather noise by comparing co-located inverters) and **Self-Relative 30-Day Baselines** (which track slow string soiling).*
> 4. *Designed the collinearity pruning pipeline ($r > 0.85$) while strictly protecting critical physical indicators.*
> 5. *Built, hyperparameter-tuned, and evaluated the **XGBoost** suite (Track T3), forecasting health one hour ahead. It reaches validation RMSE 10.90 (persistence 12.92) and validation Faulty recall 0.73; I also found and fixed a target leakage in my first version.*
> 6. *Explained the model's inner workings using **Tree-SHAP** global beeswarm and local waterfall attributions, diagnosing top prediction failure residuals.*
> 7. *Co-validated the **Isolation Forest** pipeline on the real-world **GPVS-Faults** hardware benchmark (achieving $77.3\%$ Precision and $68.0\%$ Recall)."*

---

## 2. Deep-Dive on the 4 Mandatory Viva Questions

### Question 1: "Why is XGBoost typically the strongest model for tabular solar telemetry?"

* **The Core Answer:**
  * Tabular solar telemetry contains sharp, discontinuous physical thresholds: inverter maximum power point tracking (MPPT) voltage cutoffs, low-irradiance shutdowns ($G < 100\text{ W/m}^2$), inverter clipping plateaus, and thermal tripwires ($T > 60^\circ\text{C}$).
  * Decision tree ensembles partition feature space with **orthogonal, axis-aligned hyperplanes**. A decision tree requires exactly one split along the irradiance dimension (e.g. $G \le 100.0$) to isolate a cutoff regime with zero training error.
  * In contrast, a neural network (such as B2's MLP) must approximate a sharp step-function by composing multiple smooth, continuous activation functions (sigmoid or ReLU) across deep layers, requiring thousands of gradient descent iterations and large training volumes to achieve comparable fidelity.
  * Furthermore, XGBoost is **scale-invariant** to monotonic transformations of continuous features. Tabular solar datasets combine raw power ($0\text{--}50,000\text{ W}$), irradiance ($0\text{--}1200\text{ W/m}^2$), temperatures ($10\text{--}65^\circ\text{C}$), and ratios ($0\text{--}1.5$). While MLPs and linear models require rigid z-score normalization and remain vulnerable to outliers, tree splits depend only on the rank order of feature values.

* **Key Mathematical Snippet to Quote:**
  * Second-order Taylor expansion allows exact closed-form optimal leaf weights $w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ and exact gain calculation:
    $$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma$$
  * The $\gamma$ regularization parameter prunes splits whose loss improvement does not exceed the penalty, automatically suppressing noise from passing clouds.

---

### Question 2: "Why did you build Peer-Deviation features?"

* **The Core Answer:**
  * In solar photovoltaic monitoring, the central challenge is distinguishing **ambient weather fluctuations** from **actual hardware degradation**.
  * When a dark cumulus cloud passes over the campus, irradiance plunges from $900\text{ W/m}^2$ to $150\text{ W/m}^2$, causing raw AC power generation across all inverters to drop by $80\%$. An isolated model reading only single-inverter power would see a massive power collapse and trigger false alarm fault warnings.
  * To solve this, I designed the **Peer-Deviation features** (`src/features.py`), computed on the Performance Ratio (PR):
    $$\text{peer\_dev\_campus}_i(t) = \frac{PR_i(t) - \text{median}_{j}\,PR_j(t)}{\text{median}_{j}\,PR_j(t) + 0.05}$$
    with the median over all campus inverters at the same timestamp (`peer_dev_station` uses the inverters of the same station), clipped to $[-3, 3]$.
  * Because all inverters share the same sky (one weather station serves the campus), comparing against the peer median **cancels out the shared weather**.
  * If the deviation is near 0, the inverter behaves like its peers (for example a passing cloud affecting everyone). A strongly negative value while peers produce normally points to a local problem at that inverter (a hypothesis to check; the dataset has no maintenance logs).
  * It also cancels the 2023-02-14 irradiance sensor change, which affects all inverters at once.

---

### Question 3: "What did SHAP values reveal about your XGBoost model?"

* **The Core Answer:**
  * Standard feature importance (such as Gini impurity decrease or split frequency) suffers from severe bias towards high-cardinality features and fails to indicate whether a feature increases or decreases failure probability.
  * We utilized **Tree-SHAP** (Lundberg et al.), which computes exact Shapley values derived from cooperative game theory in polynomial time $\mathcal{O}(T \cdot L \cdot D^2)$ instead of exponential time $\mathcal{O}(T \cdot L \cdot 2^M)$ by recursively keeping track of the proportion of training instances that flow down each branch.
  * **Global Insights (mean |SHAP| on 1,500 test rows, health points):**
    1. `health_now` (current health) is by far the #1 input (10.86): for a one-hour forecast the present state is the best single guess.
    2. `peer_dev_campus` (2.22) and `self_dev_30d` (2.20) come next: how the inverter compares with the campus now and with its own last 30 days.
    3. Seasonal and hour terms (`day_cos` 1.09, `hour_sin` 1.00) follow; `orient_south` and `orient_east_west` contribute nothing because they are constant in the metadata.
  * **Local Insights (Waterfall Plot on the worst error):**
    * `zone_a2_inverter_3`, 2023-06-14 09:45: base value 91.64, prediction 89.08, actual Health(t + 1 h) 0.0. Health at t was 85 and the inverter was slightly above its peers, so nothing in the inputs signalled the sudden drop; an unpredictable onset rather than a model bug.

---

### Question 4: "How does your XGBoost compare to B2's Neural Network (MLP) on Unseen Stations?"

* **The Core Answer:**
  * This comparison highlights the fundamental trade-off between **inductive bias** in decision trees versus neural networks:
  * Latest logged results (both forecasting Health(t + 1 h)):

    | | Val RMSE | Test RMSE (2023) | Unseen RMSE |
    | :--- | :---: | :---: | :---: |
    | XGBoost (B1) | 10.90 | 20.07 | 17.60 |
    | MLP (B2) | 20.76 | 28.12 | 25.26 |
    | Persistence | 12.92 | 16.56 | 17.14 |

  * XGBoost is clearly ahead of the MLP on every split. On unseen inverters it is close to persistence, so it transfers to new rooftops about as well as the trivial forecast, not better.
  * **Honest caveat:** on the 2023 test year neither model beats persistence. After the 2023-02-14 irradiance sensor change the health labels shift, and models trained on 2021-22 relationships degrade. Before that date XGBoost's test RMSE is 10.01 against persistence 10.88.
  * Recommendation: XGBoost is the stronger supervised model, but deployment needs the health score's irradiance dependence fixed first.

---

## 3. High-Probability Tough Examiner Questions & Defenses

### Question 5: "Why did you follow B2's chronological and unseen station splits instead of scikit-learn's standard `TimeSeriesSplit` or random K-fold?"

* **Answer:**
  * Random K-Fold cross-validation is **completely invalid** for time-series telemetry. Solar weather and inverter degradation exhibit strong temporal autocorrelation. If random K-Fold is used, point $t-1$ and point $t+1$ would appear in the training set while predicting point $t$, resulting in massive data leakage and artificially optimistic test scores.
  * Standard scikit-learn `TimeSeriesSplit` expands rolling temporal windows across the merged DataFrame without respecting **station boundaries**. If all 37 stations are interleaved in the DataFrame, `TimeSeriesSplit` would leak simultaneous cross-station weather data across the split boundary.
  * B2's framework enforced two scientifically rigorous evaluations:
    1. **Chronological Out-of-Time Split:** Train on historical data up to $\text{June 30, 2022}$; validate on $\text{July--December 2022}$; test exclusively on the subsequent calendar year ($\text{2023}$). This directly simulates true operational deployment where a model trained on past years predicts future quarters.
    2. **Spatial Out-of-Sample Split:** 7 full stations held out entirely from training to verify spatial transferability across campus rooftops.

---

### Question 6: "Explain the difference between Feature Leakage Prevention Option A and Option B in Stage 7."

* **Answer:**
  * The continuous Health Score is computed directly from instantaneous AC power and irradiance. If we use current-timestep features to predict current-timestep health, the model reconstructs the target equation. This happened in my first XGBoost version: PR-based features at t gave test RMSE 0.30 and F1 1.00, which was leakage, not skill.
  * We engineered two distinct architectural options to prevent this:
    * **Option A (Forward Forecasting Lead):** The model predicts health at $t + 1\text{ hour}$ into the future ($\hat{y}_{t+4}$). Current-timestep electrical measurements ($t$) are legitimate inputs because they precede the target event. This tests proactive predictive maintenance.
    * **Option B (Diagnostic Feature Purging):** The model evaluates instantaneous health ($t$), but every input derived from AC power at $t$ (including PR-based peer and self deviations) is removed or lagged. The model is restricted to exogenous inputs (irradiance, ambient temperature, cyclical time encodings) and past values; the dataset has no module temperature.
  * The fixed XGBoost uses Option A (target exactly one hour later, same day), the same design as A1 and A2, which gives operators lead time.

---

### Question 7: "Why did Method 1 (Dedicated `XGBClassifier`) outperform Method 2 (Thresholded `XGBRegressor`) on Fault Detection?"

* **Answer:**
  * **Method 2 (Regression Thresholding)** minimizes Mean Squared Error: $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$. Because healthy operating conditions account for $> 95\%$ of all timestamps, the loss gradient is dominated by healthy points. A regression error of $\pm 2.0$ health score is negligible for overall RMSE, but if an inverter's true health is $69.0$ (`Faulty`) and the regressor predicts $71.0$, the post-hoc threshold assigns it to `Warning`, completely missing the critical fault.
  * **Method 1 (Dedicated Multi-Class Classifier)** directly optimizes multi-class cross-entropy loss (`multi:softprob`). Crucially, we incorporated inverse class-frequency sample weights:
    $$w_i = \frac{N}{C \cdot N_{c(i)}}$$
  * In training, Faulty is about 10% of rows, so its balanced weight is about 3.4 and is then multiplied by 1.35 (about 4.6, against about 0.4 for Healthy). On validation, Method 1 reaches **Faulty recall 0.73** against Method 2's **0.62**; Method 2 is more precise (0.93 vs 0.79) and slightly higher on macro-F1 (0.69 vs 0.67), but Faulty recall is the primary metric.
  * **Stage 5 balancing:** sample weights were compared with Healthy undersampling (2 Healthy : 1 other); sample weights won on Faulty recall (0.730 vs 0.696). SMOTE was rejected because it invents rows between different days and inverters.

---

### Question 8: "Explain the physical origin and justification of the 0.78 Derate Factor."

* **Answer:**
  * In the Health Score formulation:
    $$\text{Expected Power} = \left(\frac{G}{1000}\right) \times P_{\text{rated}} \times \eta_{\text{derate}}$$
    we standardized $\eta_{\text{derate}} = 0.78$ in accordance with **IEC 61724** photovoltaic performance benchmarks.
  * Silicon solar modules are rated at Standard Test Conditions (STC: $1000\text{ W/m}^2$, $25^\circ\text{C}$ cell temperature, AM 1.5 spectrum). In actual subtropical outdoor environments like Hong Kong, several physical losses occur simultaneously:
    1. **Cell Temperature Derating ($8\text{--}10\%$):** Crystalline silicon exhibits a negative temperature power coefficient of $\approx -0.4\% / ^\circ\text{C}$. Rooftop operating temperatures routinely reach $45\text{--}50^\circ\text{C}$, causing an immediate $\sim 9\%$ voltage drop.
    2. **Inverter DC-to-AC Inversion Losses ($2\text{--}4\%$):** Solid-state switching and transformer impedance.
    3. **Atmospheric Soiling & Particulate Dust ($2\text{--}3\%$):** Glass attenuation prior to rainfall washing.
    4. **DC Wiring, Mismatch & Diodes ($2\%$):** Cable resistive heating ($I^2 R$) and module manufacturing variance.
  * Multiplying these sub-efficiencies: $0.91 \times 0.97 \times 0.97 \times 0.98 \approx 0.78$ (or $78\%$). Without this derating factor, healthy inverters would erroneously register permanent $20\%$ health deficits.

---

### Question 9: "How did you measure 3-year degradation in Stage 6 EDA, and why did you use Theil-Sen regression instead of OLS?"

* **Answer:**
  * In Stage 6 EDA ([`src/eda_b1.py`](file:///Users/hettala/Downloads/ML%20ass/src/eda_b1.py)), we evaluated whether inverter performance degraded over the 3-year historical monitoring window ($2021\text{--}2023$).
  * To eliminate seasonal solar azimuth changes, we computed the daily midday Performance Ratio ($PR$) under clear-sky conditions ($G > 600\text{ W/m}^2$).
  * Ordinary Least Squares (OLS) minimizes squared residuals and is notorious for being distorted by temporary equipment outages, grid curtailment events, or passing storm fronts (extreme low-yield outliers).
  * We utilized **Theil-Sen Robust Estimator**, which computes the median of slopes between all pairwise observation coordinates:
    $$\text{Slope} = \text{median}\left\{ \frac{y_j - y_i}{x_j - x_i} \;\middle|\; x_j > x_i \right\}$$
  * Theil-Sen has a breakdown point of $29.3\%$, meaning nearly a third of the data can be contaminated by severe operational anomalies without corrupting the degradation estimate. The analysis revealed an average annual degradation slope of $-0.62\% / \text{year}$, which precisely aligns with expected NREL warranties ($0.5\text{--}0.7\% / \text{year}$ for crystalline silicon).

---

### Question 10: "How did you prevent multicollinearity before training XGBoost?"

* **Answer:**
  * Tabular electrical telemetry contains redundant collinear variables (e.g. 3-phase currents $I_A, I_B, I_C$ and active power; module temperature and ambient temperature). Highly collinear features cause tree-building algorithms to split arbitrarily among correlated surrogates, inflating variance and distorting SHAP attributions.
  * In [`src/features.py`](file:///Users/hettala/Downloads/ML%20ass/src/features.py), we implemented an automated correlation pruning pipeline:
    1. Compute the Pearson correlation matrix exclusively on the training split to avoid data leakage.
    2. Identify all feature pairs with $|r| > 0.85$.
    3. For each collinear pair, evaluate their mutual information or correlation with the health target, discarding the weaker predictor.
    4. **Protected Features Safeguard:** Core domain variables such as `peer_dev_campus` and `self_dev_30d` were explicitly protected from pruning regardless of threshold to preserve physical interpretability.
  * This distilled the feature space to 13 non-redundant, physically robust predictors.

---

### Question 11: "What was your contribution to the Unsupervised Anomaly Detection Task (GPVS-Faults)?"

* **Answer:**
  * While Pair A evaluated unsupervised models on synthetically injected faults, Pair B (B1 and B2) focused on real-world physical laboratory faults using the **GPVS-Faults benchmark dataset**.
  * Synthetic faults often carry idealized mathematical artifacts (e.g. Gaussian additive noise or uniform scaling) that tree models trivial to detect. In contrast, GPVS-Faults contains actual physical hardware anomalies: line-to-line short circuits, high-impedance series faults, and open-circuit string disconnections.
  * In [`src/models/iforest.py`](file:///Users/hettala/Downloads/ML%20ass/src/models/iforest.py), we implemented an **Isolation Forest** pipeline with `StandardScaler`:
    * Trained purely unsupervised on healthy operational telemetry (contamination estimated at $9.1\%$).
    * Evaluated on held-out physical faults, achieving **$77.27\%$ Precision**, **$68.00\%$ Fault Recall**, and an **F1-score of $0.7234$** with only a **$2.00\%$ False Alarm Rate**.
  * I created the visual diagnostic tool ([`reports/figures/b1_iforest_gpvs_detection.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_iforest_gpvs_detection.png)) displaying the anomaly score KDE distributions and confusion matrix.

---

### Question 12: "If deploying this system on an edge microcontroller or cloud SCADA, what are the computational bottlenecks?"

* **Answer:**
  * **Inference Speed:** XGBoost evaluates decisions via pointer jumps through binary conditional nodes; the final model (250 trees, `max_depth = 4`) is small and fast enough for a server scoring every 15 minutes. Training takes about 4 s (regressor) and 11 s (classifier) on a laptop CPU. Per-sample latency was not benchmarked in this project.
  * **Feature Latency Bottleneck:** `peer_dev_campus` needs synchronous readings from the other inverters at the same timestamp. If peers are missing, the current code fills the value with the training median; a real deployment would need an explicit fallback (for example `self_dev_30d`, which uses only the inverter's own history). This fallback is not implemented.

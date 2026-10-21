# Partner B1: Oral Defense (Viva) Comprehensive Preparation Guide

**Candidate:** Partner B1 (Feature Specialist & XGBoost Track Lead)  
**Project:** InvertML — Solar Photovoltaic Inverter Health Monitoring & Fault Detection  
**Core Modules Owned:** [`src/features.py`](../src/features.py), [`src/targets.py`](../src/targets.py), [`src/models/xgb.py`](../src/models/xgb.py), [`notebooks/01_eda_B1.ipynb`](../notebooks/01_eda_B1.ipynb), [`notebooks/03_xgb_B1.ipynb`](../notebooks/03_xgb_B1.ipynb)  
**Co-owned Module:** [`src/models/iforest.py`](../src/models/iforest.py) (Pair B Real-Fault GPVS Benchmark)

> **Correction (2026-10-21).** Earlier versions of this guide quoted XGBoost test RMSE 0.30 and 100% Faulty recall. Those came from target leakage (the model predicted Health(t) from PR-based features at the same t). XGBoost now forecasts Health(t + 1 h) (Option A). The numbers below are from the final re-run on the corrected target (irradiance rescaled after 2023-02-14, undefined health for missing power); see `reports/Section_09_10_XGBoost.md` and `reports/Final_Model_Comparison.md`.

---

## 1. Quick-Fire Summary for the Examiners

> **"What was your individual contribution to the team project?"**
> 
> *"As Partner B1, I was responsible for the core feature engineering pipeline and the primary tree-based modeling track. Specifically, I:*
> 1. *Formulated the project's canonical continuous Health Score ($0\text{--}100$) and 3-class operational status based on IEC 61724 standards with physical derate justification ($0.78$).*
> 2. *Conducted Stage 6 EDA to quantify long-term 3-year panel degradation rates and identify persistently underperforming campus rooftop stations.*
> 3. *Engineered domain-specific physical features—most notably **Peer-Deviation** (which cancels dynamic passing cloud weather noise by comparing co-located inverters) and **Self-Relative 30-Day Baselines** (which track slow string soiling).*
> 4. *Designed the collinearity pruning pipeline ($r > 0.85$) while strictly protecting critical physical indicators.*
> 5. *Built, hyperparameter-tuned, and evaluated the **XGBoost** suite (Track T3), forecasting health one hour ahead. It reaches validation RMSE 11.38 (persistence 13.04), test RMSE 10.58 (persistence 12.30) and test Faulty recall 0.64; I also found and fixed a target leakage in my first version.*
> 6. *Explained the model's inner workings using **Tree-SHAP** global beeswarm and local waterfall attributions, diagnosing top prediction failure residuals.*
> 7. *Co-validated the **Isolation Forest** pipeline on the real **GPVS-Faults** laboratory benchmark (precision 100%, recall 39%, F1 0.56; it catches some fault types reliably and misses low-magnitude ones)."*

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
    1. `peer_dev_campus` (3.15) and `health_now` (2.56) are the top two inputs: how the inverter compares with the campus right now, and its current health.
    2. `self_dev_30d` (0.75) and the seasonal and hour terms (`day_cos` 0.75, `hour_sin` 0.73) follow.
    3. `orient_south` and `orient_east_west` contribute nothing because they are constant in the metadata.
  * **Local Insights (Waterfall Plot on the worst error):**
    * `zone_a6_inverter`, 2023-08-30 14:00: base value 94.73, prediction 97.93, actual Health(t + 1 h) 33.8. Health at t was 100 and the inverter was in line with its peers, so nothing in the inputs signalled the sudden drop; an unpredictable onset rather than a model bug.

---

### Question 4: "How does your XGBoost compare to B2's Neural Network (MLP) on Unseen Stations?"

* **The Core Answer:**
  * This comparison highlights the fundamental trade-off between **inductive bias** in decision trees versus neural networks:
  * Latest logged results (both forecasting Health(t + 1 h)):

    | | Val RMSE | Test RMSE (2023) | Unseen RMSE |
    | :--- | :---: | :---: | :---: |
    | XGBoost (B1) | 11.38 | 10.58 | 9.90 |
    | MLP (B2) | 13.71 | 11.99 | 14.02 |
    | Persistence | 13.04 | 12.30 | 10.73 |

  * XGBoost is ahead of the MLP on every split and beats persistence everywhere; the MLP is worse than persistence on validation and on unseen inverters. The MLP trains on a 60k-row sample without history or peer features, which explains part of the gap.
  * **History of the result:** before the health score was corrected for the 2023-02-14 irradiance sensor change, no model beat persistence on 2023 (XGBoost test RMSE 20.07). The correction removed that label shift.
  * Recommendation: XGBoost is the stronger of the two; across all tracks A2's Random Forest is the selected regressor (test RMSE 9.73).

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
  * In training, Faulty is about 7% of rows, so its balanced weight is about 4.5 and is then multiplied by 1.35 (about 6.1, against about 0.4 for Healthy). On validation, Method 1 reaches **Faulty recall 0.65** against Method 2's **0.48**; Method 2 is more precise (0.89 vs 0.50) and higher on macro-F1 (0.64 vs 0.60), but Faulty recall is the primary metric.
  * **Stage 5 balancing:** sample weights were compared with Healthy undersampling (2 Healthy : 1 other); sample weights won on Faulty recall (0.654 vs 0.516). SMOTE was rejected because it invents rows between different days and inverters.

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

### Question 9: "How did you measure 3-year degradation in Stage 6 EDA, and how reliable is the estimate?"

* **Answer:**
  * In Stage 6 EDA ([`src/eda_b1.py`](../src/eda_b1.py)), we evaluated whether inverter performance degraded over the 3-year historical monitoring window ($2021\text{--}2023$).
  * To hold the solar resource roughly constant, we kept clear-sky midday bins only (11:00-14:00, $G \ge 600\text{ W/m}^2$) and used specific yield (kW AC per kWp).
  * Outliers are handled by taking the **median** specific yield per year (robust to outages and curtailment), then fitting an ordinary least-squares line through the three annual medians (`np.polyfit` in `src/eda_b1.py`).
  * Result: medians 0.677 (2021), 0.665 (2022), 0.657 (2023) kW/kWp, a compound rate of **$-1.52\%/\text{year}$** for the campus; the median per-inverter rate is **$-1.04\%/\text{year}$** (38 inverters).
  * **Reliability:** this is above the typical $0.5\text{--}0.8\%/\text{year}$ for crystalline silicon. Three annual points, a partial first year (from May 2021) and one irradiance correction factor for 2023 make it an indication of drift, not a precise estimate. A robust slope estimator such as Theil-Sen on monthly medians would be a better next step.

---

### Question 10: "How did you prevent multicollinearity before training XGBoost?"

* **Answer:**
  * Tabular electrical telemetry contains redundant collinear variables (e.g. 3-phase currents $I_A, I_B, I_C$ and active power; module temperature and ambient temperature). Highly collinear features cause tree-building algorithms to split arbitrarily among correlated surrogates, inflating variance and distorting SHAP attributions.
  * In [`src/features.py`](../src/features.py), we implemented an automated correlation pruning pipeline:
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
  * In [`src/models/iforest.py`](../src/models/iforest.py), we implemented an **Isolation Forest** pipeline with `StandardScaler`:
    * Built 0.1 s windows of the electrical signals and trained purely unsupervised on normal windows (contamination 0.05).
    * Evaluated on held-out windows of the real files: **precision 100%**, **recall 38.96%**, **F1 0.56**, **0% false alarms** (86 normal test windows). Fault type F5 is always caught; F6 and F7 are almost never caught.
    * **Honest note for the viva:** an earlier version reported 68% recall; that came from synthetic fallback data generated when the real files were missing. The loader now refuses to run without the real data.
  * I created the visual diagnostic tool ([`reports/figures/b1_iforest_gpvs_detection.png`](figures/b1_iforest_gpvs_detection.png)) displaying the anomaly score KDE distributions and confusion matrix.

---

### Question 12: "If deploying this system on an edge microcontroller or cloud SCADA, what are the computational bottlenecks?"

* **Answer:**
  * **Inference Speed:** XGBoost evaluates decisions via pointer jumps through binary conditional nodes; the final model (250 trees, `max_depth = 4`) is small and fast enough for a server scoring every 15 minutes. Training takes about 4 s (regressor) and 11 s (classifier) on a laptop CPU. Per-sample latency was not benchmarked in this project.
  * **Feature Latency Bottleneck:** `peer_dev_campus` needs synchronous readings from the other inverters at the same timestamp. If peers are missing, the current code fills the value with the training median; a real deployment would need an explicit fallback (for example `self_dev_30d`, which uses only the inverter's own history). This fallback is not implemented.

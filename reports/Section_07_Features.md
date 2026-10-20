# Section 7: Feature Engineering, Target Formulation & Feature Selection

> Integration note: the implementation was updated after upstream commit
> `876302c` to preserve the A1/B2 API, retain missing physical measurements,
> use an actual trailing 30-day baseline, and respect protected features during
> pruning. The historical selection tables/plots below describe B1's original
> run and have not been regenerated with these corrections. A2's current
> integration and validation results are documented in
> [A2 preprocessing](A2_PREPROCESSING.md).

**Stage 7 Lead:** Partner B1 (Feature Specialist & XGBoost Track Lead)  
**Contributors:** Partner B2 (Cyclical Time & Station Metadata Features), Pair A (A1 Outage & A2 Rolling Lags)  
**Assignment Reference:** Innovative Assignment Guidelines §2 (Stage 7) & [`Solar_Inverter_Health_Monitoring.md`](file:///Users/hettala/Downloads/ML%20ass/Solar_Inverter_Health_Monitoring.md)

---

## 7.1 Overview & Pipeline Context

Feature engineering is the critical bridge that translates high-frequency ($15$-minute) raw electrical telemetry and meteorological readings into physically meaningful predictors of solar photovoltaic (PV) degradation. In urban rooftop solar arrays, such as the 37 rooftop stations across the HKUST campus, raw generation fluctuates violently due to passing clouds, changing seasons, and differing building geometries. 

As the **Pipeline Lead for Stage 7**, Partner B1 has designed, implemented, and validated:
1. The **canonical mathematical equations for the continuous Health Score and 3-class operational status**, establishing the unified project standard.
2. The **anti-leakage architecture** preventing circular self-prediction through forward forecasting (Option A) and diagnostic feature purging (Option B).
3. The **empirical threshold sensitivity study**, evaluating the trade-offs of shifting warning and failure boundaries.
4. The **domain-specific physical feature groups**—specifically **Peer-Deviation** (weather noise cancellation) and **Self-Relative Deviation** (long-term string degradation).
5. The **collinearity analysis and automated feature pruning framework**, reducing redundant multi-collinear variables on the training set to ensure tree stability and clean SHAP interpretability.

All reusable code is unified in [`src/targets.py`](file:///Users/hettala/Downloads/ML%20ass/src/targets.py) and [`src/features.py`](file:///Users/hettala/Downloads/ML%20ass/src/features.py).

---

## 7.2 Physical Target Formulation

The primary dataset contains raw telemetry (`ac_power_w`, `irradiance_wm2`, per-phase voltages/currents) but lacks ground-truth maintenance logs or health tags. Consequently, targets are engineered based on standardized international photovoltaic engineering principles (IEC 61724).

### 7.2.1 Continuous Health Score ($0.0 \le \text{HS} \le 100.0$)

Under standard clear-sky conditions, the theoretical expected AC power generation for a PV array is:
$$\text{Expected Power (W)} = \left(\frac{\text{irradiance\_wm2}}{1000.0}\right) \times (\text{capacity\_kwp} \times 1000.0) \times \eta_{\text{derate}}$$

$$\text{Performance Ratio (PR)} = \frac{\text{ac\_power\_w}}{\max(\text{Expected Power (W)}, \, 100.0)}$$

$$\text{Health Score} = \min\left(100.0, \, \max\left(0.0, \, \text{PR} \times 100.0\right)\right)$$

#### Physical Justification of the $0.78$ Derating Factor
The derating factor $\eta_{\text{derate}} = 0.78$ is the standardized benchmark efficiency for commercial grid-tied silicon rooftop systems. It accounts for:
* **Thermal derating ($8\text{--}10\%$):** Crystalline silicon cell operating temperature in subtropical Hong Kong frequently exceeds $45^\circ\text{C}$ (temperature coefficient $\approx -0.4\% / ^\circ\text{C}$).
* **Soiling and atmospheric dust attenuation ($2\text{--}3\%$):** Urban particulate deposition between rainfall events.
* **DC mismatch and wiring losses ($1.5\text{--}2\%$):** Cable voltage drops and manufacturing tolerance variance.
* **Inverter conversion losses ($2\text{--}4\%$):** Peak DC-to-AC conversion efficiency of modern string inverters is $96\text{--}98\%$.

During nighttime or twilight periods when irradiance $G < 100\text{ W/m}^2$, photovoltaic cell current drops below the inverter MPPT operating window. Health evaluation is physically undefined during these periods and is filtered out (`filter_daylight=True`).

### 7.2.2 Discrete Classification Target (3 Operational Classes)

To direct field maintenance queues, the continuous Health Score is mapped into three discrete operational states:
$$\text{Status} = \begin{cases} 
\textbf{Healthy} & \text{if } \text{Health Score} \ge 90.0 \\ 
\textbf{Warning} & \text{if } 70.0 \le \text{Health Score} < 90.0 \\ 
\textbf{Faulty} & \text{if } \text{Health Score} < 70.0 
\end{cases}$$

---

## 7.3 Data Leakage Prevention Engine (§7.2 Compliance)

> **Critical Methodological Vulnerability:** If a model predicts $\text{Health}(t)$ while having access to $\text{ac\_power\_w}(t)$ and $\text{irradiance\_wm2}(t)$ in its feature matrix $X(t)$, the learner trivially solves the linear algebraic formula rather than learning underlying degradation dynamics. This invalidates predictive maintenance.

To guarantee rigorous evaluation, B1 implemented two strictly compliant operational modes in [`prepare_leakage_free_matrices`](file:///Users/hettala/Downloads/ML%20ass/src/targets.py#L218-L260):

1. **Option A — Forecasting (Lead Target):**
   * The target is shifted forward: $\text{Target} = \text{Health}(t + h)$ where $h = 4$ steps ($1$ hour ahead).
   * Feature matrix $X(t)$ uses observations strictly up to time $t$. Current instantaneous power at $t$ is non-leaking because it precedes the future state by one hour.
2. **Option B — Diagnostic Proxy (Feature Purging):**
   * The target is the contemporaneous state $\text{Health}(t)$.
   * Feature matrix $X(t)$ explicitly purges `ac_power_w`, `specific_yield`, and `expected_power_w`. The model must infer inverter state purely from weather, cyclical time, physical station metadata, thermal stress, and relative peer deviation.

---

## 7.4 Empirical Threshold Sensitivity Study

To justify the selection of $90.0$ and $70.0$ thresholds, B1 executed a parameter sweep over `Faulty` boundaries $[55, 85]$ (step 5) and `Warning` boundaries $[85, 90, 95]$ across all daylight intervals.

### Table 7.1: Threshold Sensitivity Grid Sweep Summary (Excerpts from `reports/tables/b1_target_sensitivity.csv`)
| Faulty Threshold | Warning Threshold | % Healthy | % Warning | % Faulty | Faulty Sample Count | Marginal $\Delta \% / \text{unit}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 55.0 | 90.0 | 81.38% | 14.15% | 4.47% | 7,308 | +0.22 |
| 60.0 | 90.0 | 81.38% | 13.04% | 5.58% | 9,123 | +0.26 |
| 65.0 | 90.0 | 81.38% | 11.60% | 7.02% | 11,478 | +0.31 |
| **70.0 (Baseline)** | **90.0 (Baseline)** | **81.38%** | **9.95%** | **8.67%** | **14,175** | **+0.35** |
| 75.0 | 90.0 | 81.38% | 8.04% | 10.58% | 17,298 | +0.40 |
| 80.0 | 90.0 | 81.38% | 5.86% | 12.76% | 20,862 | +0.46 |
| 85.0 | 90.0 | 81.38% | 3.31% | 15.31% | 25,031 | +0.53 |

*Full results: [`reports/tables/b1_target_sensitivity.csv`](file:///Users/hettala/Downloads/ML%20ass/reports/tables/b1_target_sensitivity.csv).*

### Sensitivity Insights:
* **Baseline Selection ($70 / 90$):** At $< 70.0$, the `Faulty` class represents $8.67\%$ of daylight operations. This captures true persistent underperformance without swamping field teams with false alarms.
* **Marginal Stability:** Setting the threshold below $65.0$ risks missing partial string disconnects, while setting it above $75.0$ causes false positives during rapid passing cloud shadows.
* **Visual Validation:** Figures are stored in [`reports/figures/b1_target_distribution.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_target_distribution.png) and [`reports/figures/b1_threshold_sensitivity_study.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_threshold_sensitivity_study.png).

---

## 7.5 Physical Feature Engineering Groups

Feature transformations in [`src/features.py`](file:///Users/hettala/Downloads/ML%20ass/src/features.py) are partitioned into specialized domains:

### 1. Specific Yield & Performance Ratio (B1)
* `specific_yield` $= (P_{\text{ac}} / 1000) / \text{capacity\_kwp}$ (kW/kWp): Scales output across arrays from $10\text{ kWp}$ to $50\text{ kWp}$.
* `performance_ratio` $= P_{\text{ac}} / P_{\text{expected}}$: Real-time dimensionless operational efficiency.

### 2. Peer-Deviation Features (B1's Signature Group)
* **Campus Peer Deviation (`peer_dev_campus`):**
  $$\text{peer\_dev\_campus}(i, t) = \frac{\text{PR}_i(t) - \text{Median}_{j \in \text{Campus}}(\text{PR}_j(t))}{\text{Median}_{j \in \text{Campus}}(\text{PR}_j(t)) + \epsilon}$$
* **Rooftop Peer Deviation (`peer_dev_station`):** Normalizes an inverter against co-located inverters on the same building.
* **Physical Significance:** Since all rooftops share the same Hong Kong sky, cloud transients simultaneously depress power across all inverters. Deviations from the instantaneous peer median isolate true local electrical failures while cancelling weather transients.

### 3. Self-Relative Deviation (B1's Signature Group)
* **30-Day Irradiance-Conditioned Deviation (`self_dev_30d`):**
  $$\text{self\_dev\_30d}(i, t) = \frac{\text{PR}_i(t) - \text{Baseline}_{i, 30d}(G_{\text{bin}})}{\text{Baseline}_{i, 30d}(G_{\text{bin}}) + \epsilon}$$
* Compares an inverter against its own 30-day historical trailing median under identical solar irradiance brackets ($100\text{--}300$, $300\text{--}600$, $>600\text{ W/m}^2$).
* **Strict Anti-Leakage:** Computed strictly on historical observations up to $t-1$ using shifting windows, preventing future lookahead.

### 4. Thermal & Weather Interactions (B1)
* `thermal_stress` $= (G / 1000.0) \times T_{\text{ambient}}$: Captures combined high-sunlight, high-heat conditions that induce thermal clipping.
* `temp_delta_stc` $= T_{\text{ambient}} - 25.0^\circ\text{C}$: Temperature departure from standard laboratory test conditions.

### 5. Cyclical Time & Station Metadata (B2 Contribution)
* `hour_sin`, `hour_cos`: Fractional hour mapped to a 2D circle ($23:45$ and $00:00$ are continuous).
* `day_sin`, `day_cos`: Fractional day of year capturing annual solar declination.
* `capacity_kwp`, `tilt_deg`, `altitude_m`: Station physical specifications.
* `orient_south`, `orient_east_west`, `orient_mixed`: One-hot encoded roof orientation.

---

## 7.6 Collinearity Pruning & Feature Selection (B1 Leads)

High-dimensional physical feature engineering introduces multi-collinearity. For example, raw yield strongly correlates with solar elevation (`hour_cos`). Collinear features inflate tree variance and scramble SHAP attribution values.

### 7.6.1 Methodology
1. **Split Isolation:** To prevent data leakage, collinearity analysis was conducted exclusively on the training split ($\le 2022\text{-}06\text{-}30$, using B2's `split_chronological`).
2. **Correlation Threshold:** Pairs with absolute Pearson correlation $|r| > 0.85$ were identified.
3. **Domain Pruning Rule:** Domain-interpretable and normalized features (`peer_dev_campus`, `self_dev_30d`, `thermal_stress`, `capacity_kwp`) were protected. Redundant raw proxies were pruned.

### Table 7.2: Pruned Redundant Features (`reports/tables/b1_feature_selection_summary.csv`)
| Dropped Feature | Kept Feature | Correlation ($|r|$) | Physical Decision Rationale |
|:---|:---|:---:|:---|
| `specific_yield` | `hour_cos` | $0.92$ | Redundant with solar zenith angle; replaced by normalized `peer_dev_campus`. |
| `hour_cos` | `thermal_stress` | $0.96$ | Thermal stress captures diurnal solar cycle while incorporating ambient temperature. |
| `temp_delta_stc` | `hour_cos` | $0.94$ | Collinear with diurnal heating; `temp_c` is retained in thermal stress. |
| `performance_ratio` | `peer_dev_campus` | $1.00$ | Raw PR carries weather fluctuations; peer deviation cancels weather noise. |

*Heatmap visualization: [`reports/figures/b1_feature_correlation_matrix.png`](file:///Users/hettala/Downloads/ML%20ass/reports/figures/b1_feature_correlation_matrix.png).*

### 7.6.2 Final Selected Feature Set (13 Features)
Following collinearity pruning, the 13 final predictors are:
1. `hour_sin` (Diurnal cyclical phase)
2. `day_sin` (Annual seasonal phase)
3. `day_cos` (Annual seasonal amplitude)
4. `capacity_kwp` (Array peak DC capacity)
5. `tilt_deg` (Rooftop installation angle)
6. `altitude_m` (Building elevation)
7. `orient_south` (South-facing roof flag)
8. `orient_east_west` (East/West-facing roof flag)
9. `orient_mixed` (Multi-facet / flat roof flag)
10. `peer_dev_campus` (**Primary Signal:** Instantaneous campus peer deviation)
11. `peer_dev_station` (**Local Signal:** Instantaneous rooftop peer deviation)
12. `self_dev_30d` (**Degradation Signal:** Rolling 30-day self-relative deviation)
13. `thermal_stress` (**Thermal Signal:** $G \times T_{\text{ambient}}$ interaction)

This compact, orthogonal 13-feature set supplies maximum physical information with zero target leakage, fully prepared for Stage 9 modeling across XGBoost, MLP, Random Forest, and Ridge.

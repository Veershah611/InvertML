# InvertML: Mathematical Formulae Reference for Feature Engineering & Target Formulation

This document provides a comprehensive, rigorous reference of all mathematical formulas, physical definitions, engineering constants, and data-leakage prevention rules implemented in `InvertML` ([`src/features.py`](src/features.py) and [`src/targets.py`](src/targets.py)).

---

## 1. Physical PV Baseline & Normalization Features

### 1.1 Theoretical Expected AC Power ($P_{\text{expected}}$)
Standardized physical reference power based on the **IEC 61724** benchmark for crystalline silicon PV systems:

$$P_{\text{expected}}(t) = \max\left( \left(\frac{G(t)}{1000.0}\right) \times (\text{Capacity}_{\text{kWp}} \times 1000.0) \times \eta_{\text{derate}}, \, P_{\text{floor}} \right)$$

* **Parameters & Constants:**
  * $G(t) = \text{irradiance\_wm2}$: Plane-of-array/horizontal solar irradiance in $\text{W/m}^2$.
  * $\text{Capacity}_{\text{kWp}}$: Inverter rated peak DC array capacity (in $\text{kWp}$).
  * $\eta_{\text{derate}} = 0.78$: Standard system derating factor (covers $8\text{--}10\%$ thermal losses, $2\text{--}3\%$ soiling/dust, $1.5\text{--}2\%$ DC wiring mismatch, and $2\text{--}4\%$ inverter conversion losses).
  * $P_{\text{floor}} = 100.0\text{ W}$: Lower bound clamp preventing division-by-zero or mathematical singularity during dawn/dusk twilight.

### 1.2 Specific Yield ($Y_{\text{spec}}$)
Standardizes electricity production across heterogeneous array sizes (ranging from $10\text{ kWp}$ to $67\text{ kWp}$ across 37 rooftops):

$$Y_{\text{spec}}(t) = \frac{P_{\text{AC}}(t) / 1000.0}{\text{Capacity}_{\text{kWp}}} \quad \left[\frac{\text{kW AC}}{\text{kWp DC}}\right]$$

### 1.3 Performance Ratio ($\text{PR}$)
Instantaneous dimensionless efficiency metric comparing actual measured power to physical expectation:

$$\text{PR}(t) = \text{clip}\left( \frac{P_{\text{AC}}(t)}{P_{\text{expected}}(t)}, \, 0.0, \, 2.5 \right) \quad \text{for } G(t) \ge 100.0\text{ W/m}^2$$

* *Note:* During night or twilight ($G(t) < 100\text{ W/m}^2$), $\text{PR}$ is physically undefined ($\text{NaN}$) because inverter maximum power point tracking (MPPT) gates shut off.

---

## 2. Relative Deviation Features (Domain Noise Cancellation)

### 2.1 Campus Peer Deviation (`peer_dev_campus`)
* **Physical Insight:** All 37 rooftop stations share the same microclimate and regional sky in Hong Kong. Passing cloud covers temporarily depress generation across all inverters simultaneously. By normalizing against the instantaneous campus median, common-mode weather noise cancels out, isolating true localized inverter hardware defects.

$$\text{peer\_dev\_campus}(i, t) = \text{clip}\left( \frac{\text{PR}_i(t) - \text{Median}_{j \in \text{Campus}}(\text{PR}_j(t))}{\text{Median}_{j \in \text{Campus}}(\text{PR}_j(t)) + \epsilon}, \, -3.0, \, 3.0 \right)$$

* **Constants:**
  * $\epsilon = 0.05$: Prevents division-by-zero during low-light conditions.
  * Clipping bounds $[-3.0, +3.0]$: Stabilizes gradients for neural networks (MLP) and prevents extreme split splits in gradient boosted trees (XGBoost).

### 2.2 Rooftop / Station Peer Deviation (`peer_dev_station`)
Normalizes an inverter against co-located inverters on the *exact same building roof*:

$$\text{peer\_dev\_station}(i, t) = \text{clip}\left( \frac{\text{PR}_i(t) - \text{Median}_{k \in \text{Station}}(\text{PR}_k(t))}{\text{Median}_{k \in \text{Station}}(\text{PR}_k(t)) + \epsilon}, \, -3.0, \, 3.0 \right)$$

### 2.3 30-Day Self-Relative Historical Deviation (`self_dev_30d`)
* **Physical Insight:** Identifies slow, progressive hardware degradation, string disconnection, or persistent local shading by comparing an inverter against its own historical baseline under identical solar irradiance intensity.

$$\text{self\_dev\_30d}(i, t) = \text{clip}\left( \frac{\text{PR}_i(t) - \text{Baseline}_{i, 30d}(G_{\text{bin}})}{\text{Baseline}_{i, 30d}(G_{\text{bin}}) + \epsilon}, \, -3.0, \, 3.0 \right)$$

* **Irradiance Bins ($G_{\text{bin}}$):**
  $$G_{\text{bin}} = \begin{cases} 
  0 \text{ (Low)} & 100 \le G(t) < 300\text{ W/m}^2 \\
  1 \text{ (Medium)} & 300 \le G(t) < 600\text{ W/m}^2 \\
  2 \text{ (High)} & G(t) \ge 600\text{ W/m}^2
  \end{cases}$$
* **Baseline Calculation:**
  $$\text{Baseline}_{i, 30d}(G_{\text{bin}}) = \text{Rolling Median of } \text{PR}_i \text{ over } [t - 30\text{ days}, \, t) \text{ conditioned on } G_{\text{bin}}$$
* **Anti-Leakage Guarantee:** The rolling window is closed strictly on the left (`closed="left"`), completely excluding timestamp $t$ and future values.

---

## 3. Meteorological & Thermal Interaction Features

### 3.1 Thermal Stress Proxy (`thermal_stress`)
Captures the compounding stress of intense solar irradiation combined with elevated ambient temperatures, which triggers inverter thermal throttling and severe silicon voltage drops:

$$\text{thermal\_stress}(t) = \left(\frac{G(t)}{1000.0}\right) \times T_{\text{ambient}}(t)$$

### 3.2 Temperature Delta from Standard Test Conditions (`temp_delta_stc`)
Measures departure from standardized factory laboratory test conditions ($25.0^\circ\text{C}$):

$$\text{temp\_delta\_stc}(t) = T_{\text{ambient}}(t) - 25.0^\circ\text{C}$$

---

## 4. Cyclical Temporal Encodings (2D Circular Trigonometry)

Linear integer hours ($0\text{ to }23$) and days ($1\text{ to }365$) introduce artificial numeric discontinuities ($23 \to 0$ appears 23 steps apart; Dec 31 $\to$ Jan 1 appears 364 days apart). Continuous 2D circular projections map adjacent time intervals onto continuous Euclidean coordinates.

### 4.1 Continuous Diurnal Cycle (Hour of Day)
1. **Fractional Continuous Hour ($t_{\text{hour}}$):**
   $$t_{\text{hour}} = \text{hour} + \frac{\text{minute}}{60.0} \quad \in [0.0, 23.75]$$
2. **Radial Angle:**
   $$\theta_{\text{hour}} = 2\pi \times \left(\frac{t_{\text{hour}}}{24.0}\right)$$
3. **Circular Encodings:**
   $$\text{hour\_sin} = \sin(\theta_{\text{hour}}), \quad \text{hour\_cos} = \cos(\theta_{\text{hour}})$$

### 4.2 Continuous Annual Cycle (Day of Year & Seasonality)
1. **Fractional Continuous Day ($t_{\text{day}}$):**
   $$t_{\text{day}} = \text{day\_of\_year} + \frac{t_{\text{hour}}}{24.0} \quad \in [1.0, 366.0]$$
2. **Radial Angle:**
   $$\theta_{\text{day}} = 2\pi \times \left(\frac{t_{\text{day}}}{365.25}\right)$$
3. **Circular Encodings:**
   $$\text{day\_sin} = \sin(\theta_{\text{day}}), \quad \text{day\_cos} = \cos(\theta_{\text{day}})$$

---

## 5. Physical Station Metadata Features

Extracted from building engineering blueprints and Brick schema metadata:

### 5.1 Rated Capacity (`capacity_kwp`)
Total rated DC wattage of all connected PV panel strings:
$$\text{capacity\_kwp} = \frac{\sum_{j=1}^{M} (\text{panel\_count}_j \times \text{panel\_wattage}_j)}{1000.0}$$

### 5.2 Rooftop Tilt Angle (`tilt_deg`)
Cleaned panel inclination angle (defaults to $15.0^\circ$ campus median if missing or flat):
$$\text{tilt\_deg} = \begin{cases}
\text{parsed slope} & \text{if clean single numeric} \\
\text{first non-zero part} & \text{if multi-slope (e.g., '0/15deg')} \\
15.0^\circ & \text{if missing / unrecorded}
\end{cases}$$

### 5.3 Building Elevation (`altitude_m`)
Elevation of the rooftop installation above sea level in meters ($20.0\text{ m} \le \text{altitude\_m} \le 120.0\text{ m}$).

### 5.4 Rooftop Compass Orientation Encodings
Derived from the azimuth compass heading angle $\phi \in [0^\circ, 360^\circ)$:
* **South-Facing (`orient_south`):** Captures optimal peak solar midday exposure:
  $$\text{orient\_south} = \mathbb{I}(135^\circ \le \phi \le 225^\circ)$$
* **East/West-Facing (`orient_east_west`):** Captures asymmetric morning or evening generation:
  $$\text{orient\_east\_west} = \mathbb{I}((45^\circ \le \phi < 135^\circ) \lor (225^\circ < \phi \le 315^\circ))$$
* **Mixed / Multi-string Rooftop (`orient_mixed`):**
  $$\text{orient\_mixed} = \mathbb{I}(\text{azimuth contains 'mixed'})$$

---

## 6. Outage & Historical Lag / Rolling Window Features (Pair A)

### 6.1 Telemetry Offline Indicator (`is_offline`)
Identifies communication dropout or system shutdown:
$$\text{is\_offline}(t) = \mathbb{I}(P_{\text{AC}}(t) \text{ is NaN} \lor n_{\text{valid\_5min}}(t) = 0)$$

### 6.2 Prior 24-Hour Daylight Outage Duration (`minutes_offline_24h`)
Calculates cumulative offline duration strictly during daylight hours over the preceding 24 clock hours:
$$\text{minutes\_offline\_24h}(t) = 15 \times \sum_{\tau \in [t - 24\text{h}, \, t)} \mathbb{I}(\text{is\_offline}(\tau) = 1 \land G(\tau) \ge 50\text{ W/m}^2)$$

### 6.3 Historical Autoregressive Lags & Rolling Aggregations
For signal $S \in \{\text{performance\_ratio}, \, \text{peer\_dev\_campus}\}$:
* **Discrete Lags:**
  $$S_{\text{lag}}(t) = S(t - \Delta t), \quad \Delta t \in \{15\text{ min}, \, 1\text{ h}, \, 1\text{ day}\}$$
* **Past Rolling Statistics over Window $W \in \{1\text{ h}, \, 4\text{ h}, \, 1\text{ day}\}$:**
  $$\mu_{S, W}(t) = \frac{1}{|W|} \sum_{\tau \in [t - W, \, t)} S(\tau)$$
  $$\sigma_{S, W}(t) = \sqrt{\frac{1}{|W| - 1} \sum_{\tau \in [t - W, \, t)} \left(S(\tau) - \mu_{S, W}(t)\right)^2}$$

---

## 7. Mathematical Target Formulation & Anti-Leakage Protocol

### 7.1 Continuous Health Score Target ($0.0 \le \text{Health Score} \le 100.0$)
$$\text{Health}(t) = \min\left(100.0, \, \max\left(0.0, \, \text{PR}(t) \times 100.0\right)\right)$$

### 7.2 3-Class Discrete Operational Status Target
$$\text{Status}(t) = \begin{cases}
\textbf{Healthy} & \text{if } \text{Health}(t) \ge 90.0 \\
\textbf{Warning} & \text{if } 70.0 \le \text{Health}(t) < 90.0 \\
\textbf{Faulty} & \text{if } \text{Health}(t) < 70.0
\end{cases}$$

### 7.3 Data Leakage Prevention Engine (§7.2 Compliance)
* **Option A — Lead Forecasting Target (Project Benchmark Standard):**
  $$\text{Target} = \text{Health}(t + 1\text{ h}) = \text{Health}(t + 4 \times 15\text{ min})$$
  * The feature matrix $X(t)$ uses observations strictly up to time $t$. Current power $P_{\text{AC}}(t)$ is valid because it precedes future health by one hour.
* **Option B — Diagnostic Contemporaneous Proxy (Feature Purging):**
  $$\text{Target} = \text{Health}(t)$$
  * Explicitly purges $\{P_{\text{AC}}(t), Y_{\text{spec}}(t), P_{\text{expected}}(t)\}$ from $X(t)$ to prevent trivial linear inversion.

---

## 8. Summary of Final 13 Selected Features (Post-Collinearity Pruning)

Pairs with Pearson $|r| > 0.85$ on the training split were pruned to prevent multicollinear variance inflation. The **13 protected features** surviving selection are:

| # | Feature Name | Formula / Encoding | Physical Meaning |
|---|:---|:---|:---|
| 1 | `hour_sin` | $\sin(2\pi \cdot t_{\text{hour}} / 24)$ | Cyclical diurnal time component |
| 2 | `day_sin` | $\sin(2\pi \cdot t_{\text{day}} / 365.25)$ | Cyclical annual seasonal solar cycle |
| 3 | `day_cos` | $\cos(2\pi \cdot t_{\text{day}} / 365.25)$ | Cyclical annual seasonal solar cycle |
| 4 | `capacity_kwp` | $\sum (\text{count} \times \text{wattage}) / 1000$ | Inverter rated peak DC capacity |
| 5 | `tilt_deg` | Cleaned slope angle | Rooftop installation inclination |
| 6 | `altitude_m` | Elevation in meters | Building height above sea level |
| 7 | `orient_south` | $\mathbb{I}(135^\circ \le \phi \le 225^\circ)$ | Flag for south-facing arrays |
| 8 | `orient_east_west` | $\mathbb{I}(\text{east or west azimuth})$ | Flag for morning/evening arrays |
| 9 | `orient_mixed` | $\mathbb{I}(\text{mixed orientation string})$ | Flag for multi-tilt/flat roofs |
| 10 | `peer_dev_campus` | $(\text{PR}_i - \text{Med}_{\text{campus}}) / (\text{Med}_{\text{campus}} + 0.05)$ | Cancels campus-wide weather noise |
| 11 | `peer_dev_station`| $(\text{PR}_i - \text{Med}_{\text{station}}) / (\text{Med}_{\text{station}} + 0.05)$ | Same-rooftop relative deviation |
| 12 | `self_dev_30d` | $(\text{PR}_i - \text{Base}_{30\text{d}}) / (\text{Base}_{30\text{d}} + 0.05)$ | Trailing 30-day degradation tracking |
| 13 | `thermal_stress` | $(G / 1000.0) \times T_{\text{ambient}}$ | Combined sunlight & thermal load |

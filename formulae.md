# Formulae

Every formula used in the code, with the file it lives in. Notation: $i$ = inverter, $t$ = 15-minute time step, $P$ = AC power (W), $G$ = irradiance (W/m²), $C$ = capacity (kWp).

---

## 1. Data preparation (`src/prepare_data.py`)

**Capacity of an inverter** (from the Brick metadata, summed over its panel groups $j$):
$$C_i = \frac{1}{1000}\sum_j \text{panels}_j \times \text{panel watts}_j$$

**Tilt.** "10deg" → 10; two roof parts "0/15deg" → average 7.5; "Mixed" → 5.

**15-minute values** = mean of the 5-minute (power) or 1-minute (weather) readings in the bin. $n_{\text{valid}}$ = number of 5-minute power readings present (0-3).

**Irradiance sensor correction** (sensor change on 2023-02-14). For each month $m$ from February to December:
$$r_m = \frac{\text{median PR in month } m \text{ of 2023 (from 14 Feb)}}{\text{median PR in month } m \text{ of 2022}}, \qquad f = \text{median}(r_m) = 0.570$$
$$G_{\text{corrected}}(t) = f \times G(t) \quad \text{for } t \ge \text{2023-02-14}$$

**Daylight coverage** of an inverter (daylight = $G \ge 50$ W/m²):
$$\text{coverage}_i = 1 - \frac{\text{daylight bins with } n_{\text{valid}} = 0}{\text{all daylight bins}}, \qquad \text{kept if coverage} \ge 0.70$$

## 2. Cleaning (`src/dataset.py: clean`)

| Rule | Action |
|---|---|
| $P < 0$ or $P > 1.1 \times C \times 1000$ | power set to missing (sensor error) |
| $G < 0$ or $G > 1500$ | irradiance set to missing |
| power missing or $n_{\text{valid}} = 0$ | $\text{is\_offline} = 1$ |

## 3. Target (`src/dataset.py`)

**Expected power** (0.78 = derate for temperature, wiring and inverter losses; 100 W floor avoids dividing by almost zero):
$$P_{\text{expected}}(t) = \max\left(\frac{G(t)}{1000} \times C \times 1000 \times 0.78,\ 100\right)$$

**Health score** (defined only when $G \ge 100$ W/m² and power was measured):
$$\text{Health}(t) = \text{clip}\left(100 \times \frac{P(t)}{P_{\text{expected}}(t)},\ 0,\ 100\right)$$

**Status:**
$$\text{Status} = \begin{cases} \text{Healthy} & \text{Health} \ge 90 \\ \text{Warning} & 70 \le \text{Health} < 90 \\ \text{Faulty} & \text{Health} < 70 \end{cases}$$

**What the models predict** (same inverter, exact timestamp; missing if that time is missing):
$$y(t) = \text{Health}(t + 1\,\text{h}), \qquad \text{status\_lead}(t) = \text{Status}(t + 1\,\text{h})$$

## 4. Features (`src/dataset.py: add_features`)

**Time of day and year** (points on a circle, so 23:45 is next to 00:00 and 31 December next to 1 January):
$$h = \text{hour} + \frac{\text{minute}}{60}: \quad \text{hour\_sin} = \sin\frac{2\pi h}{24},\ \text{hour\_cos} = \cos\frac{2\pi h}{24}$$
$$d = \text{day of year}: \quad \text{day\_sin} = \sin\frac{2\pi d}{365.25},\ \text{day\_cos} = \cos\frac{2\pi d}{365.25}$$

**Performance ratio** (output per kWp per 1000 W/m²; defined when $G \ge 100$):
$$\text{PR}_i(t) = \frac{P_i(t)}{G(t) \times C_i}$$

**Peer deviation** (compares an inverter with the whole campus at the same moment; a cloud lowers everyone and cancels out):
$$\text{peer\_dev}_i(t) = \text{clip}\left(\frac{\text{PR}_i(t) - \text{median}_j \text{PR}_j(t)}{\text{median}_j \text{PR}_j(t)},\ -1,\ 2\right)$$

**Self deviation** (compares an inverter with its own recent past; the window excludes $t$):
$$\text{self\_dev\_30d}_i(t) = \text{Health}_i(t) - \text{median}\{\text{Health}_i(\tau) : t - 30\,\text{d} \le \tau < t\}$$

**History** (past only):
$$\text{health\_lag\_15m}(t) = \text{Health}(t - 15\,\text{min}), \qquad \text{health\_lag\_1d}(t) = \text{Health}(t - 1\,\text{d})$$
$$\text{health\_mean\_4h}(t) = \text{mean}\{\text{Health}(\tau) : t - 4\,\text{h} \le \tau < t\}, \qquad \text{peer\_dev\_mean\_4h} \text{ likewise}$$

**Outage history** (daylight bins only, so the 06:00 dawn sleep does not count):
$$\text{minutes\_offline\_24h}(t) = 15 \times \#\{\tau \in [t - 24\,\text{h},\ t) : \text{is\_offline}(\tau) = 1,\ G(\tau) \ge 50\}$$

**Feature selection.** Pearson correlation on the training rows; a pair with $|r| > 0.85$ keeps one feature. Only `tilt_deg` / `multi_orientation` ($r = 0.974$) qualified, so `multi_orientation` is dropped.

## 5. Models

**Baselines** (`src/models/linear.py`):
$$\widehat{\text{Health}}(t + 1\,\text{h}) = \text{Health}(t) \quad \text{(persistence)}$$
$$\widehat{\text{Status}}(t + 1\,\text{h}) = \text{Status}(\text{Health}(t)) \quad \text{(health rule)}, \qquad \text{Status}\big(\text{clip}(100(1 + \text{peer\_dev}), 0, 100)\big) \quad \text{(peer rule)}$$

**Ridge regression** (features standardised: $z = (x - \mu_{\text{train}})/\sigma_{\text{train}}$):
$$\min_\beta \sum_n \big(y_n - \beta_0 - \textstyle\sum_k \beta_k z_{nk}\big)^2 + \alpha \sum_k \beta_k^2$$

**Logistic regression** (three classes, softmax):
$$P(\text{class } c \mid x) = \frac{e^{w_c \cdot z}}{\sum_{c'} e^{w_{c'} \cdot z}}$$

**Balanced sample weights** (Stage 5; $N$ rows, $K = 3$ classes, $N_c$ rows of class $c$):
$$w_c = \frac{N}{K \times N_c}$$
so each class contributes the same total weight to training.

**Random Forest:** average of many decision trees, each trained on a random sample of rows and features. **XGBoost:** a sum of small trees, each fitted to the remaining errors of the previous ones, scaled by the learning rate $\eta$: $\hat y = \sum_k \eta\, f_k(x)$. **MLP:** layers $a^{(l+1)} = \text{ReLU}(W^{(l)} a^{(l)} + b^{(l)})$, trained with Adam and an L2 penalty $\alpha \lVert W \rVert^2$.

**Isolation Forest:** random trees split the data at random values; the anomaly score of a point is based on its average path length $E[h(x)]$ (number of splits needed to isolate it). Short paths mean unusual points. The `contamination` setting is the share of points flagged.

## 6. Metrics (`src/evaluate.py`)

$$\text{RMSE} = \sqrt{\tfrac{1}{N}\textstyle\sum (y - \hat y)^2}, \qquad \text{MAE} = \tfrac{1}{N}\textstyle\sum |y - \hat y|, \qquad R^2 = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$$

$$\text{Faulty recall} = \frac{\text{Faulty rows predicted Faulty}}{\text{all Faulty rows}}, \qquad \text{Faulty precision} = \frac{\text{Faulty rows predicted Faulty}}{\text{all rows predicted Faulty}}$$

$$F_1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}, \qquad \text{macro-}F_1 = \tfrac{1}{3}\left(F_1^{\text{Healthy}} + F_1^{\text{Warning}} + F_1^{\text{Faulty}}\right)$$

**Anomaly detection:** detection rate = flagged fault rows / all fault rows; false-alarm rate = flagged normal rows / all normal rows.

**Permutation importance** of feature $k$: score on validation minus the score after shuffling column $k$ (averaged over 3 shuffles). A large drop means the model relies on that feature.

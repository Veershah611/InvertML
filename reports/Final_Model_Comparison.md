# Final Model Comparison (Stage 11)

All four tracks evaluated on one target and one split protocol, after the 2026-10-21 fixes. Every number comes from `results/results.csv` or from `reports/tables/final_before_after_14feb.csv`, which was computed with each model's final settings. Running each model's entry point in `reports/reproducibility.md` reproduces them.

## 1. Common setup

| Item | Setting |
|---|---|
| Target | Health(t + 1 h), the health score exactly one hour later for the same inverter (§7.2 Option A); status = Healthy >= 90, Warning 70-90, Faulty < 70 |
| Health score | PR x 100 with the 0.78 derate (B1). Irradiance after 2023-02-14 is rescaled by 0.570 to undo the sensor change (`prepare_data.py`). Missing AC power gives an undefined score, not 0. |
| Splits | B2's functions: train <= 2022-06-30, validation 2022 H2, test 2023. Held-out inverters are excluded from train and validation; "unseen" = their 2023 rows. |
| Leakage controls | Imputers and scalers are fitted on training rows only. Rolling and peer features are built on the full grid and look backwards only. No input contains Health(t + 1 h). |
| Selection rule | Choose on validation; confirm once on test. Regression: lowest RMSE. Classification: highest Faulty recall (primary metric, §11.2), then macro-F1. |

Row counts differ slightly between pipelines: A2 applies its own cleaning (test 366,110 rows), while A1, B1 and B2 use 369,090 test rows. The difference is about 1%.

## 2. Regression: Health(t + 1 h)

| Model (owner) | Train RMSE | Val RMSE | Test RMSE | Test before / after 14 Feb | Unseen RMSE |
|---|---|---|---|---|---|
| Persistence (A1 baseline) | 12.11 | 13.04 | 12.30 | 10.89 / 12.44 | 10.73 |
| Ridge (A1) | 10.75 | 11.95 | 11.29 | 11.21 / 11.30 | 9.71 |
| **Random Forest (A2)** | 7.01 | **9.91** | **9.73** | **9.08** / **9.80** | **9.07** |
| XGBoost (B1) | 8.69 | 11.38 | 10.58 | 10.46 / 10.60 | 9.90 |
| MLP (B2) | 8.65 | 13.71 | 11.99 | 12.49 / 11.94 | 14.02 |

* **Every learned model except the MLP beats persistence on validation, and all of them beat it on test.** Before the correction no model did, because half of the 2023 labels were sensor artefacts.
* The gap between before and after 14 Feb is now small for every model (for example RF 9.08 / 9.80), so the correction removed the label shift.
* The MLP is the weakest learned regressor and the only one worse than persistence on validation and on unseen inverters. It trains on a 60k-row sample and has no history or peer features, unlike RF and XGBoost.

**Selected regressor: A2's Random Forest (default configuration).** It has the lowest validation RMSE (9.91), which is confirmed on test (9.73) and on unseen inverters (9.07).

## 3. Classification: status at t + 1 h

| Model (owner) | Val F1 / Faulty recall | Test F1 / Faulty recall | Test before 14 Feb | Test after 14 Feb | Unseen F1 / Faulty recall |
|---|---|---|---|---|---|
| Majority class (A1) | 0.29 / 0.00 | 0.30 / 0.00 | 0.30 / 0.00 | 0.30 / 0.00 | 0.32 / 0.00 |
| Rule: status of Health(t) (A1) | 0.60 / 0.65 | 0.59 / 0.60 | 0.67 / 0.72 | 0.58 / 0.59 | 0.47 / 0.37 |
| Rule: peer deviation (A1) | 0.62 / 0.54 | 0.61 / 0.59 | 0.68 / 0.69 | 0.60 / 0.58 | 0.49 / 0.34 |
| **Logistic Regression (A1)** | 0.54 / **0.72** | 0.53 / **0.70** | 0.63 / 0.78 | 0.52 / 0.69 | 0.35 / 0.39 |
| Random Forest (A2) | **0.65** / 0.62 | **0.61** / 0.57 | **0.70** / 0.77 | **0.60** / 0.55 | **0.50** / 0.33 |
| XGBoost (B1) | 0.60 / 0.65 | 0.56 / 0.64 | 0.67 / **0.82** | 0.55 / 0.62 | 0.41 / **0.42** |
| MLP (B2) | 0.57 / 0.41 | 0.52 / 0.34 | 0.63 / 0.54 | 0.51 / 0.32 | 0.39 / 0.21 |

All classifiers use their Stage 5 winning balancing strategy: class weights (Logistic, RF), sample weights (XGBoost), Healthy undersampling (MLP).

**Selected classifier: A1's Logistic Regression.** By the project's rule it has the highest validation Faulty recall (0.72), which is confirmed on test (0.70). It catches the most faults, at the cost of the lowest macro-F1 among the learned models (0.53 on test), so it raises more false alarms.

* **Trade-off:** if false alarms matter more than missed faults, A2's Random Forest has the best overall balance (test macro-F1 0.61), and XGBoost sits between the two (0.56 / 0.64).
* **The simple rules are hard to beat.** The Health(t) rule and the peer-deviation rule reach macro-F1 close to the best models. The one-hour status is mostly persistent, and only Logistic Regression and XGBoost catch clearly more faults than the rules.
* **Unseen inverters are the weakest split for every model** (Faulty recall 0.21-0.42). Generalising to new rooftops remains the main open problem.

## 4. Anomaly detection (unsupervised)

| Benchmark (owners) | Model | Detection rate | False-alarm rate | Precision / F1 |
|---|---|---|---|---|
| HKUST injected faults (A1, A2) | A2 inputs, A2 settings | 46.7% | 5.3% | 0.030 / 0.057 |
| HKUST injected faults (A1, A2) | A2 inputs, tuned (0.01, 200) | 18.2% | 0.7% | 0.083 / 0.114 |
| HKUST injected faults (A1, A2) | **+ peer deviation, tuned (0.01, 400)** | 31.4% | **0.6%** | **0.156 / 0.209** |
| HKUST injected faults (A1, A2) | + peer deviation, A2 settings | **65.7%** | 3.9% | 0.056 / 0.102 |
| GPVS-Faults, real lab faults (B1, B2) | Isolation Forest, normal-only training | 39.0% | 0% (86 windows) | 1.00 / 0.56 |

* On HKUST, adding peer deviation now improves every setting. Before the irradiance correction, it caused false alarms after 14 Feb (up to 64%), because the benchmark selected "normal" rows with the shifted labels.
* Precision on HKUST stays low because only 137 test rows are injected faults, about 1% of the benchmark rows. The detector is a label-free cross-check, not the main alarm.
* On the real GPVS-Faults data the detector never raised a false alarm on held-out normal windows and caught 39% of fault windows: fault type F5 always, F6 and F7 almost never (`reports/tables/gpvs_iforest_by_scenario.csv`). An earlier result of 68% recall came from synthetic fallback data and has been replaced.

## 5. Final recommendation

1. **Health forecast:** A2's Random Forest (test RMSE 9.73 against persistence 12.30).
2. **Fault alarm:** A1's Logistic Regression for the highest Faulty recall (0.70), or A2's Random Forest where fewer false alarms are preferred.
3. **Anomaly cross-check:** Isolation Forest with peer deviation, tuned on validation (0.6% false alarms).

## 6. Limitations

* **Engineered labels.** No maintenance logs exist, so "Faulty" means health below 70, not a confirmed fault.
* **Sensor correction uses test-year data.** The 0.570 factor is estimated from 2023 against 2022. No model input or training row depends on it, but the test labels do. One factor over-corrects late 2023 slightly (corrected output per unit irradiance about 1.05-1.1 in Oct-Dec).
* **One weather station** serves 37 rooftops, so shading and orientation differences show up as noise.
* **Unseen inverters:** only 6 held-out inverters have data, so unseen scores are noisy.
* **Zone D capacities** are inconsistent with the data (A1 EDA) and are not corrected.
* **Hardest inverters:** the highest Ridge test errors are S H Ho Sports Hall Inverter 2 (RMSE 20.7), UG Hall2 2F (19.6) and S H Ho Sports Hall Inverter 1 (19.2); their scatters show a second low branch (partial shading or mixed orientation, A1 EDA).
* **MLP:** trained on a 60k-row sample without history features, so it is not a like-for-like comparison with RF and XGBoost.

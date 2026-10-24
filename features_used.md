# Features Used by Each Model

All supervised models (Ridge, Logistic Regression, Random Forest, XGBoost, MLP) use **the same 20 features** from `src/dataset.py` (`FEATURES`), built from the same table, so their scores are directly comparable. Every feature is known at time $t$; the target is the health score one hour later. Formulas are in [formulae.md](formulae.md).

## 1. The 20 shared features

| # | Feature | Group (owner) | Meaning |
|---|---|---|---|
| 1 | `health_now` | current state (A1) | health score now (0-100) |
| 2 | `n_valid_5min` | current state (A1) | how many of the three 5-minute power readings exist (1-3) |
| 3 | `minutes_offline_24h` | outages (A1) | minutes offline in daylight during the past 24 hours |
| 4 | `irradiance_wm2` | weather | sunlight now (corrected after 2023-02-14) |
| 5 | `temp_c` | weather | air temperature |
| 6 | `rh_pct` | weather | relative humidity |
| 7 | `wind_speed_ms` | weather | wind speed |
| 8 | `hour_sin` | time (B2) | time of day on a circle |
| 9 | `hour_cos` | time (B2) | time of day on a circle |
| 10 | `day_sin` | time (B2) | day of year on a circle |
| 11 | `day_cos` | time (B2) | day of year on a circle |
| 12 | `capacity_kwp` | metadata (B2) | installed panel power from the Brick file |
| 13 | `tilt_deg` | metadata (B2) | panel tilt |
| 14 | `altitude_m` | metadata (B2) | roof height above sea level |
| 15 | `peer_dev` | physics (B1) | this inverter vs the campus median at the same moment |
| 16 | `self_dev_30d` | physics (B1) | health now vs own median health over the past 30 days |
| 17 | `health_lag_15m` | history (A2) | health 15 minutes ago |
| 18 | `health_lag_1d` | history (A2) | health at the same time yesterday |
| 19 | `health_mean_4h` | history (A2) | average health over the past 4 hours |
| 20 | `peer_dev_mean_4h` | history (A2) | average peer deviation over the past 4 hours |

**Dropped in feature selection:** `multi_orientation` (roof facing several directions), correlation $r = 0.974$ with `tilt_deg`.

**Not used as features on purpose:**
- `ac_power_w`, the raw irradiance and the capacity *together* at the same time would let a model rebuild Health(t) by formula; the target is one hour later, and `health_now` already carries that information.
- `inverter_id`: the models must work on unseen inverters.
- `is_offline`: every training row has a valid health score, so it would always be 0; outages enter through `minutes_offline_24h`.

## 2. How each model uses them

| Model | Owner | Features | Preparation | Training |
|---|---|---|---|---|
| Persistence | A1 | `health_now` only | none | none (rule) |
| Majority class | A1 | none | none | most common training class ("Healthy") |
| Health rule | A1 | `health_now` only | none | none (rule) |
| Peer rule | A1 | `peer_dev` only | none | none (rule) |
| Ridge / Logistic Regression | A1 | all 20 | median fill + standard scaling | balanced weights (Logistic) |
| Random Forest | A2 | all 20 | median fill | balanced weights (classifier) |
| XGBoost | B1 | all 20 | median fill | balanced weights (classifier) |
| MLP | B2 | all 20 | median fill + standard scaling | balanced weights (classifier) |

Median fill and scaling are fitted on the training period only, inside each model's pipeline.

**Most important features** (permutation importance on validation): `health_now` for every model; `peer_dev` for Random Forest and XGBoost; `self_dev_30d` for Ridge; time of day for the MLP. Full tables: `reports/tables/<model>_<task>_importance.csv`.

## 3. Anomaly detection inputs (`src/models/iforest.py`)

| Benchmark | Owners | Inputs |
|---|---|---|
| HKUST with injected faults | A1, A2 | `health_now`, `peer_dev`, `self_dev_30d`, `irradiance_wm2` |
| GPVS-Faults (real lab faults) | B1, B2 | 18 statistics per 0.1-second window: mean, std, min, max of `Ipv`, `Vpv`, `Vdc` (DC current and voltages), and RMS of `ia`, `ib`, `ic`, `va`, `vb`, `vc` (AC phase currents and voltages) |

## 4. Leakage rules

- The target is matched on the exact timestamp one hour later for the same inverter; if that time is missing, the row is not used (no jumping over gaps or nights).
- All history windows exclude the current row and use only the past.
- Fill values and scalers are learned from training rows only.
- The six unseen inverters never appear in training or validation.

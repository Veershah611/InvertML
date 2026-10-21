# A1 Work Log

## Done

**Repo setup** (Done)
- Folder layout from §13.6: `src/`, `configs/`, `notebooks/`, `results/`, `reports/figures/`, `data/` (git-ignored).
- `configs/config.yaml`: paths, seed 42, timezone, 15-min frequency, 70% coverage threshold, split dates.
- Python 3.12 `.venv`; `requirements.txt` pins the 9 top-level packages.

**Stage 2: Data understanding** (Done)
- Inspected the dataset README and every file. Column table and findings written into §2.1 of the assignment doc.
- **Scope decision:** inverter-level data only, meaning 44 inverters at 37 stations, resampled 5-min → 15-min. The 23 stations without optimizers have only hourly site-level data and are excluded.
- Missing from the dataset: DC current/power and module temperature. The conditional features in §7.1 are therefore not possible.
- Inverter files hold 4,766,698 rows (633 MB), about 1.6M after resampling.
- Data issues found:
  - start dates vary (6 inverters start only in 2023)
  - two timestamp formats
  - rows are daytime only
  - `Rainfall_2021` is an `.xlsx` file
  - the package contains no panel-level files
- Updated every "60 stations" mention in the doc to match the scope decision.

**Stage 3: Metadata step** (Done) ([src/prepare_data.py](src/prepare_data.py))
- Parses the Brick `.ttl` with rdflib and SPARQL, walking Station → Inverter → Panel group.
- Fixes a syntax error in the raw `.ttl` (Zone D block, `.` → `;`) in memory only.
- `capacity_kwp` = Σ(panel count × panel W) / 1000 per inverter (DC size; this is what PR needs). The station's rated kW is kept as a separate column.
- Matches all 44 CSV files to their Brick inverter, and stops with an error if any file is unmatched.
- Output: `data/processed/inverter_metadata.parquet` (44 rows, 19 columns).
- Run with: `.\.venv\Scripts\python.exe -m src.prepare_data` (the whole script takes ~50 s)

**Stage 3: CSV → Parquet conversion** (Done) (`src/prepare_data.py`, step 2)
- Timestamps are parsed with the two known formats tried explicitly; anything else raises an error. They are kept as naive HKT (Hong Kong has no daylight saving).
- `inverter_5min.parquet`: 4,766,698 rows. Columns are `inverter_id`, `time` and the 17 measurements as float32, renamed to snake_case with units (e.g. `ac_power_w`, `dc_voltage_v`, `l1_ac_current_a`). `NA` is stored as NaN. 201 MB on disk, 367 MB in memory.
- `weather_1min.parquet`: 1,567,792 rows (2021-01-01 → 2023-12-31). The 7 variables become 8 float32 columns, joined on time. NaNs (at most 617 per column) come only from minutes missing in one variable's file.
- `Rainfall_2021.xlsx` is read with `openpyxl` (added to requirements.txt).
- Checks: inverter IDs exactly match the metadata table; 0 duplicate (inverter, time) rows; no duplicate weather timestamps.
- Note: the raw files label reactive power in W; strictly the unit is var.

**Stage 3: 15-min resampling + merge** (Done) (`src/prepare_data.py`, step 3)
- Bins are labelled by their start (10:00 covers 10:00–10:14), the same for inverter and weather data.
- Inverters, 5 → 15 min: mean of the available readings. `n_valid_5min` (0–3) counts the valid `ac_power_w` readings in each bin. 4,766,698 → **1,588,901 rows**.
- Weather, 1 → 15 min: mean for readings, **sum** for rainfall, **circular mean** for wind direction. `n_valid_1min` (0–15) counts valid irradiance readings. 1,567,792 → 104,523 rows.
- Merge: weather joined on `time`, metadata on `inverter_id` (both checked many-to-one).
- Output: **`hkust_15min_v1.parquet`**, 1,588,901 rows × 45 columns, 101 MB (267 MB in memory). The full script runs in ~75 s.
- Hand-checked: one SQ1 bin (2022-07-01 12:00) matches the mean of the raw 5-min power and the 1-min irradiance exactly.
**Stage 3: Gap analysis + coverage filter** (Done) (`src/prepare_data.py`, step 4)
- The loggers only record **06:00–17:59**, but in June irradiance stays above 50 W/m² until ~19:00. **Summer evening generation is never recorded** (a dataset limitation for the report).
- Gaps appear as rows full of `NA`, not as missing rows. Rebuilding the full 06:00–17:45 grid per inverter (from its first bin to the end of 2023) added **0 rows**, so the raw grid is already complete.
- Missing bin = `n_valid_5min == 0`. The 06:00 hour is 28% missing (inverters asleep before sunrise); other hours are about 4%. That's why coverage counts **daylight bins only**: irradiance ≥ 50 W/m² (`daylight_irradiance_wm2` in config). Bins without weather data are excluded.
- Coverage is measured from each inverter's own start date. Min **88.8%** (SQ Block P), median 96.6%. **No inverter falls below 70%, so none are dropped.**
- Worst months: LSK South Sep 2023 (97% missing) and Oct 2023 (91%), Zone J2 Aug 2022 (92%), S H Ho Sports Hall Oct 2023 (86%), Library Dec 2023 (72%). These look like real outages and are relevant for `is_offline`.
- Outputs in `reports/tables/`: `gaps_by_inverter_month.csv`, `coverage_by_inverter.csv`, and `prep_summary.csv` (the before/after table: 4,766,698 × 19 → 1,588,901 × 20 → 1,588,901 × 45, 44 inverters throughout).
- For A2 (Stage 4): 94,441 bins have `n_valid_5min = 0` (a row exists but all power values are NA). 21k bins have 1–2 valid readings. 12,172 rows have no weather data (`n_valid_1min = 0`).

**Stage 6: EDA, Q1 power vs irradiance + Q2 gaps** (Done) ([src/eda.py](src/eda.py), [notebooks/01_eda_A1.ipynb](notebooks/01_eda_A1.ipynb))
- Code is in `src/eda.py`; the notebook is a thin driver with a finding and a decision under every plot. It runs top to bottom in about 1 minute.
- 7 figures in `reports/figures/a1_q*.png`.
- **Q1 findings:**
  - Most inverters have a slope of 0.6-0.8 kW/kWp per 1000 W/m².
  - Inverter clipping is visible at about 0.75-0.8 kW/kWp.
  - 99% of bins produce power from **50 W/m²** upward (about 75% at 20 W/m²), which supports a 50 W/m² night threshold.
- **Irradiance sensor change on 2023-02-14:**
  - Output per unit irradiance drops from about 0.9 to about 0.5 for all inverters at once.
  - Peak irradiance jumps from about 840 to about 1,400 W/m², with a 3,609 W/m² spike that day.
  - Raw irradiance is therefore not on the same scale in the 2023 test year. The team must decide: rescale, use peer deviation, or add a period flag.
- **Zone D capacities are wrong:**
  - Inverter 2 reaches 50.1 kW on a 25 kW inverter (slope 1.31).
  - Inverter 1 has slope 0.32.
- Indoor Sports Centre's capacity is consistent with its data; the "10 kW" inverter model in the metadata is the error.
- **Q2 findings:**
  - 78.5% of gaps are whole days (4-12 daylight hours) and 5.7% are multi-day, together holding 99% of missing data.
  - Gaps of up to 1 hour are 0.4% of missing data.
  - The 06:00 hour is 28% missing (inverters asleep); other hours are 3.8%.
  - Several outages hit all inverters at a station at once (Library, S H Ho, Zone J1 + A7), which points to communication failures.

**Stage 8: Baselines + Track T1 linear models** (Done, branch `a1/baselines`) ([src/models/linear.py](src/models/linear.py), [notebooks/03_baselines_A1.ipynb](notebooks/03_baselines_A1.ipynb))
- Target: B1's health score in forecasting mode (Option A): Health(t + 1 h) from data at t. The lead is taken on the full grid and kept only if exactly 1 hour later, never overnight. Rows need Health defined at both t and t+1h (B1's 100 W/m² rule).
- Rows: train 270,448 / val 190,158 / test 464,676 (35,860 before and 428,816 after 2023-02-14) / unseen inverters 46,980. Unseen inverters are excluded from train and val.
- Features (16): `health_now`, `n_valid_5min`, irradiance, temperature, RH, wind speed, and B2's time and metadata features. Mean-imputation and StandardScaler are fitted on train only.
- Tuning: GridSearchCV over a 5-fold TimeSeriesSplit. Ridge alpha = 1000 (an interior optimum after widening the grid). Logistic macro-F1 is flat for C >= 1.
- Results:

  | Model | Val | Test | Test before / after 14 Feb | Unseen |
  |---|---|---|---|---|
  | Persistence (RMSE) | 12.92 | 16.56 | 10.88 / 16.95 | 17.14 |
  | Ridge (RMSE) | 11.88 | 17.38 | 11.22 / 17.79 | 15.99 |
  | Majority (macro-F1) | 0.29 | 0.10 | 0.30 / 0.07 | 0.10 |
  | Rule: status of Health(t) (macro-F1) | 0.63 | 0.65 | 0.69 / 0.58 | 0.54 |
  | Logistic (macro-F1) | 0.62 | 0.55 | 0.68 / 0.49 | 0.42 |

- Findings:
  - Ridge beats persistence on train and val but not on test.
  - Every model holds up until 2023-02-14 and degrades after it, because B1's labels inherit the irradiance sensor change (52% of post-change rows are labelled Faulty, against 7% in train).
  - The rule baseline is the strongest classifier.
  - Ridge is dominated by `health_now` (13.6) and `n_valid_5min` (6.2).
- Default vs tuned: almost no difference (Ridge val RMSE 11.86 default vs 11.88 tuned; Logistic identical). A linear model on 270k rows has little variance to regularise.
- Permutation importance (validation sample of 20k): `health_now` (+13.7 RMSE for Ridge, -0.27 macro-F1 for Logistic), then `n_valid_5min` (+3.2 RMSE). Every other feature is below 0.13 RMSE.
- Failure cases: all 6 worst Ridge errors are **outage onsets** (health 100 at t, 0 at t+1h). They include Zone J2 2022-08-03 (its 36-day outage) and Zone A7 2023-05-04, both also found in the EDA. They are unpredictable from data at t. Health is 0 there because B1's target fills missing power with 0.
- Outputs:
  - `reports/tables/a1_baseline_results.csv`, `a1_ridge_coefficients.csv`, `a1_logistic_coefficients.csv`
  - `a1_default_vs_tuned.csv`, `a1_permutation_importance.csv`, `a1_failure_cases.csv`
  - `reports/figures/a1_baselines_by_split.png`
  - 5 rows in `results/results.csv`
- Re-run with `python -m src.models.linear` (add `--log` to append to `results.csv`) once B1 fixes the target.

**Stage 2: Licence check** (Done)
- Confirmed on the Dryad page: public domain (CC0). The paper is published: Scientific Data (2025), doi:10.1038/s41597-025-04397-y. The dataset README's "N/A" and "under review" are outdated. Recorded in §2.1 of the design doc and in the README.

**Stage 3: Tilt fix** (Done, branch `a1/tilt-fix`) ([src/prepare_data.py](src/prepare_data.py))
- Two-value tilts become their midpoint: UG Hall4 `"0/15deg"` → 7.5, UG Hall7 `"0/11deg"` → 5.5.
- Tilts recorded only as `"Mixed"` become 5.0, the midpoint of the campus 0-10° range. These are 11 inverters without data files, so the processed data is not affected today.
- New column `tilt_mixed` (1 for all of these). `hkust_15min_v1.parquet` is now 1,588,901 × 46 with 0 NaN tilts.
- Before this, `clean_tilt()` in `features.py` turned the NaN tilts into 15.

**Stage 7: Outage features** (Done, branch `a1/outage-features`) ([src/features.py](src/features.py) `add_a1_outage_features`, [tests/test_a1_outage.py](tests/test_a1_outage.py))
- `is_offline`: A2's rule (AC power missing or `n_valid_5min == 0`).
- `minutes_offline_24h`: 15 × offline bins in [t − 24 h, t), excluding t, counting only bins with irradiance ≥ 50 W/m² (`daylight_irradiance_wm2`).
- The daylight rule is needed because inverters asleep at dawn are offline by A2's rule. Without it, 43% of train rows showed offline minutes; with it, 6.5%.
- Must run on the full grid before any filtering or split.
- Validation effect when added to the 16 linear features:

  | Features | Ridge RMSE | Ridge MAE | Logistic macro-F1 | Faulty recall |
  |---|---|---|---|---|
  | Baseline 16 | 11.875 | 6.684 | 0.6228 | 0.7300 |
  | + outage | 11.858 | 6.666 | 0.6225 | 0.7304 |

- The gain is negligible: `health_now` already scores offline bins as 0. The features stay available for the tree models.

**Stage 5: Balancing for Logistic Regression** (Done, branch `a1/balancing`) ([src/models/linear_balancing.py](src/models/linear_balancing.py))
- Compared `class_weight="balanced"` against Healthy undersampling (A2's rule: 2 Healthy rows per non-Healthy row, train only).
- C was re-tuned for each strategy over a 5-fold TimeSeriesSplit, with undersampling applied inside each training fold only. SMOTE was rejected because it would invent readings between rows from different days and inverters.

  | Strategy | C | Train rows | Val macro-F1 | Faulty recall | Faulty precision | Faulty PR-AUC |
  |---|---|---|---|---|---|---|
  | **class_weight (selected)** | 1000 | 270,448 | **0.623** | **0.730** | 0.687 | 0.795 |
  | undersampling | 10 | 143,502 | 0.589 | 0.730 | 0.712 | 0.801 |

- Faulty recall is tied, and class weights win on macro-F1. Undersampling loses most Warning predictions (1,630 correct against 8,508).
- Regression (not balanced): the share of Faulty Health(t+1h) labels is 9.8% in train, 12.8% in val and 47.9% in test (the sensor change).
- Ridge validation error by score range:
  - Faulty: MAE 20.1, mean error +19.5
  - Warning: MAE 10.8
  - Healthy: MAE 3.9, mean error −3.4
  - Ridge shrinks predictions toward the mean, so it over-predicts low scores.
- Outputs: `a1_balancing.csv`, `a1_balancing_cv.csv`, `a1_target_distribution.csv`, `a1_ridge_error_by_range.csv`

**Stage 11: Evaluation of Track T1** (Done, branch `a1/evaluation`) ([src/models/linear_evaluation.py](src/models/linear_evaluation.py))
- Uses the same target, features, splits and tuning as `linear.py`.
- Adds the §8 peer-deviation rule: `peer_dev_campus` at t, with cut-offs tuned on train macro-F1 (Warning < −0.25, Faulty < −0.45). Offline bins are predicted Faulty.

  | Model (macro-F1) | Val | Test | Test before / after 14 Feb | Unseen |
  |---|---|---|---|---|
  | Logistic | 0.623 | 0.549 | 0.677 / 0.486 | 0.416 |
  | Rule: status of Health(t) | 0.629 | 0.649 | 0.686 / 0.585 | 0.543 |
  | Rule: peer deviation | **0.648** | 0.233 | **0.702** / 0.196 | 0.207 |

- Faulty PR-AUC on test: Logistic 0.773, Health(t) rule 0.822, peer rule 0.781.
- **Key finding:** the peer rule is the best classifier on val and on the test year before 14 Feb, but collapses after it (Faulty recall 0.18).
  - Peer deviation cancels the irradiance sensor change; B1's labels do not.
  - After 14 Feb, 52% of rows are labelled Faulty while the inverters perform normally compared with their peers.
  - So most post-change "Faulty" labels are caused by the sensor, not by the inverters. **For B1:** this supports fixing the target.
- Per season (test): winter is easiest (Ridge RMSE 13.1); summer is hardest (20.2), with the highest Faulty share (56%).
  - Persistence beats Ridge on test MAE in every season.
- Per inverter (test): the worst Ridge RMSE is Zone D Inverter 1 (23.7), whose test rows are 95% Faulty because of its wrong capacity (see EDA).
  - Others with high error: S H Ho Sports Hall, UG Hall2 2F, Library Inverter 3.
- Outputs:
  - `a1_eval_classifiers.csv` (with confusion matrices), `a1_eval_by_season.csv`, `a1_eval_by_inverter.csv`, `a1_peer_rule.csv`
  - `reports/figures/a1_ridge_residuals.png`

**Isolation Forest on HKUST injected faults** (Done, branch `a1/iforest`, with A2) ([src/models/iforest_a1.py](src/models/iforest_a1.py))
- Reviewed A2's `iforest_a2.py` and reproduced its result exactly: 33.3% detection, 19.4% false alarms, 36 TP / 2,396 FP. It needs `python -m src.preprocess --skip-comparison` first.
- New in A1's version:
  - contamination (0.01-0.1) and n_estimators (100-400) are tuned on B2's faults injected into Jul-Dec 2022 (validation), then 2023 test is scored once
  - a second input set adds `peer_dev_campus`, recomputed after injection against the untouched campus median
  - the false-alarm rate is split at 14 Feb

  | Inputs | Settings | Detection | False alarms | Before / after 14 Feb |
  |---|---|---|---|---|
  | A2's 3 | A2 (0.05, 200) | 33.3% | 19.4% | 23.1% / 15.1% |
  | A2's 3 | tuned (0.01, 200) | 17.6% | **6.7%** | 6.2% / 7.2% |
  | + peer deviation | A2 (0.05, 200) | **56.5%** | 39.2% | 18.2% / 64.0% |
  | + peer deviation | tuned (0.01, 400) | 30.6% | 24.9% | 11.8% / 40.4% |

- Tuning trades detection for far fewer false alarms. Peer deviation raises detection but also false alarms after 14 Feb.
- **Cause of the false alarms after 14 Feb:** the test population keeps only Health ≥ 90 rows.
  - After the sensor change, the rows that still pass are the ones outperforming their peers: mean peer deviation +0.36, against +0.03 in train.
  - Choosing "normal" rows with B1's labels stops working after 14 Feb. This is a limitation of the benchmark design, not of peer deviation.
- Precision is 1-2% in every setting (108 injected rows among about 12k).
- Outputs: `a1_iforest_tuning.csv`, `a1_iforest_test.csv`

**Write-up: Reproducibility section** (Draft) ([reports/reproducibility.md](reports/reproducibility.md))
- Covers environment, data sources and licences, pipeline commands in order with measured run times, seeds and config, leakage controls, results tracking, and caveats.
- Rows for B1's and B2's entry points are marked "to be completed".

## Open issues
- Azimuth is `"Mixed"` for several stations. (Tilt is fixed on `a1/tilt-fix`.)
- The A1 branches are independent. After they are merged, re-run `linear_evaluation.py` with the outage features and the selected balancing, and log the final rows to `results.csv` once.
- Isolation Forest: the Health ≥ 90 "normal" filter is biased after 14 Feb. A label-free normal filter (for example a peer-deviation band) would make the benchmark fairer.
- Zone D capacities are inconsistent with the data (see EDA). Indoor Sports Centre's capacity is fine.
- **For B1:** the health score uses raw irradiance, so the 2023-02-14 sensor change pushes 52% of post-change rows into Faulty. `create_forecasting_target` shifts by row, which can jump overnight on daylight-filtered data. Missing power is filled with 0, so outages score health 0.
- **For B2:**
  - `sq14_inverter` is in the unseen list but is not one of the 44 inverters.
  - Zone A2 Inverters 2 and 3 are held out while Inverter 1 of the same station is in training.
  - The MLP uses `ac_power_w`, `irradiance_wm2` and `capacity_kwp` to predict Health(t), which is computed from those same values (§7.2 leakage), and it scores night rows.
  - `orient_south` and `orient_east_west` are constant 0 for every inverter.

## To do (A1)
- [ ] Share `hkust_15min_v1.parquet` with the team (Drive/OneDrive)
- [ ] Tell the team about the irradiance sensor change and the Zone D capacities (see EDA)
- [x] Features: `is_offline`, minutes offline in the last 24 h (`a1/outage-features`)
- [x] Add the peer-deviation rule (`a1/evaluation`)
- [ ] Re-run baselines once B1 fixes the target
- [x] Stage 5 balancing for Logistic (`a1/balancing`)
- [x] Stage 11 evaluation of Ridge / Logistic (`a1/evaluation`)
- [x] Isolation Forest on HKUST with injected faults (`a1/iforest`); A2 to review
- [ ] Finish the reproducibility section once B1 and B2 add their entry points

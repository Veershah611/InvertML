# Team Progress (by pipeline stage)

Status as of 2026-10-21, after the final re-run on the corrected target. Final results: [reports/Final_Model_Comparison.md](reports/Final_Model_Comparison.md).

| # | Stage | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 0 | Repo setup | Done: skeleton, config.yaml, venv, requirements | Done | Done | Done |
| 1 | Problem definition | Done: §1 of the assignment doc (lead) | — | — | — |
| 2 | Data acquisition | Done: column table, findings, licence (CC0) | — | — | Done: real GPVS-Faults data and loader (`src/evaluate.py`) |
| 3 | Data preparation | Done (lead): metadata, Parquet, 15-min resample, merge, gap analysis, before/after table, tilt fix, irradiance sensor correction (`src/prepare_data.py`) | — | — | — |
| 4 | Preprocessing | — | Done (lead): physical cleaning, B2 splits, B1 feature integration, imputation comparison, train-only encoding/scaling (`src/preprocess.py`, `reports/A2_PREPROCESSING.md`) | — | — |
| 5 | Balancing | Done: class weights vs Healthy undersampling for Logistic (`src/models/linear_balancing.py`) | Done (lead): class weights vs undersampling for RF (`reports/A2_RESULTS.md`) | Done: sample weights vs undersampling for XGBoost (`src/models/xgb.py`) | Done: unbalanced vs undersampling vs thresholded regressor for the MLP (`src/models/mlp_balancing.py`) |
| 6 | EDA | Done: power vs. irradiance; gap heatmap; sensor change (`notebooks/01_eda_A1.ipynb`) | Done: temperature, correlation, season class shift (`reports/A2_EDA.md`) | Done: degradation; underperformers (`notebooks/01_eda_B1.ipynb`) | Done: seasonality; class distribution (`notebooks/01_eda_B2.ipynb`) |
| 7 | Features | Done: outage features (`add_a1_outage_features`) | Done: exact-time lags and past-only rolling features | Done (lead): PR, deviations, target, selection (`src/features.py`, `src/targets.py`) | Done: time + metadata (`src/features.py`) |
| 8 | Baselines | Done (lead): persistence, Ridge, majority, Health(t) rule, peer-deviation rule, Logistic (`src/models/linear.py`) | — | — | — |
| 9 | Model | Done: Ridge + Logistic | Done: Random Forest (`src/models/rf.py`) | Done: XGBoost, leakage fixed (`src/models/xgb.py`) | Done: MLP, sampling and target fixed (`src/models/mlp.py`) |
| 10 | Tuning | Done: GridSearch + TimeSeriesSplit, default vs tuned | Done: RandomizedSearch with expanding time folds | Done: validation grid search, default vs tuned | Done: GridSearchCV with TimeSeriesSplit, default vs tuned |
| 11 | Evaluation | Done: test, unseen, before/after 14 Feb, per season and inverter (`src/models/linear_evaluation.py`) | Done: baselines on same rows, unseen test, sensor-change and score-range checks | Done: test, unseen, Tree-SHAP, failure cases | Done (lead): splits, `evaluate.py`, fault injection, logger; final comparison (`reports/Final_Model_Comparison.md`) |
| — | Isolation Forest | Done: HKUST injected faults, tuned, peer-deviation input (`src/models/iforest_a1.py`) | Done: HKUST benchmark (`src/models/iforest_a2.py`) | Done: GPVS-Faults on real data (`src/models/iforest.py`) | Done: GPVS loader and evaluation |
| 12 | Impact & limitations | — | — | — | Done (lead): `reports/Section_14_Impact.md` |
| — | Write-up | Done: reproducibility (`reports/reproducibility.md`), final comparison | Done: preprocessing, EDA, leakage audit, results | Done: §7, §9-11, viva prep | Done: impact, AI-usage log |

## Final results (2023 test year)

| Task | Selected model | Test score | Persistence / rule baseline |
|---|---|---|---|
| Health forecast, Health(t + 1 h) | A2 Random Forest | RMSE 9.73 | 12.30 |
| Fault alarm, status at t + 1 h | A1 Logistic Regression | Faulty recall 0.70, macro-F1 0.53 | Health(t) rule: recall 0.60, macro-F1 0.59 |
| Anomaly cross-check (HKUST injected faults) | Isolation Forest + peer deviation, tuned | detection 31.4%, false alarms 0.6% | A2 settings: 46.7% / 5.3% |
| Real-fault check (GPVS-Faults) | Isolation Forest, normal-only training | precision 1.00, recall 0.39, F1 0.56 | — |

## Resolved on 2026-10-21

1. **B1 XGBoost target leakage:** XGBoost now forecasts Health(t + 1 h); its earlier RMSE 0.30 / F1 1.00 came from rebuilding the target from its inputs.
2. **Targets now match:** all four tracks forecast Health(t + 1 h) with an exact-time lead and use B2's splits with held-out inverters excluded from train and validation.
3. **Irradiance sensor change on 2023-02-14:** irradiance after that date is rescaled by 0.570 in `prepare_data.py`; the post-change Faulty share fell from 54% to 9.5%.
4. **Outages scored as health 0:** missing power now gives an undefined health score.
5. **B2 MLP sampling:** it trained on 7 of 36 inverters (first rows of a file sorted by inverter); it now uses a random sample of all inverters.
6. **GPVS-Faults result came from mock data:** the loader generated synthetic data when the real files were missing. The real data is now used, and the loader raises an error instead of falling back.
7. **`results/results.csv`:** one row per final model, all in the shared format.
8. **Panel tilt NaNs:** midpoint for two-value tilts, 5.0 for "Mixed", plus a `tilt_mixed` flag.
9. **Tests:** `python -m unittest discover -s tests` runs all 20 tests.

## Known limitations (documented, not fixable with this data)

- Health labels are engineered from PR; there are no maintenance logs.
- The irradiance correction factor is estimated from 2023 against 2022 (test-year data) and is one factor for the whole post-change period.
- Zone D capacities are inconsistent with the data; `azimuth` is "Mixed" for several stations; `orient_south` and `orient_east_west` are constant.
- Only 6 held-out inverters have data (`sq14_inverter` is not one of the 44).
- B1's XGBoost builds features on the full grid, so held-out inverters contribute to campus peer medians (A2 audit, item 4).
- GPVS false-alarm rate is measured on only 86 held-out normal windows.

## Remaining tasks (people, not code)

These need team members and cannot be done in the repository:

- [ ] **A1:** share `data/processed/hkust_15min_v1.parquet` (47 columns, rebuilt with the irradiance correction) on the team drive, and announce the rebuild.
- [ ] **A1:** tell the team about the irradiance correction and the Zone D capacities.
- [ ] **A2:** review A1's Isolation Forest extension (`src/models/iforest_a1.py`) and the RF numbers in `reports/A2_RESULTS.md`.
- [ ] **B1:** review the XGBoost leakage fix, the GPVS rewrite and the updated `Section_09_10_XGBoost.md` / `B1_VIVA_PREP.md`.
- [ ] **B2:** review the MLP and balancing fixes and the real GPVS loader.
- [ ] **Everyone:** run `pip install -r requirements.txt` (adds shap, xgboost, statsmodels if missing) and download GPVS-Faults (see README).
- [ ] **Everyone:** assemble the final report from the section files, then rehearse the viva (each member explains their partner's model).

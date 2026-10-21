# Team Progress (by pipeline stage)

| # | Stage | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 0 | Repo setup | Done: skeleton, config.yaml, venv, requirements | To do | To do | To do |
| 1 | Problem definition | Done: §1 of the assignment doc (lead) | — | — | — |
| 2 | Data acquisition | Done: column table, findings, licence (CC0) | — | — | To do: GPVS-Faults loader |
| 3 | Data preparation | Done (lead): metadata, Parquet, 15-min resample, merge, gap analysis, before/after table. To do: share `hkust_15min_v1.parquet` | — | — | — |
| 4 | Preprocessing | — | Implemented: physical cleaning, B2 splits, B1 physical-feature integration, mean/median validation comparison, train-only encoding/scaling (`src/preprocess.py`). Gaps preserved; see `reports/A2_PREPROCESSING.md` for interpolation policy and limitations. | — | — |
| 5 | Balancing | To do | Done: class weights vs Healthy undersampling; class weights selected by validation Faulty recall (`reports/A2_RESULTS.md`) | To do | To do |
| 6 | EDA | Done: power vs. irradiance; gap heatmap (`notebooks/01_eda_A1.ipynb`) | Done: temperature, correlation and season class shift (`reports/A2_EDA.md`) | Done: degradation; underperformers (`notebooks/01_eda_B1.ipynb`) | Done: seasonality; class distribution (`notebooks/01_eda_B2.ipynb`) |
| 7 | Features | To do: outage features | Done: exact-time PR/peer lags and past-only rolling features (`src/features.py`) | Done (lead): PR, deviations, target, selection (`src/features.py`, `src/targets.py`) | Done: time + metadata (`src/features.py`) |
| 8 | Baselines | Done (lead): persistence, Ridge, majority, rule, Logistic (`src/models/linear.py`); re-run after target fix | — | — | — |
| 9 | Model | Done: Ridge + Logistic (same file) | Done: RF regression and classification; selected untuned references by validation criteria (`src/models/rf.py`) | Done: XGBoost (`src/models/xgb.py`, `notebooks/03_xgb_B1.ipynb`) | Done: MLP (`src/models/mlp.py`) |
| 10 | Tuning | Done: GridSearch + TimeSeriesSplit, default vs tuned, permutation importance, failure cases | Done: RandomizedSearch with expanding time folds; SHAP and permutation explanations (`reports/A2_RESULTS.md`) | Done: B2 Splits GridSearch, early stopping, default vs tuned | Done: GridSearchCV (`src/models/mlp.py`) |
| 11 | Evaluation | To do | Done: same-row baselines, unseen inverter test, sensor-change and score-range checks; 2023 generalization weak (`reports/A2_RESULTS.md`) | Done: test 2023, unseen stations, Tree-SHAP, failures (`src/models/xgb.py`) | Done (lead): splits, evaluate.py, fault injection, logger |
| — | Isolation Forest | To do: HKUST injected faults | A2 benchmark done on synthetic faults; A1 joint review pending | Done: GPVS-Faults verification & plot (`src/models/iforest.py`) | Done: GPVS loader & evaluation in `evaluate.py` |
| 12 | Impact & limitations | — | — | — | To do (lead) |
| — | Write-up | Draft: reproducibility (`reports/reproducibility.md`) | Done: A2 preprocessing, EDA, leakage audit and model results (`A2_work.md`, `reports/A2_RESULTS.md`) | Done: §7 (`reports/Section_07_Features.md`), §9-11 (`reports/Section_09_10_XGBoost.md`), Viva prep (`reports/B1_VIVA_PREP.md`) | To do: AI-usage log |

## Remaining work (as of 2026-10-21, after PR #14)

- **A1:** Stage 5 balancing; Stage 7 outage features; Stage 11 evaluation of Ridge/Logistic on test + unseen stations; Isolation Forest on HKUST injected faults; joint review of A2's Isolation Forest benchmark.
- **A2:** Joint Isolation Forest review with A1.
- **B1:** Stage 5 balancing; re-run XGBoost after the target issue below is fixed, then update `reports/Section_09_10_XGBoost.md` and `reports/B1_VIVA_PREP.md`.
- **B2:** Stage 5 balancing; Stage 12 impact & limitations (lead); AI-usage log.
- **Everyone:** Stage 0 shows "To do" for A2/B1/B2 and Stage 2 shows "To do: GPVS-Faults loader" for B2 although later rows say it is done; update these cells.

## Open issues

1. **B1 XGBoost results are most likely target leakage (high priority).** The target `health_score` is `PR(t) * 100` (`src/targets.py`), and the features include `peer_dev_campus`, `peer_dev_station` and `self_dev_30d`, which are built from the same `PR(t)` at the same timestamp (`src/features.py`). The model can rebuild the target from its inputs, which explains test RMSE 0.30 and Macro-F1 1.00 against persistence RMSE 16.56. Fix: predict `Health(t+1h)` like A1/A2, or use only lagged versions of the PR-based features. Do not present XGBoost as the best model until this is re-run.
2. **Targets are not the same across models.** A1 and A2 forecast `Health(t+1h)`; B1 predicts `Health(t)`; B2's MLP target should be checked. Scores in `results/results.csv` are only comparable once everyone uses the same target and split.
3. **`results/results.csv` needs cleaning.** The GPVS Isolation Forest result is logged three times (B2 twice, B1/B2 once); B2 MLP runs are logged several times; A2 rows have bare numbers without the metric name (e.g. `21.243` instead of `RMSE: 21.24`). Keep one final row per model and use the same score format.
4. **A2 classification generalizes poorly:** Macro-F1 0.67 on validation but 0.38-0.46 on the 2023 test set. Needs discussion in Stage 12.
5. **Panel tilt NaNs not fixed on `main`.** The fix (midpoint for `"0/15deg"` / `"0/11deg"`, 5.0 for `"Mixed"`, plus a `tilt_mixed` flag) is only in a local git stash. `tilt_deg` is a Stage 7 feature, so UG_Hall4 and UG_Hall7 currently have NaN tilt.
6. **Test runner:** `pytest` is not installed in `.venv`, and `python -m unittest discover` fails because `tests/` has no `__init__.py`. Run the tests with `python -m unittest tests.test_preprocess tests.test_a2_history tests.test_feature_integration`.

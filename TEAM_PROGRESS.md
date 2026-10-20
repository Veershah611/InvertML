# Team Progress (by pipeline stage)

| # | Stage | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 0 | Repo setup | Done: skeleton, config.yaml, venv, requirements | To do | To do | To do |
| 1 | Problem definition | Done: §1 of the assignment doc (lead) | — | — | — |
| 2 | Data acquisition | Done: column table, findings, licence (CC0) | — | — | To do: GPVS-Faults loader |
| 3 | Data preparation | Done (lead): metadata, Parquet, 15-min resample, merge, gap analysis, before/after table. To do: share `hkust_15min_v1.parquet` | — | — | — |
| 4 | Preprocessing | — | Implemented: physical cleaning, B2 splits, B1 physical-feature integration, mean/median validation comparison, train-only encoding/scaling (`src/preprocess.py`). Gaps preserved; see `reports/A2_PREPROCESSING.md` for interpolation policy and limitations. | — | — |
| 5 | Balancing | To do | To do (lead): imbalance study | To do | To do |
| 6 | EDA | Done: power vs. irradiance; gap heatmap (`notebooks/01_eda_A1.ipynb`) | To do: temperature; correlation | Done: degradation; underperformers (`notebooks/01_eda_B1.ipynb`) | Done: seasonality; class distribution (`notebooks/01_eda_B2.ipynb`) |
| 7 | Features | To do: outage features | To do: lag & rolling | Done (lead): PR, deviations, target, selection (`src/features.py`, `src/targets.py`) | Done: time + metadata (`src/features.py`) |
| 8 | Baselines | Done (lead): persistence, Ridge, majority, rule, Logistic (`src/models/linear.py`); re-run after target fix | — | — | — |
| 9 | Model | Done: Ridge + Logistic (same file) | To do: Random Forest | Done: XGBoost (`src/models/xgb.py`, `notebooks/03_xgb_B1.ipynb`) | Done: MLP (`src/models/mlp.py`) |
| 10 | Tuning | Done: GridSearch + TimeSeriesSplit, default vs tuned, permutation importance, failure cases | To do: RandomizedSearch | Done: B2 Splits GridSearch, early stopping, default vs tuned | Done: GridSearchCV (`src/models/mlp.py`) |
| 11 | Evaluation | To do | To do | Done: test 2023, unseen stations, Tree-SHAP, failures (`src/models/xgb.py`) | Done (lead): splits, evaluate.py, fault injection, logger |
| — | Isolation Forest | To do: HKUST injected faults | To do: HKUST injected faults | To do: GPVS-Faults | In progress: GPVS loader in evaluate.py |
| 12 | Impact & limitations | — | — | — | To do (lead) |
| — | Write-up | Draft: reproducibility (`reports/reproducibility.md`) | Started: preprocessing decisions and leakage checks (`reports/A2_PREPROCESSING.md`); project-wide audit pending | Done: feature/target justification (`reports/Section_07_Features.md`) | To do: AI-usage log |

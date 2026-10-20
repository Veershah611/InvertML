# Team Progress (by pipeline stage)

| # | Stage | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 0 | Repo setup | Done: skeleton, config.yaml, venv, requirements | To do | To do | To do |
| 1 | Problem definition | Done: §1 of the assignment doc (lead) | — | — | — |
| 2 | Data acquisition | In progress: column table and findings done; licence check left | — | — | To do: GPVS-Faults loader |
| 3 | Data preparation | Done (lead): metadata, Parquet, 15-min resample, merge, gap analysis, before/after table. To do: share `hkust_15min_v1.parquet` | — | — | — |
| 4 | Preprocessing | — | To do (lead) | — | — |
| 5 | Balancing | To do | To do (lead): imbalance study | To do | To do |
| 6 | EDA | Done: power vs. irradiance; gap heatmap (`notebooks/01_eda_A1.ipynb`) | To do: temperature; correlation | Done: degradation; underperformers (`notebooks/01_eda_B1.ipynb`) | Done: seasonality; class distribution (`notebooks/01_eda_B2.ipynb`) |
| 7 | Features | To do: outage features | To do: lag & rolling | Done (lead): PR, deviations, target, selection (`src/features.py`, `src/targets.py`) | Done: time + metadata (`src/features.py`) |
| 8 | Baselines | To do (lead): persistence, Ridge, majority, rule, Logistic | — | — | — |
| 9 | Model | To do: linear | To do: Random Forest | To do: XGBoost | Done: MLP (`src/models/mlp.py`) |
| 10 | Tuning | To do: GridSearch | To do: RandomizedSearch | To do: Optuna | Done: GridSearchCV (`src/models/mlp.py`) |
| 11 | Evaluation | To do | To do | To do | Done (lead): splits, evaluate.py, fault injection, logger |
| — | Isolation Forest | To do: HKUST injected faults | To do: HKUST injected faults | To do: GPVS-Faults | In progress: GPVS loader in evaluate.py |
| 12 | Impact & limitations | — | — | — | To do (lead) |
| — | Write-up | To do: reproducibility | To do: leakage audit | Done: feature/target justification (`reports/Section_07_Features.md`) | To do: AI-usage log |

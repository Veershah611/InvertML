# Team Progress (by pipeline stage)

Status: ✅ done · 🔄 in progress · ⬜ not started

| # | Stage | A1 | A2 | B1 | B2 |
|---|---|---|---|---|---|
| 0 | Repo setup | ✅ Skeleton, config.yaml, venv | ⬜ | ⬜ | ⬜ |
| 1 | Problem definition | ⬜ Lead | — | — | — |
| 2 | Data acquisition | 🔄 Download, licence, README column table | — | — | ⬜ GPVS-Faults loader |
| 3 | Data preparation | ⬜ Lead: metadata, Parquet, resample, merge, gaps | — | — | — |
| 4 | Preprocessing | — | ⬜ Lead | — | — |
| 5 | Balancing | ⬜ | ⬜ Lead: imbalance study | ⬜ | ⬜ |
| 6 | EDA | ⬜ Power vs. irradiance; gap heatmap | ⬜ Temperature; correlation | ⬜ Degradation; underperformers | ⬜ Seasonality; class distribution |
| 7 | Features | ⬜ Outage features | ⬜ Lag & rolling | ⬜ Lead: PR, deviations, target | ⬜ Time + metadata |
| 8 | Baselines | ⬜ Lead: persistence, Ridge, majority, rule, Logistic | — | — | — |
| 9 | Model | ⬜ Linear | ⬜ Random Forest | ⬜ XGBoost | ⬜ MLP |
| 10 | Tuning | ⬜ GridSearch | ⬜ RandomizedSearch | ⬜ Optuna | ⬜ RandomizedSearch |
| 11 | Evaluation | ⬜ | ⬜ | ⬜ | ⬜ Lead: splits, evaluate.py |
| — | Isolation Forest | ⬜ HKUST injected faults | ⬜ HKUST injected faults | ⬜ GPVS-Faults | ⬜ GPVS-Faults |
| 12 | Impact & limitations | — | — | — | ⬜ Lead |
| — | Write-up | ⬜ Reproducibility | ⬜ Leakage audit | ⬜ Feature/target justification | ⬜ AI-usage log |

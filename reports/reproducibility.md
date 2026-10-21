# Reproducibility

Every number in the report can be regenerated from the raw datasets with the steps below, in order. The final results are collected in [Final_Model_Comparison.md](Final_Model_Comparison.md).

## 1. Environment

- **Python 3.12**, Windows 11, CPU only (no GPU needed).
- Virtual environment plus pinned packages:
  ```
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- `requirements.txt` pins every top-level package, including shap 0.54.1, statsmodels 0.15.0 and xgboost 3.4.1. Run `pip install -r requirements.txt` again after pulling; a `.venv` created earlier may be missing these three.
- **Jupyter kernel:** notebooks must run on the `.venv` interpreter. The default `python3` kernel can point at a system Python without `pyarrow`, and every notebook then fails while loading data. Register the venv once with `python -m ipykernel install --user --name invertml` and pick that kernel, or run `.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=invertml notebooks\<name>.ipynb`.

## 2. Data

| Dataset | Source | Licence | Where it goes |
|---|---|---|---|
| HKUST Rooftop PV 2021-2023 | Dryad, https://doi.org/10.5061/dryad.m37pvmd99 (version of 8 Sep 2024, `Dataset.zip`, 296 MB) | Public domain (CC0), per the Dryad page | unzip into `data/raw/` |
| GPVS-Faults | Mendeley Data, https://data.mendeley.com/datasets/n76t439f65/1, DOI 10.17632/n76t439f65.1 (`CSV_Files.zip`, 115 MB) | CC BY 4.0 | extract the 16 `F*.csv` files into `data/raw/GPVS-Faults/` |

**Citations**
- HKUST dataset: Lin Z., Zhou Q., Wang Z., Wang C., Bookhart D. B., Leung-Shea M. (2024). *A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics* [Dataset]. Dryad. https://doi.org/10.5061/dryad.m37pvmd99
- HKUST paper: Lin Z. et al. (2025). Scientific Data. https://doi.org/10.1038/s41597-025-04397-y
- GPVS-Faults: Bakdi A., Guichi A., Mekhilef S., Bounoua W. (2020). *GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes*. Mendeley Data, V1. https://doi.org/10.17632/n76t439f65.1

The raw files are never modified. A known syntax error in the metadata `.ttl` file (Zone D block) is corrected in memory by `src/prepare_data.py`. `data/` is git-ignored. The GPVS loader raises an error if its data is missing; it never substitutes synthetic data.

## 3. Pipeline, in order

Run from the repo root with the venv active. Times were measured on an 18-thread laptop CPU, one step at a time (do not run several tuning steps in parallel: on Windows the parallel joblib pools can crash with "The handle is invalid").

| # | Step (owner) | Command | Main outputs | Time |
|---|---|---|---|---|
| 1 | Data preparation (A1) | `python -m src.prepare_data` | `data/processed/*.parquet`, `reports/tables/{prep_summary, coverage_by_inverter, gaps_by_inverter_month, irradiance_correction}.csv` | about 2 min |
| 2 | Preprocessing (A2) | `python -m src.preprocess` | `data/processed/a2/*`, `reports/tables/a2_preprocessing_*.csv` | about 1 min |
| 3 | A2 EDA | `python -m src.eda_a2` | `reports/tables/a2_*correlation*`, `a2_temperature_*`, `a2_class_distribution_by_season.csv` | under 1 min |
| 4 | Random Forest (A2) | `python -m src.models.rf` (logs to `results.csv`) | `reports/tables/a2_rf_*.csv`, SHAP figures | about 17 min |
| 5 | RF diagnostics (A2) | `python -m src.models.rf_diagnostics` | `a2_rf_sensor_change.csv`, `a2_rf_error_by_score.csv` | 10 s |
| 6 | HKUST Isolation Forest (A2) | `python -m src.models.iforest_a2` | `a2_iforest_injected_faults.csv` | 5 s |
| 7 | HKUST Isolation Forest, tuned (A1) | `python -m src.models.iforest_a1` | `a1_iforest_tuning.csv`, `a1_iforest_test.csv` | 35 s |
| 8 | Baselines + linear models (A1) | `python -m src.models.linear` | `a1_baseline_results.csv`, coefficients, default vs tuned | 1 min |
| 9 | Balancing study (A1) | `python -m src.models.linear_balancing` | `a1_balancing*.csv`, `a1_target_distribution.csv`, `a1_ridge_error_by_range.csv` | 1 min |
| 10 | Final A1 evaluation | `python -m src.models.linear_evaluation --log` (6 rows) | `a1_eval_*.csv`, `a1_peer_rule.csv`, `a1_ridge_residuals.png` | about 6 min |
| 11 | XGBoost (B1) | `python -m src.models.xgb --tune` (logs 2 rows; `--no-log` to skip) | `b1_xgb_*.csv`, SHAP figures | about 1 min |
| 12 | MLP balancing (B2) | `python -m src.models.mlp_balancing` | `b2_balancing.csv` (decides the MLP classifier) | 1 min |
| 13 | MLP tuning (B2) | `python -m src.models.mlp --tune` | `b2_mlp_default_vs_tuned.csv` | about 2 min |
| 14 | MLP final run (B2) | `python -m src.models.mlp --log` (2 rows) | console metrics | 40 s |
| 15 | GPVS-Faults Isolation Forest (B1, B2) | `python -m src.models.iforest --log` (1 row) | `gpvs_iforest_by_scenario.csv`, `b1_iforest_gpvs_detection.png` | 10 s |
| 16 | Notebooks | execute all 7 notebooks in `notebooks/` (see §1 for the kernel) | figures for EDA, baselines, XGBoost, MLP | about 5 min |

Steps 4, 10, 11, 14 and 15 append to `results/results.csv`. Clear the old rows for a model before re-logging it, so the file keeps one row per model. Notebooks do not log by default.

**Tests:** `python -m unittest discover -s tests` (20 tests, about 1 s).

## 4. Determinism

- **One seed:** `random_state: 42` in `configs/config.yaml`, used for every sampled or randomised step.
- **One source of truth:** split dates, the unseen-inverter list, the resampling frequency, the 70% coverage threshold, the 50 W/m² daylight threshold and the irradiance sensor-change date are read from `configs/config.yaml`.
- **Deterministic data preparation:** `prepare_data.py` has no randomness. The irradiance correction factor (0.570) is computed from the data, not typed in.
- **No leakage through preprocessing:** imputers and scalers are fitted on the training period only; tuning uses time-ordered folds; held-out inverters are excluded from training and validation; every model forecasts Health(t + 1 h) from features at t with an exact-time lead.
- **Checked reproducibility:** re-running XGBoost tuning gave identical scores (only run times differ); the MLP and XGBoost test RMSE were reproduced exactly by independent scripts.

## 5. Results tracking

- Each final model run appends one row to `results/results.csv` via `log_experiment` in `src/evaluate.py`: student, model, task, train/val/test/unseen scores in the shared format (`RMSE: x` or `F1: x, Faulty recall: y`), notes.
- Detailed tables are in `reports/tables/`; every number quoted in the reports comes from one of these files, `results/results.csv` or a notebook cell.

## 6. Caveats affecting reproduction

- The irradiance sensor changes scale on **2023-02-14**. `prepare_data.py` rescales irradiance after that date by a factor estimated from 2023 against 2022, and keeps the raw values in `irradiance_raw_wm2`. Results are also reported before and after that date.
- A2's pipeline applies its own physical cleaning, so its test set has about 1% fewer rows than A1's, B1's and B2's.
- The MLP trains on a seeded random sample of 60,000 training rows; other splits are scored in full.
- Notebooks are committed with their outputs. Re-executing them reproduces the numbers; only timestamps and run times change.

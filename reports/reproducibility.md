# Reproducibility (draft, A1)

Every number in the report can be regenerated from the raw datasets with the steps below. This section covers the A1 pipeline in full. Teammates add the entry points for their own models (marked "to be completed").

## 1. Environment

- **Python 3.12** (developed on 3.12.4, Windows 11, CPU only; no GPU needed).
- Virtual environment plus pinned packages:
  ```
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- `requirements.txt` pins exact versions of every top-level package: pandas 3.0.6, pyarrow 26.0.0, numpy 2.5.4, rdflib 7.6.0, PyYAML 6.0.3, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, jupyter 1.1.1, openpyxl 3.1.5, statsmodels 0.15.0.
- Run notebooks with the venv's own Jupyter (`python -m nbconvert ...` or `python -m notebook`). `python -m jupyter` can pick up a system-wide installation instead.

## 2. Data

| Dataset | Source | Licence | Where it goes |
|---|---|---|---|
| HKUST Rooftop PV 2021-2023 | Dryad, https://doi.org/10.5061/dryad.m37pvmd99 (version of 8 Sep 2024, `Dataset.zip`, 296 MB) | Public domain (CC0), per the Dryad page | unzip into `data/raw/` |
| GPVS-Faults | Mendeley Data, DOI 10.17632/n76t439f65.1 | to be completed (B2) | to be completed (B2) |

**Citations**
- Dataset: Lin Z., Zhou Q., Wang Z., Wang C., Bookhart D. B., Leung-Shea M. (2024). *A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics* [Dataset]. Dryad. https://doi.org/10.5061/dryad.m37pvmd99
- Paper: Lin Z. et al. (2025). Scientific Data. https://doi.org/10.1038/s41597-025-04397-y

The raw files are never modified. A known syntax error in the metadata `.ttl` file (Zone D block) is corrected in memory by `src/prepare_data.py`, and the script prints a message when it does so. `data/` is git-ignored.

## 3. Pipeline, in order

| Step | Command | Output | Time (measured) |
|---|---|---|---|
| Stage 3: data preparation | `python -m src.prepare_data` | `data/processed/*.parquet`, `reports/tables/{prep_summary, coverage_by_inverter, gaps_by_inverter_month}.csv` | about 75 s |
| Stage 6: A1 EDA | execute `notebooks/01_eda_A1.ipynb` | `reports/figures/a1_q*.png` | about 1 min |
| Stage 8: baselines + linear models | `python -m src.models.linear` (add `--log` to append to `results/results.csv`) | `reports/tables/a1_*.csv`, `reports/figures/a1_baselines_by_split.png` | about 70 s |
| Stage 8 notebook | execute `notebooks/03_baselines_A1.ipynb` | same, with findings | about 80 s |
| Stage 7: target and sensitivity (B1) | `python -m src.targets` | to be completed (B1) | |
| Stage 9: MLP (B2) | `python -m src.models.mlp` (`--tune`, `--explain`) | to be completed (B2) | |
| Isolation Forest, GPVS (B2) | `python -m src.models.iforest` | to be completed (B2) | |

Timings are for an 18-thread laptop CPU.

## 4. Determinism

- **One seed:** `random_state: 42` in `configs/config.yaml`. It is used for every sampled or randomised step in A1's code: the plotting samples, Logistic Regression, and permutation importance.
- **One source of truth:** split dates, the unseen-inverter list, the resampling frequency, the 70% coverage threshold and the 50 W/m² daylight threshold are all read from `configs/config.yaml`, never hard-coded.
- **Deterministic data preparation:** there is no randomness in `prepare_data.py`, so re-running it gives identical Parquet contents. Timestamps are parsed against two explicit formats, and any value matching neither raises an error.
- **No leakage through preprocessing:** in `src/models/linear.py`, mean-imputation and `StandardScaler` are fitted on the training period only. Tuning uses `TimeSeriesSplit` (no shuffling). Unseen inverters are excluded from training and validation.
- **Known small source of variation:** `GridSearchCV` runs folds in parallel (`n_jobs=-1`). For Ridge and Logistic Regression this does not change results.

## 5. Results tracking

- Each final model run appends one row to `results/results.csv` via `log_experiment` in `src/evaluate.py`: student, model, task, train/val/test/unseen scores, notes.
- A1's detailed tables are in `reports/tables/a1_*.csv`; every number quoted in A1's report sections comes from one of these files or a notebook cell.

## 6. Known caveats affecting reproduction of the reported numbers

- The irradiance sensor's readings change on **2023-02-14**. Any result that depends on raw irradiance differs before and after that date. A1's tables report the test year split at that date.
- B1's target (`src/targets.py`) may still be revised. Re-running `python -m src.models.linear --log` regenerates all A1 baseline numbers without code changes.
- Notebooks are committed with their outputs. Re-executing them overwrites the outputs with identical numbers, apart from timestamps in `results.csv`.

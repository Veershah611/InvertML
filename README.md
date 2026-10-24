# Solar Inverter Health Monitoring

Forecasts the health of 44 rooftop solar inverters on the HKUST campus one hour ahead, classifies them as Healthy / Warning / Faulty, and flags unusual behaviour with an anomaly detector. A four-person project that follows all 12 stages of the Innovative Assignment pipeline.

| Document | What it contains |
|---|---|
| [REPORT.md](REPORT.md) | **The report**: every stage, every result, limitations |
| [VIVA.md](VIVA.md) | Short answers to likely viva questions |
| [features_used.md](features_used.md) | The 20 features and how each model uses them |
| [formulae.md](formulae.md) | Every formula in the code |
| [Solar_Inverter_Health_Monitoring.md](Solar_Inverter_Health_Monitoring.md) | The original project plan and team split |

## Results at a glance

Test year 2023, after training on May 2021-June 2022 and choosing on July-December 2022:

| Task | Baseline | Best model (chosen on validation) | Test score |
|---|---|---|---|
| Health one hour ahead (RMSE, health points) | Persistence 12.35 | Random Forest | **10.45** (Ridge 10.17) |
| Faulty one hour ahead (Faulty recall) | Health rule 0.60 | Logistic Regression | **0.71** |
| Injected faults on HKUST (detection / false alarms) | none | Isolation Forest | **76.8 % / 1.1 %** |
| Real lab faults, GPVS-Faults (detection / false alarms) | none | Isolation Forest | **39 % / 0 %** |

Biggest data finding: the campus irradiance sensor changed scale on 2023-02-14; irradiance is corrected from that day (factor 0.570), otherwise half of the test year looked faulty.

## Setup

**1. Python 3.12 environment**
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. HKUST dataset** (public domain): download `Dataset.zip` from https://doi.org/10.5061/dryad.m37pvmd99 and unzip it into `data/raw/`, so that `data/raw/Dataset/Metadata/` and `data/raw/Dataset/Time series dataset/` exist.

**3. GPVS-Faults dataset** (CC BY 4.0): download `CSV_Files.zip` from https://data.mendeley.com/datasets/n76t439f65/1 and extract the 16 files `F0L.csv` ... `F7M.csv` into `data/raw/GPVS-Faults/`.

`data/` is not in git.

## Run

Everything, from raw files to the final comparison (about 20 minutes):
```
python -m src.run_all
```
Or one step at a time:

| Step | Command | Output |
|---|---|---|
| Stage 3 data preparation | `python -m src.prepare_data` | `data/processed/hkust_15min.parquet` |
| Stages 4, 7 target, features, splits | `python -m src.dataset` | `data/processed/model_table.parquet` |
| Stage 6 EDA | `python -m src.eda` | `reports/figures/eda_*.png` |
| Stages 4, 5, 7 studies | `python -m src.studies` | `reports/tables/study_*.csv` |
| Stages 8-10 models | `python -m src.models.linear` (also `rf`, `xgb`, `mlp`, `iforest`) | `results/results.csv`, `reports/tables/<model>_*.csv` |
| Stage 11 comparison | `python -m src.compare` | `reports/tables/compare_*.csv`, `reports/figures/compare_*.png` |
| Tests | `python -m unittest discover -s tests` | 7 checks (formula, no leakage, splits) |

The notebooks in `notebooks/` show these outputs with short explanations; they do not retrain anything.

## Project structure
```
configs/config.yaml       paths, dates and thresholds (one place for every setting)
src/
  common.py               shared folders, config and plot helper
  prepare_data.py         Stage 3: raw files -> 15-min table (A1)
  dataset.py              Stages 4 + 7: cleaning, health target, 20 features, splits
  eda.py                  Stage 6: eight EDA questions, two per member
  studies.py              Stages 4, 5, 7: imputation, balancing, thresholds, correlation
  evaluate.py             Stage 11: metrics + run_model(), the recipe every model uses (B2)
  compare.py              Stage 11: final comparison of all models
  run_all.py              runs every step in order
  models/
    linear.py             baselines, Ridge, Logistic Regression (A1)
    rf.py                 Random Forest (A2)
    xgb.py                XGBoost + SHAP (B1)
    mlp.py                neural network (B2)
    iforest.py            Isolation Forest: HKUST injected faults (A1, A2), GPVS real faults (B1, B2)
notebooks/                01_data_and_eda, 02_target_and_features, 03_models, 04_evaluation
tests/test_pipeline.py    checks for the rules that must never break
results/results.csv       every model's score on every evaluation set
reports/figures, tables   all figures and tables used in the report
```

## Team

| Member | Leads | Model family | EDA questions |
|---|---|---|---|
| A1 | Stages 1-3 (problem, data, preparation), Stage 8 baselines | Linear (Ridge, Logistic) | power vs irradiance; gaps |
| A2 | Stages 4-5 (preprocessing, balancing) | Random Forest | temperature; feature correlation |
| B1 | Stage 7 (target, features, selection) | XGBoost | degradation; underperformers |
| B2 | Stages 11-12 (evaluation, impact) | MLP | seasonality; class distribution |

Both pairs run Isolation Forest: Pair A on HKUST with injected faults, Pair B on GPVS-Faults real faults.

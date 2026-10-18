# Solar Inverter Health Monitoring & Predictive Maintenance

A machine learning project that monitors the health of rooftop solar inverters. It uses up to three years of inverter telemetry and on-site weather data from the HKUST campus.

The full design, with every decision and its justification, is in [Solar_Inverter_Health_Monitoring.md](Solar_Inverter_Health_Monitoring.md). Who owns what, and current status, are in [TEAM_PROGRESS.md](TEAM_PROGRESS.md).

## What the project does

Inverters are the most failure-prone part of a PV system. Low output alone does not tell you whether the cause is a cloud or a fault. The project separates weather-driven drops from system-driven underperformance, and produces three outputs per inverter every 15 minutes:

1. **Health score (0-100)**, plus a short-horizon forecast of it.
2. **Status class**: `Healthy` / `Warning` / `Faulty`.
3. **Anomaly flag** from an unsupervised detector (Isolation Forest), to catch failure patterns the labels miss.

No real fault logs exist for the HKUST stations, so the health score and status labels are **engineered**. A second dataset with real fault labels is used to check that the approach catches real faults.

The work follows a 12-stage pipeline:
1. problem definition
2. data acquisition
3. data preparation
4. preprocessing
5. class balancing
6. EDA
7. feature engineering and target design
8. baselines
9. modelling: Ridge/Logistic, Random Forest, XGBoost, MLP, and Isolation Forest
10. hyperparameter tuning with time-series CV
11. evaluation: chronological split, unseen-station test, real-fault check
12. impact and limitations

Everything runs on CPU-only laptops.

## Datasets

### 1. HKUST Rooftop PV, 2021-2023 (primary)
- **Source:** Lin et al., *A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics*, Scientific Data (2025). Dryad: https://doi.org/10.5061/dryad.m37pvmd99
- **Contents:**
  - 60 rooftop PV stations on the HKUST campus in Hong Kong.
  - Inverter-level generation every 5 minutes: AC power, per-phase AC current/voltage/frequency/power, and DC voltage.
  - 1-minute weather from one on-site station: irradiance, temperature, wind, humidity, rainfall, pressure, visibility.
  - Station metadata as a Brick schema model (`.ttl`).
- **What we use:** inverter-level data only, which means **44 inverters at 37 stations**, resampled to 15-min. The other 23 stations have only hourly site-level data and are excluded.
- **Limitations:**
  - No DC current/power or module temperature.
  - Loggers record only 06:00-17:59, so summer evening generation is missing.
  - One weather station serves all rooftops.
  - Start dates vary by inverter (mid-2021 to 2023).

### 2. GPVS-Faults (validation only)
- **Source:** Bakdi et al., Mendeley Data, DOI 10.17632/n76t439f65.1
- **Contents:** lab experiments on a grid-connected PV system, with a fault-free scenario and 7 fault types (PV array, inverter, grid, sensor and MPPT faults).
- **Use:** checks that the anomaly detector catches **real** faults. It has its own small feature pipeline and is never merged with HKUST.

### 3. NIST Campus PV (fallback, not currently used)
It has DC power, module temperature and inverter fault codes, but only for 3 arrays. It would be used only if we decided we needed those signals.

## Setup

### 1. Python environment
Requires Python 3.12. From the repo root:
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Get the HKUST dataset
1. Download the dataset from Dryad: https://doi.org/10.5061/dryad.m37pvmd99 (about 296 MB zipped, about 1 GB unzipped).
2. Unzip it into `data/raw/` so the layout is:
```
data/raw/
├── README.md
└── Dataset/
    ├── Metadata/
    │   └── PV generation system metadata.ttl
    └── Time series dataset/
        ├── Meteorological dataset/      # 7 folders, one file per year
        └── PV generation dataset/
            ├── PV stations with panel level optimizer/
            │   ├── Inverter level dataset/   # 44 CSVs (these are what we use)
            │   └── Site level dataset/
            └── PV stations without panel level optimizer/
```
`data/` is git-ignored. Raw and processed data are never committed.

### 3. Build the processed data
```
python -m src.prepare_data
```
This takes about 75 seconds. It leaves the raw files untouched and writes:

| File | Contents |
|---|---|
| `data/processed/inverter_metadata.parquet` | one row per inverter: capacity (kWp), station, orientation, panel specs |
| `data/processed/inverter_5min.parquet` | all 44 inverters at 5-min, float32 |
| `data/processed/weather_1min.parquet` | all weather variables at 1-min, float32 |
| `data/processed/hkust_15min_v1.parquet` | **the main table**: inverters + weather + metadata at 15-min (1,588,901 rows x 45 columns) |
| `reports/tables/prep_summary.csv` | rows and columns before/after each preparation step |
| `reports/tables/coverage_by_inverter.csv` | daylight data coverage per inverter |
| `reports/tables/gaps_by_inverter_month.csv` | missing daylight intervals per inverter per month |

Teammates who don't want to run this step can download the shared `hkust_15min_v1.parquet` from the team drive and place it in `data/processed/`.

### 4. GPVS-Faults
Download instructions and the loader will be added by B2.

## Repository structure
```
├── configs/config.yaml   # paths, seed, timezone, thresholds, split dates (single source of truth)
├── data/                 # raw + processed data (git-ignored)
├── src/
│   ├── prepare_data.py   # Stage 3: raw files -> hkust_15min_v1.parquet (A1)
│   └── models/           # one file per model family
├── notebooks/            # thin drivers that import from src/, one owner per notebook
├── results/              # results.csv: one row per final model run
├── reports/              # figures/ and tables/ for the report
├── requirements.txt      # pinned package versions
├── Solar_Inverter_Health_Monitoring.md   # full project design
├── TEAM_PROGRESS.md      # status per pipeline stage and person
└── A1_WORKLOG.md         # detailed log of A1's work
```

## Team
Pair A (A1, A2) leads data preparation and preprocessing. Pair B (B1, B2) leads features, target design and evaluation. Every member builds one model family, answers two EDA questions and builds one feature group. See §13 of the design doc.

# A1 Work Log

Status: ✅ done · 🔄 in progress · ⬜ not started

## Done

**Repo setup** ✅
- Folder layout from §13.6: `src/`, `configs/`, `notebooks/`, `results/`, `reports/figures/`, `data/` (git-ignored).
- `configs/config.yaml`: paths, seed 42, timezone, 15-min frequency, 70% coverage threshold, split dates.
- Python 3.12 `.venv`; `requirements.txt` pins the 9 top-level packages.

**Stage 2: Data understanding** ✅ (except the licence)
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

**Stage 3: Metadata step** ✅ ([src/prepare_data.py](src/prepare_data.py))
- Parses the Brick `.ttl` with rdflib and SPARQL, walking Station → Inverter → Panel group.
- Fixes a syntax error in the raw `.ttl` (Zone D block, `.` → `;`) in memory only.
- `capacity_kwp` = Σ(panel count × panel W) / 1000 per inverter (DC size; this is what PR needs). The station's rated kW is kept as a separate column.
- Matches all 44 CSV files to their Brick inverter, and stops with an error if any file is unmatched.
- Output: `data/processed/inverter_metadata.parquet` (44 rows, 19 columns).
- Run with: `.\.venv\Scripts\python.exe -m src.prepare_data` (the whole script takes ~50 s)

**Stage 3: CSV → Parquet conversion** ✅ (`src/prepare_data.py`, step 2)
- Timestamps are parsed with the two known formats tried explicitly; anything else raises an error. They are kept as naive HKT (Hong Kong has no daylight saving).
- `inverter_5min.parquet`: 4,766,698 rows. Columns are `inverter_id`, `time` and the 17 measurements as float32, renamed to snake_case with units (e.g. `ac_power_w`, `dc_voltage_v`, `l1_ac_current_a`). `NA` is stored as NaN. 201 MB on disk, 367 MB in memory.
- `weather_1min.parquet`: 1,567,792 rows (2021-01-01 → 2023-12-31). The 7 variables become 8 float32 columns, joined on time. NaNs (at most 617 per column) come only from minutes missing in one variable's file.
- `Rainfall_2021.xlsx` is read with `openpyxl` (added to requirements.txt).
- Checks: inverter IDs exactly match the metadata table; 0 duplicate (inverter, time) rows; no duplicate weather timestamps.
- Note: the raw files label reactive power in W; strictly the unit is var.

**Stage 3: 15-min resampling + merge** ✅ (`src/prepare_data.py`, step 3)
- Bins are labelled by their start (10:00 covers 10:00–10:14), the same for inverter and weather data.
- Inverters, 5 → 15 min: mean of the available readings. `n_valid_5min` (0–3) counts the valid `ac_power_w` readings in each bin. 4,766,698 → **1,588,901 rows**.
- Weather, 1 → 15 min: mean for readings, **sum** for rainfall, **circular mean** for wind direction. `n_valid_1min` (0–15) counts valid irradiance readings. 1,567,792 → 104,523 rows.
- Merge: weather joined on `time`, metadata on `inverter_id` (both checked many-to-one).
- Output: **`hkust_15min_v1.parquet`**, 1,588,901 rows × 45 columns, 101 MB (267 MB in memory). The full script runs in ~75 s.
- Hand-checked: one SQ1 bin (2022-07-01 12:00) matches the mean of the raw 5-min power and the 1-min irradiance exactly.
- For A2 (Stage 4): 94,441 bins have `n_valid_5min = 0` (a row exists but all power values are NA). 21k bins have 1–2 valid readings. 12,172 rows have no weather data (`n_valid_1min = 0`).

## Open issues
- Licence: the README says "N/A". Confirm CC0 and the published citation on the Dryad page.
- Tilt is empty for UG Hall4 and Hall7 (`"0/15deg"`, `"0/11deg"`). Azimuth is `"Mixed"` for several stations.
- Capacity looks wrong for Indoor Sports Centre (43.8 kWp on a 10 kW inverter) and Zone D Inverter 1 (station total on a 33.3 kW inverter). To be checked against actual max AC power.
- Tell B2: the data starts mid-2021, not Jan 2021, and the unseen-station set is about 7 of 37.

## To do (A1)
- ⬜ Stage 3: gap analysis (full 15-min grid per inverter); coverage filter; before/after table; share `hkust_15min_v1.parquet`
- ⬜ EDA: power vs. irradiance; gaps/offline heatmap
- ⬜ Features: `is_offline`, minutes offline in the last 24 h
- ⬜ Baselines: persistence, Ridge; majority, rule-based, Logistic
- ⬜ GridSearch tuning; coefficients + permutation importance
- ⬜ Isolation Forest on HKUST with injected faults (with A2)
- ⬜ Reproducibility section of the report

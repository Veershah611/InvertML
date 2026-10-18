# A1 Work Log

## Done

**Repo setup** (Done)
- Folder layout from §13.6: `src/`, `configs/`, `notebooks/`, `results/`, `reports/figures/`, `data/` (git-ignored).
- `configs/config.yaml`: paths, seed 42, timezone, 15-min frequency, 70% coverage threshold, split dates.
- Python 3.12 `.venv`; `requirements.txt` pins the 9 top-level packages.

**Stage 2: Data understanding** (Done, except the licence)
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

**Stage 3: Metadata step** (Done) ([src/prepare_data.py](src/prepare_data.py))
- Parses the Brick `.ttl` with rdflib and SPARQL, walking Station → Inverter → Panel group.
- Fixes a syntax error in the raw `.ttl` (Zone D block, `.` → `;`) in memory only.
- `capacity_kwp` = Σ(panel count × panel W) / 1000 per inverter (DC size; this is what PR needs). The station's rated kW is kept as a separate column.
- Matches all 44 CSV files to their Brick inverter, and stops with an error if any file is unmatched.
- Output: `data/processed/inverter_metadata.parquet` (44 rows, 19 columns).
- Run with: `.\.venv\Scripts\python.exe -m src.prepare_data` (the whole script takes ~50 s)

**Stage 3: CSV → Parquet conversion** (Done) (`src/prepare_data.py`, step 2)
- Timestamps are parsed with the two known formats tried explicitly; anything else raises an error. They are kept as naive HKT (Hong Kong has no daylight saving).
- `inverter_5min.parquet`: 4,766,698 rows. Columns are `inverter_id`, `time` and the 17 measurements as float32, renamed to snake_case with units (e.g. `ac_power_w`, `dc_voltage_v`, `l1_ac_current_a`). `NA` is stored as NaN. 201 MB on disk, 367 MB in memory.
- `weather_1min.parquet`: 1,567,792 rows (2021-01-01 → 2023-12-31). The 7 variables become 8 float32 columns, joined on time. NaNs (at most 617 per column) come only from minutes missing in one variable's file.
- `Rainfall_2021.xlsx` is read with `openpyxl` (added to requirements.txt).
- Checks: inverter IDs exactly match the metadata table; 0 duplicate (inverter, time) rows; no duplicate weather timestamps.
- Note: the raw files label reactive power in W; strictly the unit is var.

**Stage 3: 15-min resampling + merge** (Done) (`src/prepare_data.py`, step 3)
- Bins are labelled by their start (10:00 covers 10:00–10:14), the same for inverter and weather data.
- Inverters, 5 → 15 min: mean of the available readings. `n_valid_5min` (0–3) counts the valid `ac_power_w` readings in each bin. 4,766,698 → **1,588,901 rows**.
- Weather, 1 → 15 min: mean for readings, **sum** for rainfall, **circular mean** for wind direction. `n_valid_1min` (0–15) counts valid irradiance readings. 1,567,792 → 104,523 rows.
- Merge: weather joined on `time`, metadata on `inverter_id` (both checked many-to-one).
- Output: **`hkust_15min_v1.parquet`**, 1,588,901 rows × 45 columns, 101 MB (267 MB in memory). The full script runs in ~75 s.
- Hand-checked: one SQ1 bin (2022-07-01 12:00) matches the mean of the raw 5-min power and the 1-min irradiance exactly.
**Stage 3: Gap analysis + coverage filter** (Done) (`src/prepare_data.py`, step 4)
- The loggers only record **06:00–17:59**, but in June irradiance stays above 50 W/m² until ~19:00. **Summer evening generation is never recorded** (a dataset limitation for the report).
- Gaps appear as rows full of `NA`, not as missing rows. Rebuilding the full 06:00–17:45 grid per inverter (from its first bin to the end of 2023) added **0 rows**, so the raw grid is already complete.
- Missing bin = `n_valid_5min == 0`. The 06:00 hour is 28% missing (inverters asleep before sunrise); other hours are about 4%. That's why coverage counts **daylight bins only**: irradiance ≥ 50 W/m² (`daylight_irradiance_wm2` in config). Bins without weather data are excluded.
- Coverage is measured from each inverter's own start date. Min **88.8%** (SQ Block P), median 96.6%. **No inverter falls below 70%, so none are dropped.**
- Worst months: LSK South Sep 2023 (97% missing) and Oct 2023 (91%), Zone J2 Aug 2022 (92%), S H Ho Sports Hall Oct 2023 (86%), Library Dec 2023 (72%). These look like real outages and are relevant for `is_offline`.
- Outputs in `reports/tables/`: `gaps_by_inverter_month.csv`, `coverage_by_inverter.csv`, and `prep_summary.csv` (the before/after table: 4,766,698 × 19 → 1,588,901 × 20 → 1,588,901 × 45, 44 inverters throughout).
- For A2 (Stage 4): 94,441 bins have `n_valid_5min = 0` (a row exists but all power values are NA). 21k bins have 1–2 valid readings. 12,172 rows have no weather data (`n_valid_1min = 0`).

**Stage 6: EDA, Q1 power vs irradiance + Q2 gaps** (Done) ([src/eda.py](src/eda.py), [notebooks/01_eda_A1.ipynb](notebooks/01_eda_A1.ipynb))
- Code is in `src/eda.py`; the notebook is a thin driver with a finding and a decision under every plot. It runs top to bottom in about 1 minute.
- 7 figures in `reports/figures/a1_q*.png`.
- **Q1 findings:**
  - Most inverters have a slope of 0.6-0.8 kW/kWp per 1000 W/m².
  - Inverter clipping is visible at about 0.75-0.8 kW/kWp.
  - 99% of bins produce power from **50 W/m²** upward (about 75% at 20 W/m²), which supports a 50 W/m² night threshold.
- **Irradiance sensor change on 2023-02-14:**
  - Output per unit irradiance drops from about 0.9 to about 0.5 for all inverters at once.
  - Peak irradiance jumps from about 840 to about 1,400 W/m², with a 3,609 W/m² spike that day.
  - Raw irradiance is therefore not on the same scale in the 2023 test year. The team must decide: rescale, use peer deviation, or add a period flag.
- **Zone D capacities are wrong:**
  - Inverter 2 reaches 50.1 kW on a 25 kW inverter (slope 1.31).
  - Inverter 1 has slope 0.32.
- Indoor Sports Centre's capacity is consistent with its data; the "10 kW" inverter model in the metadata is the error.
- **Q2 findings:**
  - 78.5% of gaps are whole days (4-12 daylight hours) and 5.7% are multi-day, together holding 99% of missing data.
  - Gaps of up to 1 hour are 0.4% of missing data.
  - The 06:00 hour is 28% missing (inverters asleep); other hours are 3.8%.
  - Several outages hit all inverters at a station at once (Library, S H Ho, Zone J1 + A7), which points to communication failures.

## Open issues
- Licence: the README says "N/A". Confirm CC0 and the published citation on the Dryad page.
- Tilt is empty for UG Hall4 and Hall7 (`"0/15deg"`, `"0/11deg"`). Azimuth is `"Mixed"` for several stations.
- Capacity looks wrong for Indoor Sports Centre (43.8 kWp on a 10 kW inverter) and Zone D Inverter 1 (station total on a 33.3 kW inverter). To be checked against actual max AC power.
- Tell B2: the data starts mid-2021, not Jan 2021, and the unseen-station set is about 7 of 37.

## To do (A1)
- [ ] Share `hkust_15min_v1.parquet` with the team (Drive/OneDrive)
- [ ] Tell the team about the irradiance sensor change and the Zone D capacities (see EDA)
- [ ] Features: `is_offline`, minutes offline in the last 24 h
- [ ] Baselines: persistence, Ridge; majority, rule-based, Logistic
- [ ] GridSearch tuning; coefficients + permutation importance
- [ ] Isolation Forest on HKUST with injected faults (with A2)
- [ ] Reproducibility section of the report

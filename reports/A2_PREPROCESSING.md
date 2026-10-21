# A2 preprocessing

Implemented in `src/preprocess.py`. Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Integration with B1's feature update

The workspace includes upstream commit `876302c`, which incorporates B1's
`7660e70` Stage 7 feature changes. B1 did not change `src/targets.py` in this push:
the target formula and 70/90 status thresholds below still apply.

A2's forecast datasets include all seven `B1_CORE_FEATURES`:
`specific_yield`, `performance_ratio`, `peer_dev_campus`, `peer_dev_station`,
`self_dev_30d`, `thermal_stress` and `temp_delta_stc`.

Integration corrections in the shared `src/features.py`:

- Restore `B2_ALL_FEATURES` and `build_b2_features`, still imported by A1's linear
  model and B2's MLP/notebook.
- Preserve missing power/capacity/weather in physical features. Missing inverter
  telemetry must not contribute an invented zero to peer medians. PR requires
  irradiance of at least 100 W/m², consistent with B1's target.
- Replace the incoming 200-row rolling baseline with an actual preceding
  30-calendar-day median, conditioned on inverter and irradiance bin. Exclude
  the current timestamp and leave cold-start histories missing for imputation.
- Preserve input row order/index in rolling features.
- Respect protected columns during correlation pruning and stop using a
  candidate after it has been dropped. Plotting loads seaborn only when needed.

A2 builds peer statistics after B2's split and holdout filtering. Training and
validation use known-inverter peers only. Test uses all telemetry available at
each test timestamp; unseen-test rows are then sliced from this same feature
table. The preceding 30 days of observed history carry across train/validation/
test boundaries for known inverters. Forecast labels still cannot cross those
boundaries. This assumes sequential prediction with current telemetry available;
it is not a single forecast issued for an entire unseen year in advance.

All seven B1 features remain candidates in A2. The 13-feature list in B1's
upstream report is not treated as a universal fixed selection. Feature pruning
for later model tuning must be fitted within each training fold, and any selected
subset must be evaluated on validation data.

On 21 October 2026, A2 added 18 exact-time lag and past-only rolling columns
for performance ratio and campus peer deviation. The features use 15-minute,
one-hour and one-day lags, plus 1-hour/4-hour/one-day means and standard
deviations. One-day lag means the same clock time the previous day, not 96
remaining daylight rows. They are computed separately per inverter with
historical context across split boundaries. Tests cover missing intervals,
different inverters and future perturbations.

## B1 target labels

The continuous target comes from `src/targets.py`:

```text
expected_power_w = max(irradiance_wm2 * capacity_kwp * 0.78, 100)
health_score = clip(100 * ac_power_w / expected_power_w, 0, 100)
```

Health is undefined when irradiance is below 100 W/m². Valid scores map to:

| Score | Label |
|---|---|
| 90–100 | Healthy |
| 70–<90 | Warning |
| 0–<70 | Faulty |

These are engineered labels, not measured fault records. A2 uses the forecasting
design already used by A1: features at time `t` predict `health_score_lead4` and
`status_lead4` exactly one hour later. Current power may therefore be an input.
Neither current nor future target columns are included in the feature allowlist.

B1's function currently replaces missing power with zero, and its status
function defaults missing scores to Healthy. A2 only accepts observed, valid
target measurements and explicitly keeps missing scores/statuses undefined.
This guard is local to A2; existing B1/A1/B2 model runs are not automatically fixed.
Their earlier results must not be treated as directly comparable to this run.

## B2 splits

The implementation calls `src.evaluate.split_chronological` and
`src.evaluate.split_unseen_stations`. B2's current functions hardcode these dates:

| Partition | Dates | Forecast-ready rows | Inverters |
|---|---|---:|---:|
| Training | through 30 June 2022 | 260,819 | 36 |
| Validation | 1 July–31 December 2022 | 182,731 | 38 |
| Test | 1 January–31 December 2023 | 441,205 | 44 |
| Unseen test | held-out inverters during 2023 | 44,634 | 6 |

Configured held-out inverter IDs are excluded from both training and validation.
The unseen test is a subset of the full test, not an additional disjoint partition.
B2's configuration lists seven IDs, but `sq14_inverter` does not occur in A1's
input. The six matching IDs are retained exactly as B2 defines them. This is an
inverter holdout; it does not guarantee that every inverter from the same physical
station is excluded. A whole-station benchmark needs B2 to define that grouping.

Future labels are joined on `(inverter_id, time + 1 hour)` inside each partition.
This prevents overnight jumps, bridging missing timestamps and using a target
from the next partition. Rows require observed health at both endpoints.

For tuning, `time_series_folds` applies B2's expanding-window `TimeSeriesSplit`
approach with three folds over unique timestamps. All inverters at the same
timestamp stay together. Training rows whose target timestamp reaches the next
fold are purged. The date ranges are in `tables/a2_preprocessing_cv.csv`.

## Cleaning and missing data

Input: `data/processed/hkust_15min_v1.parquet`, 1,588,901 rows and 45 columns.
The existing `v2` table is not used: it lacks inverter IDs and contains scaled
columns without a corresponding preprocessing implementation in the repository.

- Retain daylight rows from 50 W/m², following A1's recorded EDA finding in
  `A1_WORKLOG.md`. B1's separate 100 W/m² rule governs label availability.
- Reject negative power and power above `1.1 * capacity_kwp * 1000` from modeling:
  3,209 readings. This uses array DC capacity as the project's screening proxy;
  it is not a separately verified inverter AC rating. A1's Zone D capacity
  concerns remain relevant.
- Reject irradiance outside 0–1500 W/m²: 218 readings.
- Validate capacity and broad temperature, humidity, wind, rainfall and
  visibility bounds. Extreme values within the bounds are retained.
- Preserve 94,441 missing-power/zero-valid-sample events in the complete cleaned
  grid with `is_offline`. This flag indicates unavailable telemetry, not a
  confirmed hardware fault. Observed zero power is retained and can receive a
  Faulty label in daylight.
- Preserve the complete logger grid for outage analysis. The daylight table has
  1,270,728 rows; offline rows remain present there where irradiance is known.
- Do not fill missing power or fabricate target labels. Long gaps remain
  missing. Model-input missing values are handled separately by fitted imputers.

The design document suggests time interpolation for short gaps. For causal
forecasting, interpolation using a later observation can leak future information.
This initial implementation deliberately uses training-fitted mean/median input
imputation and leaves the source gaps intact. Short-gap interpolation is not
implemented; a later retrospective analysis must be kept separate from online
forecast features. Offline prediction rows are excluded from this supervised
dataset, so performance on telemetry outages is not measured by its scores.

## Encoding, scaling and imputation comparison

Explicit numeric features: B1's seven physical features, A2's 18 history
features, current power, irradiance, ambient temperature, wind speed, relative
humidity, capacity, tilt, altitude, valid 5-minute sample count,
and B2's four cyclical time features. Roof azimuth is one-hot encoded with
unknown categories ignored. Inverter/station IDs are retained as metadata but
excluded from model inputs.

Numeric imputation includes missingness indicators. Categorical missing values
use an Unknown category. All imputer statistics, encoder categories and scaler
statistics are fitted on training rows only. The tree transformer does not scale;
the second transformer applies StandardScaler for linear/neural models.

Two small, fixed Random Forest probes used identical seeded samples of 30,000
training and 15,000 validation rows. The classifier used balanced class weights
in both runs. This is an imputation study, not the A2 Random Forest or balancing
deliverable. No test scores were used to choose the imputer.

| Imputation | Validation RMSE | Validation MAE | Macro-F1 | Faulty recall |
|---|---:|---:|---:|---:|
| Median | 10.114 | 5.619 | 0.6374 | 0.6692 |
| Mean | 10.121 | 5.621 | 0.6410 | 0.6698 |

Mean was selected by validation macro-F1, with RMSE as a tie-breaker.
The differences are small and come from one fixed sample; they are not evidence
of statistical significance. Final transformers were then refitted on all
260,819 training rows. Test/validation data never contribute fitting statistics.

Before B1 feature integration, the median-imputation probe had RMSE 11.708,
macro-F1 0.5558 and Faulty recall 0.4908. After B1 integration but before A2
history, it had RMSE 10.350, macro-F1 0.6272 and Faulty recall 0.6780 on the
same sampled rows. These remain small fixed-model probes, not the final tuned
Random Forest evaluation.

When tuning a model, create a fresh transformer **inside** its Pipeline and pass
`list(time_series_folds(train))` as the cross-validation splits. Use
`feature_frame(train)` as the model input. Do not preprocess the entire training
period once before cross-validation: that would expose later fold statistics.

## Outputs and verification

`data/processed/a2/` contains:

- `cleaned_grid.parquet`: full grid with invalid measurements masked and flags.
- `daylight.parquet`: daylight rows after fixed validity filters.
- `train.parquet`, `val.parquet`, `test.parquet`, `test_unseen.parquet`:
  chronological, labeled forecast datasets; model features are still unscaled
  so preprocessing can be fitted independently within each CV fold.
- `preprocessor_tree.joblib`, `preprocessor_scaled.joblib`: transformers fitted
  only on the complete training partition.
- `manifest.json`: input path, target definition, selected strategy and features.

Tracked audit tables are `tables/a2_preprocessing_{audit,splits,cv,imputation}.csv`.
Audit flag counts can overlap; they should not be summed as independent removals.

Seventeen automated tests passed: missing/invalid labels, weather/capacity validity,
exact forecast timing, split/holdout boundaries, train-only imputation/scaling,
timestamp-grouped CV with label purging, duplicate/input protection, A1/B2 import
compatibility, missing-data-safe physical/peer features, actual calendar lookbacks,
irradiance-conditioned history, invariance to future rows, cross-partition history,
held-out peer isolation, and pruning protections. A real-data
check also verified that saved imputer statistics equal full training medians,
that all saved forecast rows have exactly one-hour horizons, and that both saved
transformers produce finite outputs on samples from every partition.

Execution environment: Python 3.14.6 with the repository-pinned pandas 3.0.6,
NumPy 2.5.4, PyArrow 26.0.0, scikit-learn 1.9.1, PyYAML 6.0.3 and matplotlib
3.11.2. The README recommends Python 3.12; that interpreter was not installed.

## Remaining A2 work

Class-balancing study, temperature/correlation EDA, lag/rolling features, full
Random Forest training/tuning/explainability, paired Isolation Forest evaluation,
and the project-wide leakage audit remain separate tasks. A1's documented
February 2023 irradiance sensor change also remains unresolved and affects B1
labels; the 2023 test distribution must be interpreted with that limitation.

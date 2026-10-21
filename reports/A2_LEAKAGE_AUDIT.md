# A2 leakage audit

This audit covers A2 preprocessing, EDA, feature generation and Random Forest
experiments. It also records issues found when reviewing other members' current
model scripts. The A2 safeguards do not automatically change those scripts.

| Risk | A2 control | Verification |
|---|---|---|
| Current-target circularity | Inputs at `t` predict health/status at exactly `t+1h` | Forecast rows record `target_time`; tests assert the one-hour difference |
| Overnight or missing-bin shift | Future target joined by `(inverter_id, timestamp)` | Test uses missing intervals and an overnight boundary |
| Crossing train/validation/test labels | Future labels joined separately inside each B2 partition | Boundary test and saved split tables |
| Future values in history | Lags use exact earlier timestamps; rolling windows exclude current and future rows | Future-perturbation and calendar-lag tests |
| Inverter mixing | History grouped by inverter ID | Group-isolation test |
| Held-out inverter influence | B2 IDs excluded from training/validation and from their peer calculations | Partition feature test and distinct inverter sets |
| Imputer/encoder/scaler leakage | Each model has a fitted `Pipeline`; fitting uses training rows only | Train-statistic test and saved transformer check |
| Hyperparameter CV leakage | Expanding time folds group all inverters at one timestamp and purge training labels touching validation | CV fold test and `a2_preprocessing_cv.csv` |
| Resampling leakage | Healthy undersampling is applied to training rows only | `src/models/rf.py` uses the same untouched validation sample for both strategies |
| Feature-selection leakage | Correlations calculated on training only; the final RF retains all features | `a2_selection_validation.csv` records a separate validation comparison |
| Tuning on test results | Balancing, imputation and RF settings use train CV/2022 validation; 2023 is scored afterward | Code execution order in `src/models/rf.py` |

## Target and data limitations

B1's labels are computed from irradiance, rated array capacity and measured AC
power. Missing power is *not* interpreted as a zero-power fault in A2. Healthy,
Warning and Faulty are engineered labels and should not be described as confirmed
maintenance events.

The input uses one campus weather station. A1 documented an irradiance sensor
change on 14 February 2023. Because irradiance enters the target formula, this
change can alter the apparent 2023 fault prevalence even if inverter condition is
unchanged. Since 2026-10-21, `prepare_data.py` rescales irradiance after the
change (factor 0.570, estimated from 2023 against 2022), so the 2023 Faulty
prevalence is back in line with earlier periods. The factor uses test-year
data; results before and after the change are still reported separately.

B2's unseen split is an inverter-ID holdout. It is not guaranteed to be a
whole-station holdout, and `sq14_inverter` is absent from the dataset. The
available test contains six matching held-out IDs. At test time, peer features
use contemporaneous campus telemetry, including available test inverters. This
matches a sequential monitoring system where current peer readings are present.

When class weights are compared with undersampling, the 2022 validation scores
guide selection. Validation scores are therefore model-selection evidence, while
the 2023 test estimates final performance. The small imputation and feature
selection comparisons use fixed samples and should not be read as significance
tests.

The synthetic-fault Isolation Forest check uses B2's injected 35% power drop
and complete outage. The healthy source rows were selected using B1's engineered
proxy. Detection results therefore measure the synthetic exercise only and do
not establish real-world fault detection ability.

## Other team scripts requiring follow-up

These findings come from code inspection, not from rerunning or modifying the
other members' experiments. They should be reviewed by the respective owners
before cross-model scores are put into one comparison table.

1. **B2 MLP (`src/models/mlp.py`):** `prepare_mlp_data()` creates a current
   health score from current `ac_power_w` and irradiance. `get_clean_features()`
   then includes those same current columns while targeting current health.
   This directly exposes the target formula. It also calculates missing-value
   means independently for each supplied partition. B2 should move to the
   exact one-hour target or a truly independent diagnostic target and fit an
   imputer inside the training/CV Pipeline.
   *Status 2026-10-21: fixed.* The MLP now uses the exact one-hour target,
   training-only mean imputation, and a random sample across all inverters.
2. **B1 target helpers (`src/targets.py`):** `compute_health_score()` replaces
   missing AC power with zero, which converts unavailable telemetry into an
   apparent low-health event. `create_forecasting_target()` shifts by four
   remaining rows without verifying a one-hour timestamp difference. The
   `prepare_leakage_free_matrices()` diagnostic allowlist removes raw power but
   can still admit derived current PR/peer-deviation features, and fills inputs
   with means from the entire passed frame. A2 uses local safeguards for each
   issue. B1 should fix the shared helpers or document a dataset contract that
   prevents their unsafe use.
   *Status 2026-10-21: partly fixed.* `compute_health_score()` now returns an
   undefined score for missing power. `create_forecasting_target()` and
   `prepare_leakage_free_matrices()` are unchanged but no model uses them any
   more; every model builds the exact-time lead itself.
3. **A1 baseline (`src/models/linear.py`):** the lead timestamp is checked
   correctly, and the train-only mean fill is directionally sound. However,
   it calls B1's score helper on missing-power rows, so outage gaps receive
   zero-health labels. Its prior scores use more rows and a different target
   treatment than A2; they cannot be compared directly. A2 computes its own
   persistence, current-status and majority baselines on identical rows.
   *Status 2026-10-21: fixed* through the shared `compute_health_score()`
   change; A1's baselines were re-run. Row counts still differ from A2's by
   about 1% because A2 applies its own physical cleaning.
4. **B1 feature standalone (`src/features.py`):** the standalone workflow builds
   features over all dates and then splits. Its PR/peer/history transforms are
   causal in time after A2's corrections, but B2's held-out inverters can
   contribute to pre-2023 peer medians in that standalone path. A2 computes
   peers after holdout filtering. B1 should adopt that order for a strict
   unseen-inverter evaluation.
   *Status 2026-10-21: open.* B1's XGBoost still builds features on the full
   grid, so held-out inverters contribute to campus peer medians. The effect
   is limited to one of about 40 inverters in each median, but it is a known
   deviation from a strict unseen-inverter test.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m src.eda_a2
.\.venv\Scripts\python.exe -m src.models.rf
.\.venv\Scripts\python.exe -m src.models.iforest_a2
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Do not rerun `src.preprocess` with `--skip-comparison` and then cite the existing
imputation CSV as a new result; that option leaves the prior comparison table
untouched.

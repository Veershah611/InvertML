# A2 exploratory analysis

Analysis uses B2's training period only (through 30 June 2022). Code and figure
generation are in `src/eda_a2.py`. The plots are `figures/a2_temperature_effect.png`
and `figures/a2_feature_correlation.png`.

Class distributions are also tabulated by split and season in
`tables/a2_class_distribution_by_season.csv`, with a training-only stacked plot
in `figures/a2_class_distribution_by_season.png`. Training Faulty prevalence
ranges from 4.2% in summer to 9.8% in winter. Before the irradiance
correction, 2023 rose to 54.7% in spring and 55.2% in summer, far beyond
ordinary seasonality, consistent with A1's documented February 2023 sensor
change. With irradiance rescaled after 2023-02-14 (`prepare_data.py`, factor
0.570), 2023 ranges from 6.2% in spring to 11.1% in autumn, within the range
seen in training and validation (updated 2026-10-21).

## Temperature and output

Raw AC power rises with sunlight, which also heats the surroundings. To reduce
that confounding, A2 examines performance ratio for midday rows (10:00–14:59)
with irradiance between 300 and 800 W/m². This leaves 118,468 training rows.
Five-degree temperature bins and per-inverter linear slopes are recorded in
`tables/a2_temperature_bins.csv` and `tables/a2_temperature_inverter_slopes.csv`.

The per-inverter temperature/PR slope is positive for 26 of 36 inverters and
negative for 10. The median is +0.0057 PR units per degree Celsius. The binned
median PR does not fall monotonically across the observed temperature range.
This is an **observational association**, not evidence that heat improves PV
performance: irradiance within the broad bin, cloud conditions, orientation,
season, and the target's assumed derating factor can still influence it.

Decision: retain `temp_c` and B1's thermal interaction as candidate predictors;
let validation decide their model value. Do not assert a universal heat derating
effect from this dataset. The engineered health score is clipped at 100, so PR
is the more informative response for this EDA question.

## Feature correlation

The Pearson matrix uses training inputs only. Seven pairs have |r| >= 0.85.
The strongest are:

| Pair | Pearson r |
|---|---:|
| `temp_c` / `temp_delta_stc` | 1.000 |
| `irradiance_wm2` / `thermal_stress` | 0.946 |
| 1-hour / 4-hour peer-deviation rolling mean | 0.922 |
| 1-hour / 4-hour PR rolling mean | 0.893 |

All pairs are in `tables/a2_feature_correlations.csv`. B1's correlation filter
suggests removing `irradiance_wm2` and `temp_delta_stc` while keeping protected
domain/history features. A fixed Random Forest validation check compared the
39-input full set against that 37-input set:

| Inputs | Regression RMSE ↓ | Classification macro-F1 ↑ |
|---|---:|---:|
| All 39 | 10.055 | 0.6386 |
| Pruned 37 | 10.064 | 0.6372 |

The difference is small: both tasks slightly favor all inputs on the
corrected target (re-run 2026-10-21; before, classification slightly favored
pruning). Decision: keep the full feature set for the
final Random Forest comparison and report the near-tie. The validation sample
is 25,000 training and 12,000 validation rows; these numbers are not a
statistical-significance result. Correlation does not imply that a feature is
useless, particularly for tree interactions.

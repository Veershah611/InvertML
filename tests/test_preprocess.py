"""Leakage and data-integrity checks for A2 preprocessing."""

import unittest

import numpy as np
import pandas as pd

from src.preprocess import (
    NUMERIC_FEATURES, build_preprocessor, clean_measurements, feature_frame,
    forecasting_rows, make_b2_splits, time_series_folds,
)
from src.features import build_b1_features


def example(times, inverter="known"):
    return pd.DataFrame({
        "time": pd.to_datetime(times), "inverter_id": inverter,
        "ac_power_w": 3900.0, "irradiance_wm2": 500.0, "capacity_kwp": 10.0,
        "temp_c": 25.0, "n_valid_5min": 3, "azimuth": "180deg",
    })


class PreprocessingTests(unittest.TestCase):
    def test_missing_and_invalid_power_never_become_fault_labels(self):
        raw = example(pd.date_range("2022-01-01 09:00", periods=6, freq="15min"))
        raw["ac_power_w"] = [np.nan, -1, 12000, 0, 3900, 3900]
        raw.loc[4, "irradiance_wm2"] = 1600
        raw.loc[5, "irradiance_wm2"] = 75
        df = clean_measurements(raw)
        self.assertTrue(df.loc[[0, 1, 2, 4, 5], "health_score"].isna().all())
        self.assertTrue(df.loc[[0, 1, 2, 4, 5], "status"].isna().all())
        self.assertEqual(df.loc[3, "status"], "Faulty")  # observed zero is valid
        self.assertTrue(df.loc[0, "is_offline"])
        self.assertTrue(df.loc[5, "is_daylight"])  # cleaning and label thresholds differ

    def test_physical_weather_bounds_and_missing_capacity(self):
        raw = example(["2022-01-01 09:00", "2022-01-01 09:15"])
        raw["rh_pct"] = [130.0, 50.0]
        raw["capacity_kwp"] = [10.0, np.nan]
        df = clean_measurements(raw)
        self.assertTrue(pd.isna(df.loc[0, "rh_pct"]))
        self.assertTrue(pd.isna(df.loc[1, "health_score"]))

    def test_forecast_is_exact_hour_not_next_available_rows(self):
        raw = example(["2022-01-01 16:30", "2022-01-01 17:30",
                       "2022-01-02 06:00", "2022-01-02 06:15",
                       "2022-01-02 06:30", "2022-01-02 06:45"])
        rows = forecasting_rows(clean_measurements(raw))
        self.assertEqual(len(rows), 1)
        self.assertTrue((rows["target_time"] - rows["time"]).eq(pd.Timedelta("1h")).all())

    def test_split_and_target_boundaries_and_holdout(self):
        dates = ["2022-06-30 22:30", "2022-06-30 23:30",
                 "2022-07-01 00:30", "2022-07-01 01:30",
                 "2022-12-31 23:30", "2023-01-01 00:30", "2023-01-01 01:30"]
        df = clean_measurements(pd.concat([
            example(dates), example(dates, "held-out"),
        ], ignore_index=True))
        splits = make_b2_splits(df, held_out_stations=["held_out"])
        for name in ["train", "val"]:
            self.assertEqual(set(splits[name]["inverter_id"]), {"known"})
        self.assertEqual(set(splits["test_unseen"]["inverter_id"]), {"held-out"})
        for name, end in [("train", "2022-07-01"), ("val", "2023-01-01")]:
            forecast = forecasting_rows(splits[name])
            self.assertTrue((forecast["target_time"] < pd.Timestamp(end)).all())
        self.assertEqual(len(forecasting_rows(splits["train"])), 1)

    def test_imputer_scaler_and_categories_fit_training_only(self):
        train = build_b1_features(clean_measurements(
            example(pd.date_range("2022-01-01 09:00", periods=3, freq="15min"))
        ))
        train["temp_c"] = [10.0, 20.0, np.nan]
        val = train.copy()
        val["temp_c"] = [1000.0, np.nan, 2000.0]
        val["azimuth"] = "never_seen"
        transformer = build_preprocessor(scale=True).fit(feature_frame(train))
        numeric = transformer.named_transformers_["numeric"]
        pos = NUMERIC_FEATURES.index("temp_c")
        self.assertEqual(numeric.named_steps["imputer"].statistics_[pos], 15.0)
        self.assertEqual(numeric.named_steps["scaler"].mean_[pos], 15.0)
        before = numeric.named_steps["imputer"].statistics_.copy()
        result = transformer.transform(feature_frame(val))
        np.testing.assert_equal(before, numeric.named_steps["imputer"].statistics_)
        self.assertTrue(np.isfinite(result).all())

    def test_cv_groups_timestamps_and_purges_future_labels(self):
        times = pd.date_range("2022-01-01 09:00", periods=40, freq="15min")
        df = pd.DataFrame({"time": np.repeat(times, 2)})
        df["target_time"] = df["time"] + pd.Timedelta("1h")
        for tr, va in time_series_folds(df):
            self.assertLess(df.iloc[tr]["target_time"].max(), df.iloc[va]["time"].min())
            self.assertFalse(set(df.iloc[tr]["time"]) & set(df.iloc[va]["time"]))
            self.assertTrue(df.iloc[va].groupby("time").size().eq(2).all())
        with self.assertRaisesRegex(ValueError, "Sort rows"):
            list(time_series_folds(df.iloc[::-1]))

    def test_duplicates_fail_and_input_is_unchanged(self):
        raw = example(["2022-01-01 09:00"])
        before = raw.copy(deep=True)
        clean_measurements(raw)
        pd.testing.assert_frame_equal(raw, before)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            clean_measurements(pd.concat([raw, raw], ignore_index=True))


if __name__ == "__main__":
    unittest.main()

"""Regression checks for B1's Stage 7 integration with A1/A2/B2."""

import importlib
import unittest

import numpy as np
import pandas as pd

from src.features import (
    B1_CORE_FEATURES, B2_ALL_FEATURES, add_performance_ratio_and_yield,
    add_self_relative_deviation_features, add_weather_interaction_features,
    build_b1_features, build_b2_features, prune_collinear_features,
)
from src.preprocess import clean_measurements, engineer_b2_partitions, make_b2_splits


def telemetry(times, inverter="known", power=3900.0):
    return pd.DataFrame({
        "time": pd.to_datetime(times), "inverter_id": inverter, "station_id": inverter,
        "ac_power_w": power, "irradiance_wm2": 500.0, "capacity_kwp": 10.0,
        "temp_c": 25.0, "n_valid_5min": 3, "azimuth": "180deg",
    })


class FeatureIntegrationTests(unittest.TestCase):
    def test_a1_and_b2_imports_and_old_feature_contract(self):
        for module in ("src.models.linear", "src.models.mlp"):
            importlib.import_module(module)
        df = build_b2_features(telemetry(["2022-06-30 09:00"]))
        self.assertTrue(set(B2_ALL_FEATURES).issubset(df.columns))

    def test_physical_features_preserve_missing_measurements(self):
        df = telemetry(pd.date_range("2022-01-01 09:00", periods=5, freq="15min"))
        df["ac_power_w"] = [np.nan, 0, 3900, 3900, 3900]
        df.loc[2, "capacity_kwp"] = np.nan
        df.loc[3, "irradiance_wm2"] = 50
        df.loc[4, "n_valid_5min"] = 0
        out = add_performance_ratio_and_yield(df)
        self.assertTrue(out.loc[[0, 2, 3, 4], "performance_ratio"].isna().all())
        self.assertEqual(out.loc[1, "performance_ratio"], 0)
        df.loc[0, "temp_c"] = np.nan
        weather = add_weather_interaction_features(df)
        self.assertTrue(pd.isna(weather.loc[0, "thermal_stress"]))
        self.assertTrue(pd.isna(weather.loc[0, "temp_delta_stc"]))

    def test_missing_peer_is_not_an_observed_zero(self):
        df = pd.concat([
            telemetry(["2022-01-01 09:00"], "a"),
            telemetry(["2022-01-01 09:00"], "b", np.nan),
        ], ignore_index=True)
        out = build_b1_features(df)
        self.assertEqual(out.loc[0, "peer_dev_campus"], 0)
        self.assertTrue(pd.isna(out.loc[1, "peer_dev_campus"]))

    def test_calendar_window_ignores_old_rows_and_current_observation(self):
        df = telemetry(["2022-01-01", "2022-01-20", "2022-02-10", "2022-04-01"])
        df["performance_ratio"] = [1.0, 2.0, 3.0, 4.0]
        df.index = [10, 20, 30, 40]
        shuffled = df.iloc[[2, 0, 3, 1]]
        out = add_self_relative_deviation_features(shuffled)
        self.assertEqual(list(out.index), list(shuffled.index))
        self.assertTrue(pd.isna(out.loc[10, "self_dev_30d"]))
        self.assertAlmostEqual(out.loc[20, "self_dev_30d"], 1 / 1.05, places=5)
        self.assertAlmostEqual(out.loc[30, "self_dev_30d"], 1 / 2.05, places=5)
        self.assertTrue(pd.isna(out.loc[40, "self_dev_30d"]))
        short = add_self_relative_deviation_features(df, lookback_days=1)
        self.assertTrue(short["self_dev_30d"].isna().all())

    def test_self_history_respects_irradiance_bin_and_future_invariance(self):
        df = telemetry(pd.date_range("2022-01-01 09:00", periods=4, freq="15min"))
        df["performance_ratio"] = [1.0, 2.0, 3.0, 4.0]
        df["irradiance_wm2"] = [200, 500, 200, 200]
        before = add_self_relative_deviation_features(df.iloc[:3])
        full = add_self_relative_deviation_features(df)
        pd.testing.assert_series_equal(before["self_dev_30d"], full.iloc[:3]["self_dev_30d"])
        self.assertTrue(pd.isna(full.loc[1, "self_dev_30d"]))
        self.assertAlmostEqual(full.loc[2, "self_dev_30d"], 2 / 1.05, places=5)

    def test_partition_features_isolate_holdout_and_carry_past_history(self):
        times = ["2022-06-30 09:00", "2022-06-30 10:00",
                 "2022-07-01 09:00", "2022-07-01 10:00",
                 "2023-01-01 09:00", "2023-01-01 10:00"]
        known = telemetry(times)
        held = telemetry(times, "held", power=7800.0)
        df = clean_measurements(pd.concat([known, held], ignore_index=True))
        partitions = make_b2_splits(df, held_out_stations=["held"])
        out = engineer_b2_partitions(partitions)
        self.assertTrue(out["train"]["peer_dev_campus"].eq(0).all())
        self.assertTrue(out["val"]["peer_dev_campus"].eq(0).all())
        self.assertEqual(out["val"].iloc[0]["self_dev_30d"], 0)
        # Old June/July data cannot become a baseline in January.
        self.assertTrue(pd.isna(out["test"].iloc[0]["self_dev_30d"]))
        test_held = out["test"].loc[out["test"]["inverter_id"].eq("held")].reset_index(drop=True)
        pd.testing.assert_frame_equal(out["test_unseen"].reset_index(drop=True), test_held)
        self.assertGreater(out["test_unseen"].iloc[0]["peer_dev_campus"], 0)
        self.assertTrue(set(B1_CORE_FEATURES).issubset(out["train"].columns))

    def test_pruning_keeps_protected_features(self):
        df = pd.DataFrame({"a": range(10), "b": range(10), "c": range(10)})
        selected, _ = prune_collinear_features(
            df, list(df), protected_features=["a", "b"], save_report=False
        )
        self.assertEqual(selected, ["a", "b"])

    def test_dropped_candidate_cannot_prune_another_feature(self):
        # a correlates with b and c; b and c do not exceed the threshold.
        rng = np.random.default_rng(42)
        b, c = rng.normal(size=(2, 2000))
        df = pd.DataFrame({"a": b + c, "b": b, "c": c})
        selected, log = prune_collinear_features(
            df, list(df), threshold=0.6, protected_features=["b"], save_report=False
        )
        self.assertEqual(selected, ["b", "c"])
        self.assertEqual(len(log), 1)


if __name__ == "__main__":
    unittest.main()

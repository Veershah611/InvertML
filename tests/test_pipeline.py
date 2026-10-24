"""Checks for the rules that must never break. Run: python -m unittest discover -s tests"""
import unittest

import numpy as np
import pandas as pd

from src.dataset import add_split, add_target, clean, health_score, past_rolling, status
from src.evaluate import classification_scores, regression_scores


def one_inverter(values, start="2022-01-03 10:00", inverter="inv_a"):
    times = pd.date_range(start, periods=len(values), freq="15min")
    return pd.DataFrame({"inverter_id": inverter, "time": times, "health_now": values})


class TestTarget(unittest.TestCase):
    def test_health_score(self):
        # 1000 W/m2 on 10 kWp with 0.78 derate -> expected 7800 W
        health = health_score(pd.Series([7800.0, 3900.0, 9000.0, 5000.0]),
                              pd.Series([1000.0, 1000.0, 1000.0, 50.0]),
                              pd.Series([10.0, 10.0, 10.0, 10.0]))
        self.assertAlmostEqual(health[0], 100)
        self.assertAlmostEqual(health[1], 50)
        self.assertEqual(health[2], 100)            # capped at 100
        self.assertTrue(np.isnan(health[3]))        # too dark to judge

    def test_status_thresholds(self):
        self.assertEqual(list(status(pd.Series([95, 90, 89.9, 70, 69.9]))),
                         ["Healthy", "Healthy", "Warning", "Warning", "Faulty"])

    def test_target_is_exactly_one_hour_later(self):
        df = one_inverter([10, 20, 30, 40, 50, 60])
        df = df.drop(index=4)                       # a gap: 11:00 is missing
        out = add_target(df.reset_index(drop=True))
        lead =dict(zip(out["time"].dt.strftime("%H:%M"), out["health_lead"]))
        self.assertTrue(np.isnan(lead["10:00"]))    # 11:00 missing -> no target, no shifting
        self.assertEqual(lead["10:15"], 60)         # 11:15 exists


class TestNoLeakage(unittest.TestCase):
    def test_rolling_uses_only_the_past(self):
        df = one_inverter([10.0, 20.0, 30.0, 40.0])
        mean = past_rolling(df, "health_now", "4h", "mean")
        self.assertTrue(np.isnan(mean[0]))          # nothing before the first row
        self.assertEqual(mean[1], 10)               # current value (20) excluded
        self.assertEqual(mean[3], 20)               # mean of 10, 20, 30

    def test_unseen_inverters_only_in_test(self):
        unseen = "sq_block_p_inverter"
        df = pd.concat([one_inverter([1, 2], "2022-01-03 10:00", unseen),
                        one_inverter([1, 2], "2023-03-01 10:00", unseen),
                        one_inverter([1, 2], "2022-01-03 10:00", "inv_a")])
        out = add_split(df)
        self.assertEqual(set(out.loc[out["inverter_id"] == unseen, "split"]), {"test"})


class TestCleaningAndMetrics(unittest.TestCase):
    def test_impossible_values_become_missing(self):
        df = pd.DataFrame({"ac_power_w": [-5.0, 20000.0, 5000.0], "capacity_kwp": [10.0] * 3,
                           "irradiance_wm2": [500.0, 2000.0, 500.0], "n_valid_5min": [3, 3, 0]})
        out = clean(df)
        self.assertTrue(out["ac_power_w"].isna().all())   # negative, too high, offline
        self.assertTrue(np.isnan(out.loc[1, "irradiance_wm2"]))
        self.assertEqual(list(out["is_offline"]), [1, 1, 1])

    def test_metrics(self):
        self.assertEqual(regression_scores([1, 2, 3], [1, 2, 3])["rmse"], 0)
        scores = classification_scores(["Faulty", "Faulty", "Healthy"], ["Faulty", "Healthy", "Healthy"])
        self.assertEqual(scores["faulty_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()

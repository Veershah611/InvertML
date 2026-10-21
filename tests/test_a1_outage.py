"""Outage feature checks: offline rule, 24 h past-only window, inverter isolation."""
import unittest
import numpy as np
import pandas as pd
from src.features import A1_OUTAGE_FEATURES, add_a1_outage_features


class OutageTests(unittest.TestCase):
    def test_offline_rule_and_past_only_window(self):
        times = pd.to_datetime([
            "2022-01-01 10:00", "2022-01-01 10:15", "2022-01-01 10:30",
            "2022-01-02 10:00", "2022-01-02 10:30",
        ])
        df = pd.DataFrame({"inverter_id": "a", "time": times,
                           "ac_power_w": [np.nan, 5., 5., np.nan, 5.],
                           "n_valid_5min": [0, 0, 3, 0, 3],
                           "irradiance_wm2": 400.})
        out = add_a1_outage_features(df)
        self.assertEqual(out["is_offline"].tolist(), [1, 1, 0, 1, 0])
        # The current bin is excluded, so the first row has no history.
        self.assertEqual(out.loc[0, "minutes_offline_24h"], 0.)
        self.assertEqual(out.loc[2, "minutes_offline_24h"], 30.)
        # 2022-01-02 10:00 is exactly 24 h after the first bin, which still counts.
        self.assertEqual(out.loc[3, "minutes_offline_24h"], 30.)
        # By 10:30 both earlier offline bins have left the window.
        self.assertEqual(out.loc[4, "minutes_offline_24h"], 15.)
        changed = df.copy()
        changed.loc[4, "ac_power_w"] = np.nan
        pd.testing.assert_frame_equal(out.iloc[:4], add_a1_outage_features(changed).iloc[:4])

    def test_group_isolation_and_row_order(self):
        t = pd.to_datetime(["2022-01-01 10:15", "2022-01-01 10:00"])
        a = pd.DataFrame({"inverter_id": "a", "time": t, "ac_power_w": [5., np.nan],
                          "irradiance_wm2": 400.})
        b = a.assign(inverter_id="b", ac_power_w=[5., 5.])
        out = add_a1_outage_features(pd.concat([a, b], ignore_index=True))
        self.assertEqual(out.loc[0, "minutes_offline_24h"], 15.)
        self.assertEqual(out.loc[2, "minutes_offline_24h"], 0.)
        self.assertEqual(A1_OUTAGE_FEATURES, ["is_offline", "minutes_offline_24h"])

    def test_dawn_bins_flagged_but_not_counted(self):
        t = pd.to_datetime(["2022-01-01 06:00", "2022-01-01 06:15", "2022-01-01 06:30"])
        df = pd.DataFrame({"inverter_id": "a", "time": t, "ac_power_w": [np.nan, np.nan, 5.],
                           "irradiance_wm2": [10., 80., 120.]})
        out = add_a1_outage_features(df)
        self.assertEqual(out["is_offline"].tolist(), [1, 1, 0])
        # Only the 06:15 bin is offline in daylight (>= 50 W/m2).
        self.assertEqual(out.loc[2, "minutes_offline_24h"], 15.)


if __name__ == "__main__":
    unittest.main()

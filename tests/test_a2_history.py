"""History checks that distinguish elapsed time from row count."""
import unittest
import numpy as np
import pandas as pd
from src.features import A2_HISTORY_FEATURES, add_a2_history_features


class HistoryTests(unittest.TestCase):
    def test_exact_lags_and_no_future_lookahead(self):
        times = pd.to_datetime([
            "2022-01-01 10:00", "2022-01-01 10:15", "2022-01-01 11:00",
            "2022-01-02 11:00", "2022-01-02 11:15",
        ])
        df = pd.DataFrame({"inverter_id": "a", "time": times,
                           "performance_ratio": [1., 2., 3., 4., 5.],
                           "peer_dev_campus": [0., .1, .2, .3, .4]})
        out = add_a2_history_features(df)
        self.assertTrue(np.isnan(out.loc[2, "performance_ratio_lag_15m"]))
        self.assertEqual(out.loc[2, "performance_ratio_lag_1h"], 1.)
        self.assertEqual(out.loc[3, "performance_ratio_lag_1d"], 3.)
        self.assertEqual(out.loc[2, "performance_ratio_roll_1h_mean"], 1.5)
        self.assertTrue(np.isnan(out.loc[3, "performance_ratio_roll_1h_mean"]))
        changed = df.copy()
        changed.loc[4, "performance_ratio"] = 999.
        pd.testing.assert_frame_equal(out.iloc[:4], add_a2_history_features(changed).iloc[:4])

    def test_group_isolation_and_missing_rows(self):
        t = pd.to_datetime(["2022-01-01 10:00", "2022-01-01 10:15"])
        a = pd.DataFrame({"inverter_id": "a", "time": t,
                          "performance_ratio": [1., np.nan], "peer_dev_campus": [0., .1]})
        b = a.assign(inverter_id="b", performance_ratio=[10., 20.])
        out = add_a2_history_features(pd.concat([a, b], ignore_index=True))
        self.assertEqual(out.loc[3, "performance_ratio_lag_15m"], 10.)
        self.assertEqual(out.loc[1, "performance_ratio_lag_15m"], 1.)
        self.assertEqual(len(A2_HISTORY_FEATURES), 18)


if __name__ == "__main__":
    unittest.main()

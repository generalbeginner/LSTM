import unittest

import numpy as np

from run_yfinance_lstm import (
    compare_pivots,
    construct_lstm_data,
    detect_pivots,
    inverse_target,
    validate_target,
)


class TestConfigurableTargets(unittest.TestCase):
    def test_accepts_supported_price_features(self):
        for target in ("Open", "Close", "High", "Low"):
            self.assertEqual(validate_target(target), target)

    def test_rejects_non_price_target(self):
        with self.assertRaises(ValueError):
            validate_target("Volume")

    def test_sequences_use_selected_target_column(self):
        data = np.arange(30, dtype=float).reshape(5, 6)
        _, y = construct_lstm_data(data, sequence_size=2, target_idx=3)
        np.testing.assert_array_equal(y, [15, 21, 27])

    def test_inverse_target_uses_scaler_column(self):
        class OffsetScaler:
            def inverse_transform(self, values):
                return values + np.arange(values.shape[1])

        values = inverse_target(OffsetScaler(), np.array([1.0, 2.0]), target_idx=2, n_features=6)
        np.testing.assert_array_equal(values, [3.0, 4.0])

    def test_detects_local_highs_and_lows(self):
        dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
        pivots = detect_pivots(dates, [10, 12, 9, 11, 8], pivot_order=1)
        self.assertEqual(list(pivots["pivot_type"]), ["high", "low", "high"])
        self.assertEqual(list(pivots["Date"].dt.strftime("%Y-%m-%d")), dates[1:4])

    def test_compares_predicted_pivots_to_nearest_true_pivots(self):
        predicted = detect_pivots(["2026-01-01", "2026-01-02", "2026-01-03"], [1, 3, 1], 1)
        true = detect_pivots(["2026-01-02", "2026-01-03", "2026-01-04"], [1, 4, 1], 1)
        compared = compare_pivots(predicted, true, max_date_gap=1)
        self.assertEqual(compared.loc[0, "predicted_date"].strftime("%Y-%m-%d"), "2026-01-02")
        self.assertEqual(compared.loc[0, "true_date"].strftime("%Y-%m-%d"), "2026-01-03")
        self.assertEqual(compared.loc[0, "date_error_days"], 1)


if __name__ == "__main__":
    unittest.main()

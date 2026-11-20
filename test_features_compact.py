import unittest
from math import log10

import numpy as np

from features_compact import (
    FEATURE_NAMES, N_FEATURES, extract_compact_features, higuchi_fd, katz_fd,
    permutation_entropy,
)

FS = 173.61


class CompactFeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.windows = rng.standard_normal((6, int(round(2 * FS)))) * 50

    def test_shape_and_finite(self):
        for seconds in (1.0, 2.0, 5.0):
            windows = np.random.default_rng(1).standard_normal((4, int(round(seconds * FS))))
            X = extract_compact_features(windows, FS)
            self.assertEqual(X.shape, (4, N_FEATURES))
            self.assertEqual(len(FEATURE_NAMES), N_FEATURES)
            self.assertTrue(np.isfinite(X).all())

    def test_amplitude_scaling_only_shifts_log_power(self):
        a = extract_compact_features(self.windows, FS)
        b = extract_compact_features(self.windows * 10, FS)
        log_columns = list(range(0, 5)) + list(range(10, 20)) + [31]
        np.testing.assert_allclose(b[:, log_columns] - a[:, log_columns],
                                   [[2 * np.log(10)] * 5 + [2 * np.log(10)] * 5 + [np.log(10)] * 5 + [2 * np.log(10)]] * 6,
                                   atol=1e-6)
        scale_free = [5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]
        np.testing.assert_allclose(b[:, scale_free], a[:, scale_free], atol=1e-8)

    def test_permutation_entropy_extremes(self):
        monotone = np.arange(200, dtype=float)[None, :]
        self.assertAlmostEqual(float(permutation_entropy(monotone)[0]), 0.0)
        noise = np.random.default_rng(2).standard_normal((1, 20000))
        self.assertGreater(float(permutation_entropy(noise)[0]), 0.98)

    def test_higuchi_reference_values(self):
        line = np.linspace(0, 1, 1000)[None, :]
        self.assertAlmostEqual(float(higuchi_fd(line)[0]), 1.0, places=2)
        noise = np.random.default_rng(3).standard_normal((1, 5000))
        self.assertAlmostEqual(float(higuchi_fd(noise)[0]), 2.0, delta=0.05)

    def test_katz_straight_line_is_one(self):
        line = np.linspace(0, 5, 300)[None, :]
        self.assertAlmostEqual(float(katz_fd(line)[0]), log10(299) / log10(299), places=6)

    def test_rejects_too_short_windows(self):
        with self.assertRaises(ValueError):
            extract_compact_features(np.zeros((2, 60)) + np.arange(60), FS)


if __name__ == "__main__":
    unittest.main()

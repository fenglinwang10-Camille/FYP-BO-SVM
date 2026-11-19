"""Checks for the paired training-only feature comparison."""
import unittest
import numpy as np
from compare_feature_deduplication import KEEP, evaluate
from features import extract_features_by_setting
from analyze_grouped_features import REFERENCE_PARAMS


class FeatureComparisonTests(unittest.TestCase):
    def test_removed_features_really_are_duplicates(self):
        rng = np.random.default_rng(7)
        windows = rng.normal(size=(4, 868))
        features = extract_features_by_setting(windows, 173.61, "nf_enhanced")
        self.assertEqual(features.shape, (4, 55))
        self.assertEqual(features[:, KEEP].shape, (4, 53))
        np.testing.assert_array_equal(features[:, 18], features[:, 32])
        np.testing.assert_allclose(features[:, 9], features[:, 15], atol=1e-12)

    def test_fold_overlap_is_rejected(self):
        with self.assertRaises(AssertionError):
            evaluate(np.zeros((4, 2)), np.array([0, 0, 1, 1]),
                     np.array([0, 0, 1, 1]),
                     [{"train_segment_ids": [0], "validation_segment_ids": [0, 1]}],
                     REFERENCE_PARAMS)


if __name__ == "__main__":
    unittest.main()

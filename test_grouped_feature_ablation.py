"""Checks for recording-level out-of-fold feature screening."""

import unittest
from unittest.mock import patch

import numpy as np

from analyze_grouped_features import _cross_validated_recordings


class LabelModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0].astype(int)


class FeatureAblationTests(unittest.TestCase):
    def setUp(self):
        recording_ids = np.arange(10)
        self.groups = np.repeat(recording_ids, 2)
        self.y = np.repeat(recording_ids % 5, 2)
        self.X = self.y.reshape(-1, 1)
        self.folds = [
            {
                "train_segment_ids": recording_ids[recording_ids // 2 != fold].tolist(),
                "validation_segment_ids": recording_ids[recording_ids // 2 == fold].tolist(),
            }
            for fold in range(5)
        ]

    def test_each_training_recording_is_predicted_once(self):
        with patch("analyze_grouped_features._make_model", return_value=LabelModel()):
            rows = _cross_validated_recordings(
                self.X, self.y, self.groups, self.folds, {}, 42, "original",
            )
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["segment_id"] for row in rows}, set(range(10)))
        self.assertTrue(all(row["true_label"] == row["predicted_label"] for row in rows))

    def test_repeated_validation_recording_is_rejected(self):
        self.folds[1]["validation_segment_ids"][0] = 0
        with patch("analyze_grouped_features._make_model", return_value=LabelModel()):
            with self.assertRaises(ValueError):
                _cross_validated_recordings(
                    self.X, self.y, self.groups, self.folds, {}, 42, "original",
                )


if __name__ == "__main__":
    unittest.main()

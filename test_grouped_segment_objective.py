"""Focused regression checks for the grouped segment-level objective."""

import unittest
from unittest.mock import patch

import numpy as np
from skopt import gp_minimize

from grouped_bo_optimizer import (
    SEARCH_SPACE, _cross_validate_training_segments, _initial_random_points,
    _objective_score, _recording_folds,
)


class FixedWindowPredictions:
    def fit(self, X, y):
        return self

    def predict(self, X):
        labels = X[:, 0].astype(int)
        positions = X[:, 1].astype(int)
        return np.where(positions == 1, 0, np.where(positions == 2, 1, labels))


class SegmentObjectiveTests(unittest.TestCase):
    def test_recording_folds_do_not_change_with_window_count(self):
        recording_labels = np.repeat(np.arange(5), 10)
        recording_ids = np.arange(len(recording_labels))

        def validation_sets(windows_per_recording):
            groups = np.repeat(recording_ids, windows_per_recording)
            labels = np.repeat(recording_labels, windows_per_recording)
            return [tuple(validation.tolist()) for _, validation in
                    _recording_folds(labels, groups, 42)]

        self.assertEqual(validation_sets(4), validation_sets(23))

    def test_resumed_initial_points_match_uninterrupted_run(self):
        uninterrupted = gp_minimize(
            lambda point: float(point[0]), SEARCH_SPACE,
            n_calls=16, n_initial_points=16, random_state=43, acq_func="EI",
        )
        planned = _initial_random_points(SEARCH_SPACE, 43)
        np.testing.assert_allclose(planned, uninterrupted.x_iters, rtol=0, atol=1e-12)

    def test_segment_accuracy_is_used_instead_of_window_accuracy(self):
        labels = np.repeat(np.arange(2).repeat(5), 3)
        groups = np.repeat(np.arange(10), 3)
        positions = np.tile(np.arange(3), 10)
        X = np.column_stack((labels, positions))
        params = {"f_low": 1, "f_high": 40, "window_size": 5, "C": 1, "gamma": 0.1}

        with patch("grouped_bo_optimizer._build_feature_matrix", return_value=(X, labels, groups)), patch(
            "grouped_bo_optimizer._make_model", return_value=FixedWindowPredictions()
        ):
            score, metrics, _, fold_counts = _cross_validate_training_segments(
                None, None, None, params, objective_mode="segment_accuracy",
            )

        self.assertEqual(len(fold_counts), 5)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["segment"]["accuracy"], 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_old_objectives_keep_window_metrics(self):
        metrics = {
            "accuracy": 0.6,
            "macro_recall": 0.5,
            "macro_f1": 0.4,
            "per_class_recall": {"N": 0.3, "F": 0.2},
            "segment": {"accuracy": 0.9},
        }
        self.assertAlmostEqual(_objective_score(metrics, "segment_accuracy"), 0.9)
        self.assertAlmostEqual(_objective_score(metrics, "macro_f1"), 0.4)
        self.assertAlmostEqual(_objective_score(metrics, "nf_focused"), 0.435)


if __name__ == "__main__":
    unittest.main()

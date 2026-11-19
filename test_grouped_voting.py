"""Tests for confidence-based resolution of recording vote ties."""

import unittest

import numpy as np

from analyze_grouped_voting import _recording_votes


class VotingTests(unittest.TestCase):
    def test_tie_uses_decision_scores_only_among_tied_classes(self):
        predictions = np.array([2, 2, 3, 3, 4, 4, 4, 3])
        groups = np.array([10] * 4 + [11] * 4)
        labels = np.array([3] * 4 + [4] * 4)
        scores = np.zeros((8, 5))
        scores[:4, 3] = 2.0
        scores[:4, 2] = 1.0
        scores[:4, 0] = 100.0
        rows = _recording_votes(predictions, scores, groups, labels)

        self.assertEqual(rows[0]["majority_prediction"], 2)
        self.assertEqual(rows[0]["confidence_tiebreak_prediction"], 3)
        self.assertTrue(rows[0]["tie"])
        self.assertEqual(rows[1]["majority_prediction"], 4)
        self.assertEqual(rows[1]["confidence_tiebreak_prediction"], 4)
        self.assertFalse(rows[1]["tie"])


if __name__ == "__main__":
    unittest.main()

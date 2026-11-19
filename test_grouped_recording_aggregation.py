import numpy as np

from analyze_grouped_recording_aggregation import (
    _aggregate_recordings,
    _cross_validated_predictions,
)


def test_aggregation_preserves_recordings_and_dimensions():
    X = np.array([[1., 2.], [3., 4.], [5., 7.], [9., 11.]])
    y = np.array([0, 0, 1, 1])
    groups = np.array([11, 11, 22, 22])
    means, labels, ids = _aggregate_recordings(X, y, groups, include_std=False)
    np.testing.assert_array_equal(ids, [11, 22])
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(means, [[2., 3.], [7., 9.]])
    combined, _, _ = _aggregate_recordings(X, y, groups, include_std=True)
    np.testing.assert_allclose(combined, [[2., 3., 1., 1.], [7., 9., 2., 2.]])


def test_aggregation_rejects_mixed_recording_labels():
    X = np.array([[1.], [2.]])
    y = np.array([0, 1])
    groups = np.array([11, 11])
    try:
        _aggregate_recordings(X, y, groups, include_std=False)
    except ValueError as exc:
        assert "inconsistent labels" in str(exc)
    else:
        raise AssertionError("Expected a mixed-label recording to be rejected")


def test_cv_predicts_each_training_recording_once():
    X = np.array([[0.], [0.1], [1.], [1.1], [0.2], [0.3], [1.2], [1.3]])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    folds = [
        {"train_segment_ids": [0, 1], "validation_segment_ids": [2, 3]},
        {"train_segment_ids": [2, 3], "validation_segment_ids": [0, 1]},
    ]
    params = {"C": 1.0, "gamma": 0.1}
    for method in ("majority_vote", "feature_mean", "feature_mean_std"):
        rows = _cross_validated_predictions(X, y, groups, folds, params, 42, method)
        assert len(rows) == 4
        assert {row["recording_id"] for row in rows} == {0, 1, 2, 3}

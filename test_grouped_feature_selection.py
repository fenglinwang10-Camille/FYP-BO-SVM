import numpy as np
from sklearn.feature_selection import SelectKBest

from analyze_grouped_feature_selection import _evaluate_count, _make_selected_model


def test_selector_is_inside_training_pipeline():
    model = _make_selected_model({"C": 1.0, "gamma": 0.1}, 2)
    assert list(model.named_steps) == ["scaler", "selection", "svm"]
    assert isinstance(model.named_steps["selection"], SelectKBest)


def test_fold_selection_does_not_use_validation_labels():
    X = np.array([
        [0., 0., 0.], [0.1, 0., 0.], [1., 0., 0.], [1.1, 0., 0.],
        [0., 0., 9.], [0.1, 0., 9.], [1., 0., -9.], [1.1, 0., -9.],
    ])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.arange(8)
    folds = [
        {"train_segment_ids": [0, 1, 2, 3], "validation_segment_ids": [4, 5, 6, 7]},
        {"train_segment_ids": [4, 5, 6, 7], "validation_segment_ids": [0, 1, 2, 3]},
    ]
    predictions, selections = _evaluate_count(
        X, y, groups, folds, {"C": 1.0, "gamma": 0.1}, 42, 1,
    )
    assert len(predictions) == 8
    assert len(selections) == 2
    assert selections[0]["selected_indices"] == "[0]"
    assert selections[1]["selected_indices"] == "[2]"

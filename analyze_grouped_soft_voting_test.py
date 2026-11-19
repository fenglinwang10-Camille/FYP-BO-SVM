"""Exploratory paired test of hard and soft voting on completed BO splits.

The held-out test sets have previously been inspected, so this analysis is
post-hoc and must not be presented as independent confirmatory evidence.
"""

import json

import numpy as np
import pandas as pd

from analyze_grouped_features import _saved_training_folds
from analyze_grouped_soft_voting import _recording_votes
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES, DATASET_PATH, FIXED_FOLD_DIR, _build_feature_matrix,
    _make_model, _metrics, _select_segments,
)


SEEDS = (42, 43, 44, 45, 46)
OUTPUT_DIR = FIXED_FOLD_DIR / "soft_voting_posthoc_test"


def run():
    signals, labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    labels = np.asarray(labels, dtype=int)
    per_seed = []
    predictions = []
    for seed in SEEDS:
        train_ids, _ = _saved_training_folds(seed, labels)
        test_ids = np.setdiff1d(np.arange(len(labels)), train_ids)
        source = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform" / "grouped_bo_final_result.json"
        saved = json.loads(source.read_text(encoding="utf-8"))
        params = saved["best_parameters"]
        train_signals, train_labels, train_groups = _select_segments(signals, labels, train_ids)
        test_signals, test_labels, test_groups = _select_segments(signals, labels, test_ids)
        X_train, y_train, groups_train = _build_feature_matrix(
            train_signals, train_labels, train_groups, params)
        X_test, y_test, groups_test = _build_feature_matrix(
            test_signals, test_labels, test_groups, params)
        if set(groups_train) & set(groups_test):
            raise ValueError("Outer train/test recording leakage")
        model = _make_model(params)
        model.named_steps["svm"].set_params(probability=True, random_state=seed)
        model.fit(X_train, y_train)
        if not np.array_equal(model.named_steps["svm"].classes_,
                              np.arange(len(CLASS_NAMES))):
            raise ValueError("Probability columns do not match class labels")
        rows = _recording_votes(model.predict(X_test), model.predict_proba(X_test),
                                groups_test, y_test)
        if len(rows) != len(test_ids) or {r["recording_id"] for r in rows} != set(test_ids):
            raise ValueError("Each held-out recording must be predicted exactly once")
        true = np.asarray([row["true_label"] for row in rows])
        hard = np.asarray([row["majority_prediction"] for row in rows])
        soft = np.asarray([row["soft_prediction"] for row in rows])
        hard_metrics = _metrics(true, hard)
        soft_metrics = _metrics(true, soft)
        expected = saved["segment_level_test_metrics"]
        if not np.isclose(hard_metrics["accuracy"], expected["accuracy"]):
            raise ValueError(f"Seed {seed} hard voting does not reproduce saved test result")
        per_seed.append({
            "seed": seed,
            "hard_accuracy": hard_metrics["accuracy"],
            "soft_accuracy": soft_metrics["accuracy"],
            "accuracy_difference": soft_metrics["accuracy"] - hard_metrics["accuracy"],
            "hard_recall_F": hard_metrics["per_class_recall"]["F"],
            "soft_recall_F": soft_metrics["per_class_recall"]["F"],
            "changed_predictions": int(np.sum(hard != soft)),
        })
        predictions.extend({"seed": seed, **row} for row in rows)
        print(f"Seed {seed}: hard={hard_metrics['accuracy']:.4f}, "
              f"soft={soft_metrics['accuracy']:.4f}, "
              f"F recall={soft_metrics['per_class_recall']['F']:.4f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(per_seed)
    frame.to_csv(OUTPUT_DIR / "per_seed_test.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUTPUT_DIR / "recording_predictions.csv", index=False)
    summary = {
        "hard_accuracy_mean": float(frame["hard_accuracy"].mean()),
        "soft_accuracy_mean": float(frame["soft_accuracy"].mean()),
        "hard_accuracy_sd": float(frame["hard_accuracy"].std()),
        "soft_accuracy_sd": float(frame["soft_accuracy"].std()),
        "paired_accuracy_difference_mean": float(frame["accuracy_difference"].mean()),
        "hard_recall_F_mean": float(frame["hard_recall_F"].mean()),
        "soft_recall_F_mean": float(frame["soft_recall_F"].mean()),
        "note": "Post-hoc exploratory comparison on previously inspected test sets. "
                "BO parameters selected for hard voting, not retuned for soft voting. "
                "SVC probability calibration uses internal window-level CV.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

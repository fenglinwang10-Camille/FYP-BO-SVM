"""Compare majority-vote tie rules on completed BO training folds only."""

import json

import numpy as np
import pandas as pd

from analyze_grouped_features import _saved_training_folds
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES,
    DATASET_PATH,
    FIXED_FOLD_DIR,
    _build_feature_matrix,
    _make_model,
    _metrics,
    _select_segments,
)


OUTPUT_DIR = FIXED_FOLD_DIR / "voting_tie_break_training_cv"
SEEDS = (42, 43, 44, 45, 46)


def _recording_votes(window_predictions, decision_scores, groups, labels):
    rows = []
    for recording_id in np.unique(groups):
        mask = groups == recording_id
        true_classes = np.unique(labels[mask])
        if len(true_classes) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent labels")
        counts = np.bincount(window_predictions[mask], minlength=len(CLASS_NAMES))
        tied = np.flatnonzero(counts == counts.max())
        original = int(tied[0])
        if len(tied) > 1:
            average_scores = decision_scores[mask][:, tied].mean(axis=0)
            confidence = int(tied[np.argmax(average_scores)])
        else:
            confidence = original
        rows.append({
            "recording_id": int(recording_id),
            "true_label": int(true_classes[0]),
            "majority_prediction": original,
            "confidence_tiebreak_prediction": confidence,
            "tie": len(tied) > 1,
        })
    return rows


def run():
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    per_seed = []
    all_predictions = []
    for seed in SEEDS:
        train_ids, folds = _saved_training_folds(seed, raw_labels)
        path = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform" / "grouped_bo_final_result.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("cv_split_protocol") != "recording_fixed_v2":
            raise ValueError(f"Seed {seed} is not a fixed-fold BO result")
        params = saved["best_parameters"]
        signals, labels, recording_ids = _select_segments(raw_signals, raw_labels, train_ids)
        X, y, groups = _build_feature_matrix(signals, labels, recording_ids, params)
        predictions = []
        for fold in folds:
            train_mask = np.isin(groups, fold["train_segment_ids"])
            validation_mask = np.isin(groups, fold["validation_segment_ids"])
            model = _make_model(params)
            model.fit(X[train_mask], y[train_mask])
            window_predictions = model.predict(X[validation_mask])
            decision_scores = model.decision_function(X[validation_mask])
            if (decision_scores.shape[1] != len(CLASS_NAMES)
                    or not np.array_equal(model.named_steps["svm"].classes_,
                                          np.arange(len(CLASS_NAMES)))):
                raise ValueError("SVM decision scores do not align with class labels")
            predictions.extend(_recording_votes(
                window_predictions, decision_scores, groups[validation_mask],
                y[validation_mask],
            ))
        if (len(predictions) != len(train_ids)
                or {row["recording_id"] for row in predictions} != set(train_ids.tolist())):
            raise ValueError(f"Seed {seed} did not predict each training recording once")
        true = np.asarray([row["true_label"] for row in predictions], dtype=int)
        majority = np.asarray([row["majority_prediction"] for row in predictions], dtype=int)
        confidence = np.asarray(
            [row["confidence_tiebreak_prediction"] for row in predictions], dtype=int,
        )
        original_metrics = _metrics(true, majority)
        if not np.isclose(original_metrics["accuracy"], saved["best_validation_score"]):
            raise ValueError(f"Seed {seed} majority CV did not reproduce saved BO score")
        alternate_metrics = _metrics(true, confidence)
        per_seed.append({
            "seed": seed,
            "majority_accuracy": original_metrics["accuracy"],
            "confidence_tiebreak_accuracy": alternate_metrics["accuracy"],
            "accuracy_difference": alternate_metrics["accuracy"] - original_metrics["accuracy"],
            "majority_F_recall": original_metrics["per_class_recall"]["F"],
            "confidence_tiebreak_F_recall": alternate_metrics["per_class_recall"]["F"],
            "tied_recordings": sum(row["tie"] for row in predictions),
            "changed_predictions": int(np.sum(majority != confidence)),
        })
        all_predictions.extend({"seed": seed, **row} for row in predictions)
        print(f"Seed {seed}: majority={original_metrics['accuracy']:.4f}, "
              f"confidence tie-break={alternate_metrics['accuracy']:.4f}, "
              f"ties={per_seed[-1]['tied_recordings']}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(per_seed)
    frame.to_csv(OUTPUT_DIR / "per_seed_training_cv.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(OUTPUT_DIR / "training_cv_recording_predictions.csv", index=False)
    summary = {
        "n_outer_training_splits": len(SEEDS),
        "majority_accuracy_mean": float(frame["majority_accuracy"].mean()),
        "confidence_tiebreak_accuracy_mean": float(frame["confidence_tiebreak_accuracy"].mean()),
        "accuracy_difference_mean": float(frame["accuracy_difference"].mean()),
        "majority_F_recall_mean": float(frame["majority_F_recall"].mean()),
        "confidence_tiebreak_F_recall_mean": float(frame["confidence_tiebreak_F_recall"].mean()),
        "total_tied_recordings": int(frame["tied_recordings"].sum()),
        "total_changed_predictions": int(frame["changed_predictions"].sum()),
        "held_out_test_evaluations": 0,
        "note": "Training CV only; BO parameters were selected with the original majority rule.",
    }
    (OUTPUT_DIR / "summary_training_cv.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

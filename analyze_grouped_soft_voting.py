"""Compare hard and probability-mean votes on saved training folds only."""

import json

import numpy as np
import pandas as pd

from analyze_grouped_features import REFERENCE_PARAMS, _saved_training_folds
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES, DATASET_PATH, FIXED_FOLD_DIR, _build_feature_matrix,
    _make_model, _metrics, _select_segments,
)


OUTPUT_DIR = FIXED_FOLD_DIR / "soft_voting_fixed_config_training_cv"
SEEDS = (42, 43, 44, 45, 46)


def _recording_votes(window_predictions, probabilities, groups, labels):
    rows = []
    for recording_id in np.unique(groups):
        mask = groups == recording_id
        true = np.unique(labels[mask])
        if len(true) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent labels")
        counts = np.bincount(window_predictions[mask], minlength=len(CLASS_NAMES))
        rows.append({
            "recording_id": int(recording_id),
            "true_label": int(true[0]),
            "majority_prediction": int(np.argmax(counts)),
            "soft_prediction": int(np.argmax(probabilities[mask].mean(axis=0))),
            "window_count": int(mask.sum()),
        })
    return rows


def run():
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    per_seed = []
    all_predictions = []
    for seed in SEEDS:
        train_ids, folds = _saved_training_folds(seed, raw_labels)
        signals, labels, ids = _select_segments(raw_signals, raw_labels, train_ids)
        X, y, groups = _build_feature_matrix(signals, labels, ids, REFERENCE_PARAMS)
        predictions = []
        for fold in folds:
            train = np.isin(groups, fold["train_segment_ids"])
            validation = np.isin(groups, fold["validation_segment_ids"])
            if not train.any() or not validation.any() or np.any(train & validation):
                raise ValueError("Invalid grouped fold")
            model = _make_model(REFERENCE_PARAMS)
            model.named_steps["svm"].set_params(probability=True, random_state=seed)
            model.fit(X[train], y[train])
            classes = model.named_steps["svm"].classes_
            if not np.array_equal(classes, np.arange(len(CLASS_NAMES))):
                raise ValueError("Probability columns do not match class labels")
            predictions.extend(_recording_votes(
                model.predict(X[validation]), model.predict_proba(X[validation]),
                groups[validation], y[validation],
            ))
        if (len(predictions) != len(train_ids)
                or {row["recording_id"] for row in predictions} != set(train_ids.tolist())):
            raise ValueError(f"Seed {seed} did not predict each training recording once")
        true = np.asarray([row["true_label"] for row in predictions], dtype=int)
        majority = np.asarray([row["majority_prediction"] for row in predictions], dtype=int)
        soft = np.asarray([row["soft_prediction"] for row in predictions], dtype=int)
        hard_metrics = _metrics(true, majority)
        soft_metrics = _metrics(true, soft)
        per_seed.append({
            "seed": seed,
            "majority_accuracy": hard_metrics["accuracy"],
            "soft_accuracy": soft_metrics["accuracy"],
            "accuracy_difference": soft_metrics["accuracy"] - hard_metrics["accuracy"],
            "majority_F_recall": hard_metrics["per_class_recall"]["F"],
            "soft_F_recall": soft_metrics["per_class_recall"]["F"],
            "changed_predictions": int(np.sum(majority != soft)),
        })
        all_predictions.extend({"seed": seed, **row} for row in predictions)
        print(f"Seed {seed}: majority={hard_metrics['accuracy']:.4f}, "
              f"soft={soft_metrics['accuracy']:.4f}, "
              f"F recall={soft_metrics['per_class_recall']['F']:.4f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(per_seed)
    frame.to_csv(OUTPUT_DIR / "per_seed_training_cv.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(
        OUTPUT_DIR / "training_cv_recording_predictions.csv", index=False)
    summary = {
        "n_outer_training_splits": len(SEEDS),
        "majority_accuracy_mean": float(frame["majority_accuracy"].mean()),
        "soft_accuracy_mean": float(frame["soft_accuracy"].mean()),
        "accuracy_difference_mean": float(frame["accuracy_difference"].mean()),
        "majority_F_recall_mean": float(frame["majority_F_recall"].mean()),
        "soft_F_recall_mean": float(frame["soft_F_recall"].mean()),
        "total_changed_predictions": int(frame["changed_predictions"].sum()),
        "held_out_test_evaluations": 0,
        "parameters": REFERENCE_PARAMS,
        "note": "Fixed configuration; training CV only. SVC probability calibration "
                "uses its internal CV on training windows, not grouped calibration.",
    }
    (OUTPUT_DIR / "summary_training_cv.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

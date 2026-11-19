"""Compare recording representations on saved training folds only.

This fixed-configuration screen does not access the outer held-out test set
or retune BO. Window features are averaged per original recording, optionally
with their within-recording standard deviations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_grouped_features import REFERENCE_PARAMS, _saved_training_folds
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES,
    DATASET_PATH,
    FIXED_FOLD_DIR,
    _build_feature_matrix,
    _make_model,
    _majority_vote_by_segment,
    _metrics,
    _select_segments,
)


METHODS = ("majority_vote", "feature_mean", "feature_mean_std")
OUTPUT_DIR = FIXED_FOLD_DIR / "recording_aggregation_fixed_config"


def _aggregate_recordings(X, y, groups, include_std):
    recording_ids = np.unique(groups)
    feature_rows = []
    labels = []
    for recording_id in recording_ids:
        mask = groups == recording_id
        recording_labels = np.unique(y[mask])
        if len(recording_labels) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent labels")
        values = X[mask]
        mean = values.mean(axis=0)
        row = np.concatenate((mean, values.std(axis=0, ddof=0))) if include_std else mean
        feature_rows.append(row)
        labels.append(int(recording_labels[0]))
    return np.vstack(feature_rows), np.asarray(labels, dtype=int), recording_ids


def _cross_validated_predictions(X, y, groups, folds, params, seed, method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == "majority_vote":
        features, labels, ids = X, y, groups
    else:
        features, labels, ids = _aggregate_recordings(
            X, y, groups, include_std=(method == "feature_mean_std"),
        )

    rows = []
    seen = set()
    for fold in folds:
        train_ids = np.asarray(fold["train_segment_ids"], dtype=int)
        validation_ids = np.asarray(fold["validation_segment_ids"], dtype=int)
        train_mask = np.isin(ids, train_ids)
        validation_mask = np.isin(ids, validation_ids)
        if (not train_mask.any() or not validation_mask.any()
                or np.intersect1d(train_ids, validation_ids).size):
            raise ValueError(f"Invalid fold in seed {seed}")
        model = _make_model(params)
        model.fit(features[train_mask], labels[train_mask])
        predictions = model.predict(features[validation_mask])
        if method == "majority_vote":
            true, pred = _majority_vote_by_segment(
                predictions, labels[validation_mask], ids[validation_mask],
            )
            predicted_ids = np.unique(ids[validation_mask])
        else:
            true, pred = labels[validation_mask], predictions
            predicted_ids = ids[validation_mask]
        for recording_id, true_label, predicted_label in zip(predicted_ids, true, pred):
            if int(recording_id) in seen:
                raise ValueError(f"Recording {recording_id} predicted more than once")
            seen.add(int(recording_id))
            rows.append({
                "seed": seed,
                "method": method,
                "recording_id": int(recording_id),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
            })
    if seen != set(np.unique(groups).tolist()):
        raise ValueError(f"Seed {seed} did not predict every training recording")
    return rows


def run(seeds=(42, 43, 44, 45, 46), output_dir=OUTPUT_DIR):
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    output_dir = Path(output_dir)
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    per_seed_rows = []
    prediction_rows = []
    for seed in seeds:
        train_ids, folds = _saved_training_folds(seed, raw_labels)
        signals, labels, recording_ids = _select_segments(raw_signals, raw_labels, train_ids)
        X, y, groups = _build_feature_matrix(signals, labels, recording_ids, REFERENCE_PARAMS)
        if X.shape[1] != 55 or not np.isfinite(X).all():
            raise ValueError(f"Unexpected feature matrix for seed {seed}: {X.shape}")
        for method in METHODS:
            predictions = _cross_validated_predictions(
                X, y, groups, folds, REFERENCE_PARAMS, seed, method,
            )
            prediction_rows.extend(predictions)
            true = np.asarray([row["true_label"] for row in predictions], dtype=int)
            pred = np.asarray([row["predicted_label"] for row in predictions], dtype=int)
            metrics = _metrics(true, pred)
            per_seed_rows.append({
                "seed": seed,
                "method": method,
                "feature_columns": 110 if method == "feature_mean_std" else 55,
                "cv_recording_accuracy": metrics["accuracy"],
                "cv_recording_macro_f1": metrics["macro_f1"],
                **{f"cv_recall_{name}": metrics["per_class_recall"][name]
                   for name in CLASS_NAMES},
            })
            print(f"Seed {seed} {method}: accuracy={metrics['accuracy']:.4f}, "
                  f"F recall={metrics['per_class_recall']['F']:.4f}", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    summary = per_seed.groupby("method", sort=False).agg(
        n_outer_training_splits=("seed", "size"),
        feature_columns=("feature_columns", "first"),
        cv_recording_accuracy_mean=("cv_recording_accuracy", "mean"),
        cv_recording_accuracy_sd=("cv_recording_accuracy", "std"),
        cv_recording_macro_f1_mean=("cv_recording_macro_f1", "mean"),
        cv_recall_F_mean=("cv_recall_F", "mean"),
        cv_recall_F_sd=("cv_recall_F", "std"),
    ).reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_training_cv.csv", index=False)
    summary.to_csv(output_dir / "summary_training_cv.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "training_cv_recording_predictions.csv", index=False,
    )
    (output_dir / "protocol.json").write_text(json.dumps({
        "purpose": "Exploratory recording representation comparison on training CV only",
        "methods": list(METHODS),
        "feature_setting": "nf_enhanced",
        "fixed_parameters": REFERENCE_PARAMS,
        "cv_split_protocol": "recording_fixed_v2",
        "seeds": list(seeds),
        "held_out_test_evaluations": 0,
        "note": "Same fixed C and gamma for all methods; no BO retuning. "
                "Outer training sets overlap, so SD is descriptive.",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    run(args.seeds)

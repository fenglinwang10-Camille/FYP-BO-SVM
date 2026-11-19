"""Training-CV-only feature selection ablation on saved recording folds.

This fixed-configuration screen never reads held-out test recordings. It is
exploratory; selecting a feature count here requires a separate outer test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

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


OUTPUT_DIR = FIXED_FOLD_DIR / "feature_selection_fixed_config"
FEATURE_COUNTS = (55, 35, 20)


def _make_selected_model(params, n_features):
    if not 1 <= n_features <= 55:
        raise ValueError("n_features must be between 1 and 55")
    base = _make_model(params)
    if n_features == 55:
        return base
    return Pipeline([
        ("scaler", base.named_steps["scaler"]),
        ("selection", SelectKBest(score_func=f_classif, k=n_features)),
        ("svm", base.named_steps["svm"]),
    ])


def _evaluate_count(X, y, groups, folds, params, seed, n_features):
    predictions = []
    selections = []
    seen = set()
    for fold_number, fold in enumerate(folds, start=1):
        train_ids = np.asarray(fold["train_segment_ids"], dtype=int)
        validation_ids = np.asarray(fold["validation_segment_ids"], dtype=int)
        train_mask = np.isin(groups, train_ids)
        validation_mask = np.isin(groups, validation_ids)
        if (not train_mask.any() or not validation_mask.any()
                or np.intersect1d(train_ids, validation_ids).size):
            raise ValueError(f"Invalid recording fold {fold_number} in seed {seed}")

        model = _make_selected_model(params, n_features)
        model.fit(X[train_mask], y[train_mask])
        indices = (model.named_steps["selection"].get_support(indices=True)
                   if n_features < X.shape[1] else np.arange(X.shape[1]))
        selections.append({
            "seed": seed,
            "n_features": n_features,
            "fold": fold_number,
            "selected_indices": json.dumps(indices.tolist()),
        })
        window_predictions = model.predict(X[validation_mask])
        recording_true, recording_pred = _majority_vote_by_segment(
            window_predictions, y[validation_mask], groups[validation_mask],
        )
        for recording_id, true, pred in zip(
            np.unique(groups[validation_mask]), recording_true, recording_pred,
        ):
            if int(recording_id) in seen:
                raise ValueError(f"Recording {recording_id} appears in multiple folds")
            seen.add(int(recording_id))
            predictions.append({
                "seed": seed,
                "n_features": n_features,
                "segment_id": int(recording_id),
                "true_label": int(true),
                "predicted_label": int(pred),
            })
    if seen != set(np.unique(groups).tolist()):
        raise ValueError(f"Seed {seed} did not predict every training recording once")
    return predictions, selections


def run(seeds=(42, 43, 44, 45, 46), output_dir=OUTPUT_DIR):
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    output_dir = Path(output_dir)
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    metric_rows = []
    prediction_rows = []
    selection_rows = []
    for seed in seeds:
        train_ids, folds = _saved_training_folds(seed, raw_labels)
        signals, labels, recording_ids = _select_segments(raw_signals, raw_labels, train_ids)
        X, y, groups = _build_feature_matrix(
            signals, labels, recording_ids, REFERENCE_PARAMS,
        )
        if X.shape[1] != 55 or not np.isfinite(X).all():
            raise ValueError(f"Unexpected feature matrix in seed {seed}: {X.shape}")
        for n_features in FEATURE_COUNTS:
            predictions, selections = _evaluate_count(
                X, y, groups, folds, REFERENCE_PARAMS, seed, n_features,
            )
            prediction_rows.extend(predictions)
            selection_rows.extend(selections)
            true = np.asarray([row["true_label"] for row in predictions])
            pred = np.asarray([row["predicted_label"] for row in predictions])
            metrics = _metrics(true, pred)
            metric_rows.append({
                "seed": seed,
                "n_features": n_features,
                "cv_segment_accuracy": metrics["accuracy"],
                "cv_segment_macro_f1": metrics["macro_f1"],
                **{f"cv_recall_{name}": metrics["per_class_recall"][name]
                   for name in CLASS_NAMES},
            })
            print(f"Seed {seed}, k={n_features}: accuracy={metrics['accuracy']:.4f}, "
                  f"F recall={metrics['per_class_recall']['F']:.4f}", flush=True)

    per_seed = pd.DataFrame(metric_rows)
    summary = per_seed.groupby("n_features", sort=False).agg(
        n_outer_training_splits=("seed", "size"),
        cv_segment_accuracy_mean=("cv_segment_accuracy", "mean"),
        cv_segment_accuracy_sd=("cv_segment_accuracy", "std"),
        cv_segment_macro_f1_mean=("cv_segment_macro_f1", "mean"),
        cv_recall_F_mean=("cv_recall_F", "mean"),
        cv_recall_F_sd=("cv_recall_F", "std"),
    ).reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_training_cv.csv", index=False)
    summary.to_csv(output_dir / "summary_training_cv.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "training_cv_recording_predictions.csv", index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "selected_feature_indices.csv", index=False,
    )
    (output_dir / "protocol.json").write_text(json.dumps({
        "purpose": "Exploratory fixed-configuration, training-CV-only feature selection",
        "objective": "recording-level accuracy after window majority vote",
        "cv_split_protocol": "recording_fixed_v2",
        "seeds": list(seeds),
        "feature_setting": "nf_enhanced",
        "feature_counts": list(FEATURE_COUNTS),
        "selection": "SelectKBest(f_classif) after scaling, fit within each training fold",
        "reference_parameters": REFERENCE_PARAMS,
        "held_out_test_evaluations": 0,
        "note": "Outer training sets overlap; SD is descriptive. No BO retuning.",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    run(args.seeds)

"""Screen existing feature sets on fixed recording folds without test scoring.

This is a fixed-configuration training-CV ablation, not a new BO run or an
unbiased estimate of a selected model's held-out performance.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_bonn_dataset
from features import extract_features_by_setting
from grouped_bo_optimizer import (
    CLASS_NAMES,
    DATASET_PATH,
    FIXED_FOLD_DIR,
    FS,
    _make_model,
    _majority_vote_by_segment,
    _metrics,
    _recording_folds,
    _segment_with_groups,
    _select_segments,
    _split_original_segments,
)
from preprocessing import bandpass_filter_signals


OUTPUT_DIR = FIXED_FOLD_DIR / "feature_ablation_fixed_config"
FEATURE_SETTINGS = ("original", "entropy_hjorth", "expanded", "nf_enhanced")
EXPECTED_COLUMNS = (23, 29, 41, 55)
REFERENCE_PARAMS = {
    "f_low": 1.0,
    "f_high": 40.0,
    "window_size": 5.0,
    "C": 10.0,
    "gamma": 0.01,
}


def _saved_training_folds(seed, raw_labels):
    run_dir = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform"
    split = json.loads((run_dir / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    folds = json.loads((run_dir / "grouped_cv_folds.json").read_text(encoding="utf-8"))
    expected_train, expected_test = _split_original_segments(raw_labels, seed)
    train_ids = np.asarray(split["train_segment_ids"], dtype=int)
    test_ids = np.asarray(split["test_segment_ids"], dtype=int)
    if not (np.array_equal(train_ids, expected_train)
            and np.array_equal(test_ids, expected_test)):
        raise ValueError(f"Seed {seed} outer split differs from the completed BO run")
    expected_folds = list(_recording_folds(raw_labels[train_ids], train_ids, seed))
    if len(folds) != len(expected_folds):
        raise ValueError(f"Seed {seed} has a different CV fold count")
    for saved, (train_fold, validation_fold) in zip(folds, expected_folds):
        if (not np.array_equal(saved["train_segment_ids"], train_fold)
                or not np.array_equal(saved["validation_segment_ids"], validation_fold)):
            raise ValueError(f"Seed {seed} CV folds differ from the completed BO run")
    return train_ids, folds


def _cross_validated_recordings(X, y, groups, folds, params, seed, setting):
    predictions = []
    seen = set()
    for fold in folds:
        train_ids = np.asarray(fold["train_segment_ids"], dtype=int)
        validation_ids = np.asarray(fold["validation_segment_ids"], dtype=int)
        train_mask = np.isin(groups, train_ids)
        validation_mask = np.isin(groups, validation_ids)
        if (not train_mask.any() or not validation_mask.any()
                or np.intersect1d(train_ids, validation_ids).size):
            raise ValueError(f"Invalid fold in seed {seed}")
        model = _make_model(params)
        model.fit(X[train_mask], y[train_mask])
        window_predictions = model.predict(X[validation_mask])
        recording_true, recording_pred = _majority_vote_by_segment(
            window_predictions, y[validation_mask], groups[validation_mask],
        )
        for segment_id, true, pred in zip(
            np.unique(groups[validation_mask]), recording_true, recording_pred,
        ):
            if int(segment_id) in seen:
                raise ValueError(f"Recording {segment_id} appears in multiple folds")
            seen.add(int(segment_id))
            predictions.append({
                "seed": seed,
                "feature_setting": setting,
                "segment_id": int(segment_id),
                "true_class": CLASS_NAMES[int(true)],
                "predicted_class": CLASS_NAMES[int(pred)],
                "true_label": int(true),
                "predicted_label": int(pred),
            })
    if seen != set(np.unique(groups).tolist()):
        raise ValueError(f"Seed {seed} CV did not predict every training recording once")
    return predictions


def _plot_f_errors(predictions, output_dir):
    frame = predictions[predictions["true_class"] == "F"]
    counts = (
        frame.groupby(["feature_setting", "predicted_class"]).size()
        .unstack(fill_value=0)
        .reindex(index=FEATURE_SETTINGS, columns=CLASS_NAMES, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(FEATURE_SETTINGS))
    bottom = np.zeros(len(FEATURE_SETTINGS), dtype=int)
    palette = {"Z": "#557a95", "O": "#97b7a7", "N": "#d67a51",
               "F": "#29806f", "S": "#8c6ba8"}
    for name in CLASS_NAMES:
        values = counts[name].to_numpy(dtype=int)
        ax.bar(x, values, bottom=bottom, label=f"Predicted {name}", color=palette[name])
        bottom += values
    ax.set_xticks(x, FEATURE_SETTINGS, rotation=12)
    ax.set_ylabel("F recordings across five training-CV splits")
    ax.set_ylim(0, int(bottom.max()) * 1.12)
    ax.legend(frameon=False, ncol=5, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "f_class_cv_predictions.png", dpi=220)
    plt.close(fig)
    counts.to_csv(output_dir / "f_class_cv_prediction_counts.csv")


def run(seeds=(42, 43, 44, 45, 46), output_dir=OUTPUT_DIR):
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    output_dir = Path(output_dir)
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    rows = []
    oof_rows = []
    for seed in seeds:
        train_ids, folds = _saved_training_folds(seed, raw_labels)
        signals, labels, recording_ids = _select_segments(raw_signals, raw_labels, train_ids)
        filtered = bandpass_filter_signals(
            signals, FS, REFERENCE_PARAMS["f_low"], REFERENCE_PARAMS["f_high"],
        )
        windows, y, groups = _segment_with_groups(
            filtered, labels, recording_ids, REFERENCE_PARAMS["window_size"],
        )
        for setting, expected_columns in zip(FEATURE_SETTINGS, EXPECTED_COLUMNS):
            X = extract_features_by_setting(windows, FS, setting)
            if X.shape != (len(y), expected_columns) or not np.isfinite(X).all():
                raise ValueError(f"Unexpected feature matrix for {setting}: {X.shape}")
            predicted = _cross_validated_recordings(
                X, y, groups, folds, REFERENCE_PARAMS, seed, setting,
            )
            oof_rows.extend(predicted)
            true = np.asarray([row["true_label"] for row in predicted], dtype=int)
            pred = np.asarray([row["predicted_label"] for row in predicted], dtype=int)
            metrics = _metrics(true, pred)
            rows.append({
                "seed": seed,
                "feature_setting": setting,
                "feature_columns": expected_columns,
                "cv_segment_accuracy": metrics["accuracy"],
                "cv_segment_macro_f1": metrics["macro_f1"],
                **{f"cv_recall_{name}": metrics["per_class_recall"][name]
                   for name in CLASS_NAMES},
            })
            print(f"Seed {seed} {setting}: CV segment accuracy="
                  f"{metrics['accuracy']:.4f}, F recall="
                  f"{metrics['per_class_recall']['F']:.4f}", flush=True)

    per_seed = pd.DataFrame(rows)
    predictions = pd.DataFrame(oof_rows)
    summary_rows = []
    for setting, group in per_seed.groupby("feature_setting", sort=False):
        summary_rows.append({
            "feature_setting": setting,
            "feature_columns": int(group["feature_columns"].iloc[0]),
            "n_outer_training_splits": len(group),
            **{f"{name}_{stat}": float(getattr(group[name], stat)())
               for name in ("cv_segment_accuracy", "cv_segment_macro_f1", "cv_recall_F")
               for stat in ("mean", "std")},
        })
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_training_cv.csv", index=False)
    summary.to_csv(output_dir / "summary_training_cv.csv", index=False)
    predictions.to_csv(output_dir / "training_cv_recording_predictions.csv", index=False)
    _plot_f_errors(predictions, output_dir)
    metadata = {
        "purpose": "Exploratory fixed-configuration feature ablation and F-class error analysis",
        "objective": "recording-level accuracy after window majority vote",
        "cv_split_protocol": "recording_fixed_v2",
        "seeds": list(seeds),
        "feature_settings": list(FEATURE_SETTINGS),
        "reference_parameters": REFERENCE_PARAMS,
        "held_out_test_evaluations": 0,
        "note": "Outer training sets overlap; SD is descriptive. Feature settings were not BO-tuned.",
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Saved training-CV-only analysis in {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    run(args.seeds)

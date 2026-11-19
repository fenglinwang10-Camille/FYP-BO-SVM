"""Fair grouped comparison of Manual SVM, Random Search, and BO-SVM.

All methods share the same original-segment train/test split, grouped CV folds,
55-feature extraction, fold-local scaling, SVM settings, validation objective,
and final one-time held-out test evaluation. BO and Random Search each use 50
candidate evaluations. Manual SVM evaluates one pre-specified configuration.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES,
    CLASS_WEIGHT,
    CV_FOLDS,
    DATASET_PATH,
    FEATURE_SETTING,
    FIXED_FOLD_DIR,
    RANDOM_STATE,
    WINDOW_OVERLAP,
    _cross_validate_training_segments,
    _final_test_once,
    _json_metrics,
    _recording_folds,
    _select_segments,
    _split_original_segments,
    _save_split_manifest,
)


N_CALLS = 50
RESULTS_DIR = FIXED_FOLD_DIR / "method_comparison_seed_42"
MANUAL_PARAMS = {
    "f_low": 0.5,
    "f_high": 40.0,
    "window_size": 2.0,
    "C": 10.0,
    "gamma": 0.1,
}


def _load_shared_data(seed, results_dir):
    raw_signals, raw_labels = load_bonn_dataset(
        DATASET_PATH,
        classification="five-class",
    )
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids = _split_original_segments(raw_labels, seed)
    train_data = _select_segments(raw_signals, raw_labels, train_ids)
    test_data = _select_segments(raw_signals, raw_labels, test_ids)
    if set(train_ids.tolist()).intersection(set(test_ids.tolist())):
        raise RuntimeError("Shared outer split contains overlapping segments")
    bo_dir = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform"
    split = json.loads((bo_dir / "grouped_split_manifest.json").read_text(encoding="utf-8"))
    if (split["random_state"] != seed
            or split["train_segment_ids"] != train_ids.tolist()
            or split["test_segment_ids"] != test_ids.tolist()):
        raise ValueError(f"Outer split differs from saved BO split for seed {seed}")
    folds = [
        {"fold": index, "train_segment_ids": train.tolist(),
         "validation_segment_ids": validation.tolist()}
        for index, (train, validation) in enumerate(
            _recording_folds(raw_labels[train_ids], train_ids, seed), start=1
        )
    ]
    saved_folds = json.loads((bo_dir / "grouped_cv_folds.json").read_text(encoding="utf-8"))
    if folds != saved_folds:
        raise ValueError(f"CV folds differ from saved BO folds for seed {seed}")
    _save_split_manifest(train_ids, test_ids, raw_labels, results_dir, seed)
    _save_json(results_dir / "grouped_cv_folds.json", folds)
    return train_data, test_data


def _result_payload(method, params, validation_score, validation_metrics,
                    window_metrics, segment_metrics, test_evaluations=1):
    return {
        "method": method,
        "search_evaluations": 1 if method == "Manual SVM" else N_CALLS,
        "validation_score": float(validation_score),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_macro_recall": float(validation_metrics["macro_recall"]),
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_segment_accuracy": float(validation_metrics["segment"]["accuracy"]),
        "validation_segment_macro_f1": float(validation_metrics["segment"]["macro_f1"]),
        "validation_per_class_recall": validation_metrics["per_class_recall"],
        "parameters": {key: float(value) for key, value in params.items()},
        "class_weight": CLASS_WEIGHT,
        "feature_setting": FEATURE_SETTING,
        "feature_columns": 55,
        "window_overlap": WINDOW_OVERLAP,
        "cv_folds": CV_FOLDS,
        "window_level_test_metrics": _json_metrics(window_metrics),
        "segment_level_test_metrics": _json_metrics(segment_metrics),
        "test_evaluations": int(test_evaluations),
    }


def _save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_manual(train_data, test_data):
    train_signals, train_labels, train_ids = train_data
    test_signals, test_labels, test_ids = test_data
    print("\nRunning Manual SVM under the shared grouped protocol...")
    score, validation_metrics, _, _ = _cross_validate_training_segments(
        train_signals,
        train_labels,
        train_ids,
        MANUAL_PARAMS,
        objective_mode="segment_accuracy",
    )
    window_metrics, segment_metrics, _, _ = _final_test_once(
        train_signals,
        train_labels,
        train_ids,
        test_signals,
        test_labels,
        test_ids,
        MANUAL_PARAMS,
    )
    payload = _result_payload(
        "Manual SVM",
        MANUAL_PARAMS,
        score,
        validation_metrics,
        window_metrics,
        segment_metrics,
    )
    _save_json(RESULTS_DIR / "grouped_manual_final_result.json", payload)
    print("Manual grouped validation accuracy:", round(validation_metrics["accuracy"], 4))
    print("Manual window-level test accuracy:", round(window_metrics["accuracy"], 4))
    print("Manual segment-level test accuracy:", round(segment_metrics["accuracy"], 4))
    return payload


def _sample_params(rng):
    return {
        "f_low": float(rng.uniform(0.5, 5.0)),
        "f_high": float(rng.uniform(30.0, 60.0)),
        "window_size": float(rng.uniform(1.0, 5.0)),
        "C": float(np.exp(rng.uniform(np.log(0.1), np.log(100.0)))),
        "gamma": float(np.exp(rng.uniform(np.log(1e-4), np.log(1.0)))),
    }


def run_random_search(train_data, test_data, *, seed=RANDOM_STATE, results_dir=RESULTS_DIR):
    train_signals, train_labels, train_ids = train_data
    test_signals, test_labels, test_ids = test_data
    print("\nRunning 50-evaluation Random Search under the shared grouped protocol...")
    rng = np.random.default_rng(seed)
    history = []
    running_path = results_dir / "grouped_random_search_history_running.csv"

    for iteration in range(1, N_CALLS + 1):
        params = _sample_params(rng)
        score, metrics, feature_shape, _ = _cross_validate_training_segments(
            train_signals,
            train_labels,
            train_ids,
            params,
            objective_mode="segment_accuracy",
            random_state=seed,
        )
        row = {
            "iteration": iteration,
            "score": score,
            **params,
            "class_weight": CLASS_WEIGHT,
            "feature_setting": FEATURE_SETTING,
            "window_overlap": WINDOW_OVERLAP,
            "cv_folds": CV_FOLDS,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_recall": metrics["macro_recall"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_segment_accuracy": metrics["segment"]["accuracy"],
            "validation_segment_macro_f1": metrics["segment"]["macro_f1"],
            "feature_rows": int(feature_shape[0]),
            "feature_columns": int(feature_shape[1]),
        }
        for class_name in CLASS_NAMES:
            row[f"validation_recall_{class_name}"] = metrics["per_class_recall"][
                class_name
            ]
        history.append(row)
        pd.DataFrame(history).to_csv(running_path, index=False)
        print(
            f"Grouped RS {iteration:02d}/{N_CALLS} | score={score:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"recall_N={metrics['per_class_recall']['N']:.4f} | "
            f"recall_F={metrics['per_class_recall']['F']:.4f}"
        )

    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    best_params = {
        name: float(best_row[name])
        for name in ["f_low", "f_high", "window_size", "C", "gamma"]
    }
    validation_metrics = {
        "accuracy": float(best_row["validation_accuracy"]),
        "macro_recall": float(best_row["validation_macro_recall"]),
        "macro_f1": float(best_row["validation_macro_f1"]),
        "segment": {
            "accuracy": float(best_row["validation_segment_accuracy"]),
            "macro_f1": float(best_row["validation_segment_macro_f1"]),
        },
        "per_class_recall": {
            name: float(best_row[f"validation_recall_{name}"])
            for name in CLASS_NAMES
        },
    }

    # First and only test evaluation for Random Search.
    window_metrics, segment_metrics, _, _ = _final_test_once(
        train_signals,
        train_labels,
        train_ids,
        test_signals,
        test_labels,
        test_ids,
        best_params,
    )
    payload = _result_payload(
        "Random Search SVM",
        best_params,
        float(best_row["score"]),
        validation_metrics,
        window_metrics,
        segment_metrics,
    )
    payload["best_iteration"] = int(best_row["iteration"])
    payload["random_state"] = seed
    payload["objective_mode"] = "segment_accuracy"
    payload["cv_split_protocol"] = "recording_fixed_v2"
    payload["parameter_prior"] = "uniform"
    history_df.to_csv(results_dir / "grouped_random_search_history.csv", index=False)
    _save_json(results_dir / "grouped_random_search_final_result.json", payload)
    print("Random Search best iteration:", payload["best_iteration"])
    print("Random Search window-level test accuracy:", round(window_metrics["accuracy"], 4))
    print("Random Search segment-level test accuracy:", round(segment_metrics["accuracy"], 4))
    print("Random Search held-out test evaluations: 1")
    return payload


def _load_bo_result(seed=RANDOM_STATE):
    path = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform" / "grouped_bo_final_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("objective_mode") != "segment_accuracy"
            or payload.get("cv_split_protocol") != "recording_fixed_v2"
            or payload.get("random_state") != seed
            or payload.get("cv_folds") != CV_FOLDS
            or payload.get("feature_setting") != FEATURE_SETTING
            or payload.get("feature_columns") != 55
            or payload.get("window_overlap") != WINDOW_OVERLAP
            or payload.get("class_weight") != CLASS_WEIGHT
            or payload.get("parameter_prior") != "uniform"
            or payload.get("bo_evaluations_used") != N_CALLS
            or payload.get("test_evaluations") != 1):
        raise ValueError(f"BO result does not match the fixed-fold comparison: {path}")
    return {
        "method": "BO-SVM",
        "search_evaluations": int(payload["bo_evaluations_used"]),
        "validation_score": float(payload["best_validation_score"]),
        "validation_accuracy": float(payload["best_validation_accuracy"]),
        "validation_macro_f1": float(payload["best_validation_macro_f1"]),
        "validation_segment_accuracy": float(payload["best_validation_segment_accuracy"]),
        "validation_segment_macro_f1": float(payload["best_validation_segment_macro_f1"]),
        "parameters": payload["best_parameters"],
        "window_level_test_metrics": payload["window_level_test_metrics"],
        "segment_level_test_metrics": payload["segment_level_test_metrics"],
        "test_evaluations": int(payload["test_evaluations"]),
    }


def _summary_row(payload):
    window = payload["window_level_test_metrics"]
    segment = payload["segment_level_test_metrics"]
    return {
        "method": payload["method"],
        "search_evaluations": payload["search_evaluations"],
        "validation_score": payload["validation_score"],
        "validation_accuracy": payload["validation_accuracy"],
        "validation_macro_f1": payload["validation_macro_f1"],
        "validation_segment_accuracy": payload["validation_segment_accuracy"],
        "validation_segment_macro_f1": payload["validation_segment_macro_f1"],
        "window_test_accuracy": window["accuracy"],
        "window_test_macro_recall": window["macro_recall"],
        "window_test_macro_f1": window["macro_f1"],
        "window_test_seizure_sensitivity": window["per_class_recall"]["S"],
        "segment_test_accuracy": segment["accuracy"],
        "segment_test_macro_recall": segment["macro_recall"],
        "segment_test_macro_f1": segment["macro_f1"],
        "segment_test_seizure_sensitivity": segment["per_class_recall"]["S"],
        "test_evaluations": payload["test_evaluations"],
        "best_parameters": json.dumps(payload["parameters"]),
    }


def run_comparison(seed=RANDOM_STATE):
    bo = _load_bo_result(seed)
    results_dir = FIXED_FOLD_DIR / f"method_comparison_seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_data, test_data = _load_shared_data(seed, results_dir)
    rows = []
    if seed == RANDOM_STATE:
        rows.append(_summary_row(run_manual(train_data, test_data)))
    random_search = run_random_search(train_data, test_data, seed=seed, results_dir=results_dir)
    rows.extend([_summary_row(random_search), _summary_row(bo)])
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "grouped_method_comparison.csv", index=False)
    print("\nFair grouped comparison completed:")
    print(
        summary[
            [
                "method",
                "search_evaluations",
                "window_test_accuracy",
                "window_test_macro_recall",
                "window_test_seizure_sensitivity",
                "segment_test_accuracy",
            ]
        ].to_string(index=False)
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    run_comparison(parser.parse_args().seed)

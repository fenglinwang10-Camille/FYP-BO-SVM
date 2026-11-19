"""Matched-budget BO for mean-pooled recording features on saved Bonn splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import gp_minimize

from analyze_grouped_features import _saved_training_folds
from analyze_grouped_recording_aggregation import _aggregate_recordings
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    DATASET_PATH, FIXED_FOLD_DIR, N_CALLS, N_INITIAL_POINTS, SEARCH_SPACE,
    _build_feature_matrix, _initial_random_points, _make_model, _metrics,
    _select_segments,
)


OUTPUT_DIR = FIXED_FOLD_DIR / "recording_mean_bo"


def _recording_features(signals, labels, ids, params, cache=None):
    key = tuple(float(params[name]) for name in ("f_low", "f_high", "window_size"))
    if cache is not None and key in cache:
        return cache[key]
    X, y, groups = _build_feature_matrix(signals, labels, ids, params)
    result = _aggregate_recordings(X, y, groups, include_std=False)
    if cache is not None:
        cache[key] = result
    return result


def _cv_score(features, labels, ids, folds, params):
    predictions = []
    for fold in folds:
        train = np.isin(ids, fold["train_segment_ids"])
        validation = np.isin(ids, fold["validation_segment_ids"])
        if not train.any() or not validation.any() or np.any(train & validation):
            raise ValueError("Invalid recording-level CV fold")
        model = _make_model(params)
        model.fit(features[train], labels[train])
        predictions.extend(zip(ids[validation], labels[validation], model.predict(features[validation])))
    if len(predictions) != len(ids) or len({int(row[0]) for row in predictions}) != len(ids):
        raise ValueError("Each training recording must be predicted exactly once")
    true = np.asarray([row[1] for row in predictions], dtype=int)
    pred = np.asarray([row[2] for row in predictions], dtype=int)
    return _metrics(true, pred)


def run_seed(seed, raw_signals, raw_labels, output_dir=OUTPUT_DIR):
    train_ids, folds = _saved_training_folds(seed, raw_labels)
    test_ids = np.setdiff1d(np.arange(len(raw_labels)), train_ids)
    train_signals, train_labels, train_groups = _select_segments(raw_signals, raw_labels, train_ids)
    directory = Path(output_dir) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "final_result.json"
    if final_path.exists():
        saved = json.loads(final_path.read_text(encoding="utf-8"))
        if saved.get("bo_evaluations_used") == N_CALLS:
            print(f"Seed {seed}: already complete", flush=True)
            return saved
        raise ValueError(f"Unexpected existing result: {final_path}")

    history_path = directory / "bo_history_running.csv"
    history = []
    if history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")
        if len(history) >= N_CALLS:
            raise ValueError("Complete history exists without final result; inspect before resuming")
    x0 = [[row[name] for name in ("f_low", "f_high", "window_size", "C", "gamma")]
          for row in history]
    y0 = [-row["cv_recording_accuracy"] for row in history]
    feature_cache = {}

    def objective(point):
        params = dict(zip(("f_low", "f_high", "window_size", "C", "gamma"),
                          map(float, point)))
        features, labels, ids = _recording_features(
            train_signals, train_labels, train_groups, params, feature_cache)
        metrics = _cv_score(features, labels, ids, folds, params)
        row = {"iteration": len(history) + 1, **params,
               "cv_recording_accuracy": metrics["accuracy"],
               "cv_recording_macro_f1": metrics["macro_f1"],
               "cv_recall_F": metrics["per_class_recall"]["F"]}
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Seed {seed} mean BO {len(history):02d}/{N_CALLS}: "
              f"CV accuracy={metrics['accuracy']:.4f}", flush=True)
        return -metrics["accuracy"]

    if history and len(history) < N_INITIAL_POINTS:
        initial = _initial_random_points(SEARCH_SPACE, seed)
        if not np.allclose(np.asarray(x0), np.asarray(initial[:len(history)]),
                           rtol=0, atol=1e-12):
            raise ValueError("Saved initial candidates do not match the seed")
        for point in initial[len(history):]:
            objective(point)
        x0 = initial
        y0 = [-row["cv_recording_accuracy"] for row in history]

    gp_minimize(objective, SEARCH_SPACE, n_calls=N_CALLS - len(history),
                n_initial_points=max(0, N_INITIAL_POINTS - len(history)),
                random_state=seed, acq_func="EI", x0=x0 or None, y0=y0 or None)
    best = max(history, key=lambda row: row["cv_recording_accuracy"])
    params = {name: float(best[name]) for name in ("f_low", "f_high", "window_size", "C", "gamma")}
    test_signals, test_labels, test_groups = _select_segments(raw_signals, raw_labels, test_ids)
    X_train, y_train, ids_train = _recording_features(
        train_signals, train_labels, train_groups, params, feature_cache)
    X_test, y_test, ids_test = _recording_features(
        test_signals, test_labels, test_groups, params)
    if set(ids_train) & set(ids_test):
        raise ValueError("Outer recording leakage")
    model = _make_model(params)
    model.fit(X_train, y_train)
    test_metrics = _metrics(y_test, model.predict(X_test))
    result = {
        "seed": seed, "method": "feature_mean", "parameter_prior": "uniform",
        "objective": "cv_recording_accuracy", "bo_evaluations_used": N_CALLS,
        "n_initial_points": N_INITIAL_POINTS, "best_iteration": int(best["iteration"]),
        "best_cv_recording_accuracy": float(best["cv_recording_accuracy"]),
        "best_parameters": params, "train_recordings": len(y_train),
        "test_recordings": len(y_test), "feature_columns": X_train.shape[1],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_class_recall": test_metrics["per_class_recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        "test_evaluations": 1,
    }
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(directory / "bo_history.csv", index=False)
    print(f"Seed {seed} held-out test accuracy={test_metrics['accuracy']:.4f}", flush=True)
    return result


def run(seeds=(42, 43, 44, 45, 46)):
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    raw_labels = np.asarray(raw_labels, dtype=int)
    results = [run_seed(seed, raw_signals, raw_labels) for seed in seeds]
    if len(results) == 5:
        baseline = []
        for seed in seeds:
            path = FIXED_FOLD_DIR / f"seed_{seed}" / "uniform" / "grouped_bo_final_result.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            baseline.append({"seed": seed, "method": "majority_vote",
                             "test_accuracy": data["segment_level_test_metrics"]["accuracy"],
                             "test_recall_F": data["segment_level_test_metrics"]["per_class_recall"]["F"]})
        rows = baseline + [{"seed": row["seed"], "method": "feature_mean",
                            "test_accuracy": row["test_accuracy"],
                            "test_recall_F": row["test_class_recall"]["F"]} for row in results]
        frame = pd.DataFrame(rows)
        frame.to_csv(OUTPUT_DIR / "matched_comparison.csv", index=False)
        frame.groupby("method").agg(
            accuracy_mean=("test_accuracy", "mean"), accuracy_sd=("test_accuracy", "std"),
            recall_F_mean=("test_recall_F", "mean"),
        ).to_csv(OUTPUT_DIR / "matched_summary.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    run(parser.parse_args().seeds)

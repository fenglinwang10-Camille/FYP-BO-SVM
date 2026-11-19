"""Leakage-controlled grouped Bayesian optimisation for five-class Bonn EEG.

The outer test set is split at the original-recording level before any
windowing. Bayesian optimisation sees training recordings only and uses
StratifiedGroupKFold, so windows from one original EEG recording never cross
folds. The held-out test recordings are evaluated exactly once after the best
hyperparameters have been selected.
"""

from pathlib import Path
import argparse
import json
from importlib.util import find_spec

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import Optimizer, gp_minimize
from skopt.space import Real

from data_loader import load_bonn_dataset
from features import extract_features_by_setting
from parameter_priors import (
    CDF_EPSILON,
    PREPROCESSING_BOUNDS,
    SUPPORTED_PRIORS,
    parameter_to_latent,
    transform_preprocessing_parameters,
)
from preprocessing import bandpass_filter_signals


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "04_grouped_bo_svm"
OBJECTIVE_COMPARISON_DIR = PROJECT_DIR / "results" / "05_grouped_objective_comparison"
SEGMENT_OBJECTIVE_DIR = PROJECT_DIR / "results" / "07_grouped_segment_objective"
FIXED_FOLD_DIR = PROJECT_DIR / "results" / "09_fixed_fold_segment_accuracy"

FS = 173.61
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_CALLS = 50
N_INITIAL_POINTS = 16
CLASS_WEIGHT = "balanced"
FEATURE_SETTING = "nf_enhanced"
WINDOW_OVERLAP = 0.0
CLASS_NAMES = ["Z", "O", "N", "F", "S"]
CLASS_LABELS = np.arange(len(CLASS_NAMES), dtype=int)
OBJECTIVE_MODES = ("nf_focused", "macro_f1", "segment_accuracy")

SEARCH_SPACE = [
    Real(0.5, 5.0, name="f_low"),
    Real(30.0, 60.0, name="f_high"),
    Real(1.0, 5.0, name="window_size"),
    Real(0.1, 100.0, prior="log-uniform", name="C"),
    Real(1e-4, 1.0, prior="log-uniform", name="gamma"),
]


def _initial_random_points(search_space, random_state):
    """Recreate gp_minimize's initial candidate sequence for a resumed run."""
    optimizer = Optimizer(
        search_space, n_initial_points=N_INITIAL_POINTS,
        random_state=random_state, acq_func="EI",
    )
    points = []
    for index in range(N_INITIAL_POINTS):
        point = optimizer.ask()
        points.append(point)
        if index + 1 < N_INITIAL_POINTS:
            optimizer.tell(point, 0.0)
    return points


def _split_original_segments(raw_labels, random_state=RANDOM_STATE):
    """Create one fixed stratified 80/20 split of original EEG segments."""
    segment_ids = np.arange(len(raw_labels), dtype=int)
    train_ids, test_ids = train_test_split(
        segment_ids,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=raw_labels,
    )
    return np.sort(train_ids), np.sort(test_ids)


def _select_segments(raw_signals, raw_labels, segment_ids):
    """Select original recordings while retaining stable segment IDs."""
    signals = [raw_signals[index] for index in segment_ids]
    labels = np.asarray(raw_labels, dtype=int)[segment_ids]
    return signals, labels, np.asarray(segment_ids, dtype=int)


def _segment_with_groups(signals, labels, segment_ids, window_size):
    """Window complete filtered segments and propagate original segment IDs."""
    samples_per_window = int(round(float(window_size) * FS))
    if samples_per_window <= 0:
        raise ValueError("window_size produces no samples")
    step = max(1, int(round(samples_per_window * (1.0 - WINDOW_OVERLAP))))

    windows = []
    window_labels = []
    window_groups = []
    for signal, label, segment_id in zip(signals, labels, segment_ids):
        signal = np.asarray(signal, dtype=float)
        for start in range(0, len(signal) - samples_per_window + 1, step):
            windows.append(signal[start : start + samples_per_window])
            window_labels.append(int(label))
            window_groups.append(int(segment_id))

    if not windows:
        raise ValueError("No windows were created")
    return (
        np.asarray(windows, dtype=float),
        np.asarray(window_labels, dtype=int),
        np.asarray(window_groups, dtype=int),
    )


def _build_feature_matrix(signals, labels, segment_ids, params):
    """Filter each complete segment, then window and extract features."""
    filtered = bandpass_filter_signals(
        signals,
        FS,
        float(params["f_low"]),
        float(params["f_high"]),
    )
    windows, y, groups = _segment_with_groups(
        filtered,
        labels,
        segment_ids,
        float(params["window_size"]),
    )
    X = extract_features_by_setting(windows, FS, FEATURE_SETTING)
    if X.shape[1] != 55:
        raise RuntimeError(
            f"Expected 55 {FEATURE_SETTING} features, received {X.shape[1]}. "
            "Check that PyWavelets is available."
        )
    return X, y, groups


def _make_model(params):
    """Fit scaling and SVM as one leakage-safe pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(params["C"]),
                    gamma=float(params["gamma"]),
                    class_weight=CLASS_WEIGHT,
                ),
            ),
        ]
    )


def _metrics(y_true, y_pred):
    recalls = recall_score(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            name: float(value) for name, value in zip(CLASS_NAMES, recalls)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
        ),
    }


def _objective_score(metrics, objective_mode="segment_accuracy"):
    if objective_mode == "segment_accuracy":
        return float(metrics["segment"]["accuracy"])
    if objective_mode == "macro_f1":
        return float(metrics["macro_f1"])
    if objective_mode == "nf_focused":
        return float(
            0.35 * metrics["accuracy"]
            + 0.25 * metrics["macro_recall"]
            + 0.20 * metrics["per_class_recall"]["N"]
            + 0.20 * metrics["per_class_recall"]["F"]
        )
    raise ValueError(f"Unknown objective_mode: {objective_mode}")


def _recording_folds(y, groups, random_state):
    """Assign original recordings to folds independently of window count."""
    recording_ids, first_indices = np.unique(groups, return_index=True)
    recording_labels = y[first_indices]
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=random_state,
    )
    for train_indices, validation_indices in splitter.split(
        np.zeros((len(recording_ids), 1)), recording_labels, recording_ids,
    ):
        yield recording_ids[train_indices], recording_ids[validation_indices]


def _cross_validate_training_segments(
    signals, labels, segment_ids, params, objective_mode="segment_accuracy",
    random_state=RANDOM_STATE, feature_cache=None,
):
    """Evaluate one candidate without accessing held-out test segments."""
    feature_key = tuple(float(params[name]) for name in ("f_low", "f_high", "window_size"))
    if feature_cache is not None and feature_key in feature_cache:
        X, y, groups = feature_cache[feature_key]
    else:
        X, y, groups = _build_feature_matrix(signals, labels, segment_ids, params)
        if feature_cache is not None:
            feature_cache[feature_key] = (X, y, groups)
    fold_metrics = []
    fold_segment_metrics = []
    fold_segment_counts = []
    for train_ids, validation_ids in _recording_folds(y, groups, random_state):
        train_index = np.flatnonzero(np.isin(groups, train_ids))
        validation_index = np.flatnonzero(np.isin(groups, validation_ids))
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups.intersection(validation_groups):
            raise RuntimeError("Grouped CV leakage detected")

        model = _make_model(params)
        model.fit(X[train_index], y[train_index])
        predictions = model.predict(X[validation_index])
        fold_metrics.append(_metrics(y[validation_index], predictions))
        if objective_mode == "segment_accuracy":
            segment_true, segment_predictions = _majority_vote_by_segment(
                predictions, y[validation_index], groups[validation_index],
            )
            fold_segment_metrics.append(_metrics(segment_true, segment_predictions))
        fold_segment_counts.append(
            {
                "train": len(train_groups),
                "validation": len(validation_groups),
            }
        )

    averaged = {
        "accuracy": float(np.mean([item["accuracy"] for item in fold_metrics])),
        "macro_recall": float(
            np.mean([item["macro_recall"] for item in fold_metrics])
        ),
        "macro_f1": float(np.mean([item["macro_f1"] for item in fold_metrics])),
        "per_class_recall": {
            name: float(
                np.mean([item["per_class_recall"][name] for item in fold_metrics])
            )
            for name in CLASS_NAMES
        },
    }
    if fold_segment_metrics:
        averaged["segment"] = {
            "accuracy": float(np.mean([item["accuracy"] for item in fold_segment_metrics])),
            "macro_f1": float(np.mean([item["macro_f1"] for item in fold_segment_metrics])),
            "per_class_recall": {
                name: float(np.mean([
                    item["per_class_recall"][name] for item in fold_segment_metrics
                ])) for name in CLASS_NAMES
            },
        }
    return _objective_score(averaged, objective_mode), averaged, X.shape, fold_segment_counts


def _flatten_history(iteration, params, score, metrics, feature_shape):
    row = {
        "iteration": int(iteration),
        "score": float(score),
        "cv_split_protocol": "recording_fixed_v2",
        **params,
        "class_weight": CLASS_WEIGHT,
        "feature_setting": FEATURE_SETTING,
        "window_overlap": WINDOW_OVERLAP,
        "cv_folds": CV_FOLDS,
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_recall": metrics["macro_recall"],
        "validation_macro_f1": metrics["macro_f1"],
        "feature_rows": int(feature_shape[0]),
        "feature_columns": int(feature_shape[1]),
    }
    for class_name in CLASS_NAMES:
        row[f"validation_recall_{class_name}"] = metrics["per_class_recall"][
            class_name
        ]
    if "segment" in metrics:
        row["validation_segment_accuracy"] = metrics["segment"]["accuracy"]
        row["validation_segment_macro_f1"] = metrics["segment"]["macro_f1"]
        for class_name in CLASS_NAMES:
            row[f"validation_segment_recall_{class_name}"] = (
                metrics["segment"]["per_class_recall"][class_name]
            )
    return row


def _majority_vote_by_segment(predictions, y, groups):
    segment_true = []
    segment_pred = []
    for segment_id in np.unique(groups):
        mask = groups == segment_id
        true_values = np.unique(y[mask])
        if len(true_values) != 1:
            raise RuntimeError("One segment contains more than one true class")
        counts = np.bincount(predictions[mask], minlength=len(CLASS_NAMES))
        segment_true.append(int(true_values[0]))
        segment_pred.append(int(np.argmax(counts)))
    return np.asarray(segment_true), np.asarray(segment_pred)


def _final_test_once(
    train_signals,
    train_labels,
    train_segment_ids,
    test_signals,
    test_labels,
    test_segment_ids,
    params,
):
    """Fit all training segments and access the held-out test set once."""
    X_train, y_train, train_groups = _build_feature_matrix(
        train_signals,
        train_labels,
        train_segment_ids,
        params,
    )
    X_test, y_test, test_groups = _build_feature_matrix(
        test_signals,
        test_labels,
        test_segment_ids,
        params,
    )
    if set(train_groups.tolist()).intersection(set(test_groups.tolist())):
        raise RuntimeError("Outer train/test segment leakage detected")

    model = _make_model(params)
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    window_metrics = _metrics(y_test, test_predictions)

    segment_true, segment_predictions = _majority_vote_by_segment(
        test_predictions,
        y_test,
        test_groups,
    )
    segment_metrics = _metrics(segment_true, segment_predictions)
    return window_metrics, segment_metrics, X_train.shape, X_test.shape


def _json_metrics(metrics):
    result = dict(metrics)
    result["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    return result


def _save_split_manifest(
    train_ids, test_ids, raw_labels, results_dir=RESULTS_DIR,
    random_state=RANDOM_STATE,
):
    manifest = {
        "random_state": random_state,
        "test_size": TEST_SIZE,
        "train_segment_ids": train_ids.tolist(),
        "test_segment_ids": test_ids.tolist(),
        "train_class_counts": {
            CLASS_NAMES[label]: int(np.sum(raw_labels[train_ids] == label))
            for label in CLASS_LABELS
        },
        "test_class_counts": {
            CLASS_NAMES[label]: int(np.sum(raw_labels[test_ids] == label))
            for label in CLASS_LABELS
        },
    }
    (results_dir / "grouped_split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _finalize_saved_history(
    first_n,
    train_signals,
    train_labels,
    train_segment_ids,
    test_signals,
    test_labels,
    test_segment_ids,
):
    """Select from the first N saved BO evaluations and test exactly once."""
    running_path = RESULTS_DIR / "grouped_bo_history_running.csv"
    history_df = pd.read_csv(running_path).head(int(first_n)).copy()
    if len(history_df) < int(first_n):
        raise ValueError(
            f"Requested {first_n} evaluations, but only {len(history_df)} are saved."
        )
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    best_params = {
        name: float(best_row[name])
        for name in ["f_low", "f_high", "window_size", "C", "gamma"]
    }
    window_metrics, segment_metrics, train_shape, test_shape = _final_test_once(
        train_signals,
        train_labels,
        train_segment_ids,
        test_signals,
        test_labels,
        test_segment_ids,
        best_params,
    )
    history_df.to_csv(RESULTS_DIR / "grouped_bo_history.csv", index=False)
    best_output = {
        "bo_evaluations_used": int(first_n),
        "best_iteration": int(best_row["iteration"]),
        "best_validation_score": float(best_row["score"]),
        "best_validation_accuracy": float(best_row["validation_accuracy"]),
        "best_validation_macro_f1": float(best_row["validation_macro_f1"]),
        "best_parameters": best_params,
        "class_weight": CLASS_WEIGHT,
        "feature_setting": FEATURE_SETTING,
        "feature_columns": 55,
        "window_overlap": WINDOW_OVERLAP,
        "train_feature_shape": list(train_shape),
        "test_feature_shape": list(test_shape),
        "window_level_test_metrics": _json_metrics(window_metrics),
        "segment_level_test_metrics": _json_metrics(segment_metrics),
        "test_evaluations": 1,
    }
    (RESULTS_DIR / "grouped_bo_final_result.json").write_text(
        json.dumps(best_output, indent=2),
        encoding="utf-8",
    )
    print(f"Finalized using only the first {first_n} BO evaluations.")
    print("Best iteration:", best_output["best_iteration"])
    print("Best parameters:", best_params)
    print("Window-level held-out test accuracy:", round(window_metrics["accuracy"], 4))
    print("Window-level held-out test macro F1:", round(window_metrics["macro_f1"], 4))
    print("Segment-level held-out test accuracy:", round(segment_metrics["accuracy"], 4))
    print("Segment-level held-out test macro F1:", round(segment_metrics["macro_f1"], 4))
    print("Held-out test evaluations: 1")
    return best_output


def run_grouped_bo(
    smoke_test=False, finalize_first=None, objective_mode="segment_accuracy",
    output_dir=None, random_state=RANDOM_STATE, feature_cache=None,
    parameter_prior="uniform",
):
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(f"objective_mode must be one of {OBJECTIVE_MODES}")
    if parameter_prior not in SUPPORTED_PRIORS:
        raise ValueError(f"parameter_prior must be one of {SUPPORTED_PRIORS}")
    if find_spec("pywt") is None:
        raise RuntimeError(
            "PyWavelets is required for the 55-feature nf_enhanced protocol. "
            "Install it in the active Python environment before running BO."
        )
    if feature_cache is None:
        feature_cache = {}
    if output_dir:
        results_dir = Path(output_dir)
    elif objective_mode == "segment_accuracy":
        results_dir = FIXED_FOLD_DIR / f"seed_{random_state}" / parameter_prior
    elif parameter_prior != "uniform":
        results_dir = (
            PROJECT_DIR / "results" / "06_grouped_prior_comparison"
            / f"seed_{random_state}" / objective_mode / parameter_prior
        )
    elif random_state == RANDOM_STATE:
        results_dir = (
            RESULTS_DIR if objective_mode == "nf_focused"
            else OBJECTIVE_COMPARISON_DIR / objective_mode
        )
    else:
        results_dir = (
            OBJECTIVE_COMPARISON_DIR / "repeated" / f"seed_{random_state}" / objective_mode
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_signals, raw_labels = load_bonn_dataset(
        DATASET_PATH,
        classification="five-class",
    )
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids = _split_original_segments(raw_labels, random_state)
    if set(train_ids.tolist()).intersection(set(test_ids.tolist())):
        raise RuntimeError("Outer segment split overlaps")
    _save_split_manifest(train_ids, test_ids, raw_labels, results_dir, random_state)

    train_signals, train_labels, train_segment_ids = _select_segments(
        raw_signals,
        raw_labels,
        train_ids,
    )
    test_signals, test_labels, test_segment_ids = _select_segments(
        raw_signals,
        raw_labels,
        test_ids,
    )
    fold_manifest = [
        {"fold": index, "train_segment_ids": train_fold.tolist(),
         "validation_segment_ids": validation_fold.tolist()}
        for index, (train_fold, validation_fold) in enumerate(
            _recording_folds(train_labels, train_segment_ids, random_state), start=1
        )
    ]
    (results_dir / "grouped_cv_folds.json").write_text(
        json.dumps(fold_manifest, indent=2), encoding="utf-8",
    )

    print("Grouped outer split created before windowing.")
    print(f"Training segments: {len(train_ids)}; held-out test segments: {len(test_ids)}")
    print("The BO objective receives training segments only.")

    if finalize_first is not None:
        if (objective_mode != "nf_focused" or results_dir != RESULTS_DIR
                or random_state != RANDOM_STATE or parameter_prior != "uniform"):
            raise ValueError("--finalize-first supports only the archived N/F-focused run")
        return _finalize_saved_history(
            finalize_first,
            train_signals,
            train_labels,
            train_segment_ids,
            test_signals,
            test_labels,
            test_segment_ids,
        )

    if smoke_test:
        params = {
            "f_low": 1.0,
            "f_high": 40.0,
            "window_size": 5.0,
            "C": 10.0,
            "gamma": 0.01,
        }
        score, metrics, feature_shape, fold_counts = _cross_validate_training_segments(
            train_signals,
            train_labels,
            train_segment_ids,
            params,
            objective_mode=objective_mode,
            random_state=random_state,
            feature_cache=feature_cache,
        )
        print("Smoke test passed without accessing held-out test features.")
        print("Feature shape:", feature_shape)
        print("Fold segment counts:", fold_counts)
        print("Grouped CV score:", round(score, 4))
        print("Grouped CV macro F1:", round(metrics["macro_f1"], 4))
        return None

    running_path = results_dir / "grouped_bo_history_running.csv"
    search_space = SEARCH_SPACE if parameter_prior == "uniform" else [
        Real(CDF_EPSILON, 1.0 - CDF_EPSILON, name=f"u_{name}")
        for name in PREPROCESSING_BOUNDS
    ] + SEARCH_SPACE[3:]
    history = []
    x0 = None
    y0 = None
    if running_path.exists():
        saved = pd.read_csv(running_path)
        if len(saved) > N_CALLS:
            raise ValueError(f"Running history exceeds {N_CALLS} evaluations: {running_path}")
        if len(saved):
            if parameter_prior != "uniform" and (
                "parameter_prior" not in saved
                or not saved["parameter_prior"].eq(parameter_prior).all()
            ):
                raise ValueError(f"Running history has a different prior: {running_path}")
            if parameter_prior == "uniform" and "parameter_prior" in saved and (
                not saved["parameter_prior"].eq(parameter_prior).all()
            ):
                raise ValueError(f"Running history has a different prior: {running_path}")
            if not saved["iteration"].tolist() == list(range(1, len(saved) + 1)):
                raise ValueError(f"Running history has invalid iteration numbers: {running_path}")
            if not saved["feature_setting"].eq(FEATURE_SETTING).all():
                raise ValueError(f"Running history has a different feature setting: {running_path}")
            if objective_mode == "segment_accuracy" and (
                "cv_split_protocol" not in saved
                or not saved["cv_split_protocol"].eq("recording_fixed_v2").all()
                or "validation_segment_accuracy" not in saved
                or not np.allclose(saved["score"], saved["validation_segment_accuracy"])
            ):
                raise ValueError(f"Running history uses an older CV protocol: {running_path}")
            history = saved.to_dict("records")
            x0 = [
                [
                    parameter_to_latent(row[name], parameter_prior, bounds)
                    for name, bounds in PREPROCESSING_BOUNDS.items()
                ] + [float(row["C"]), float(row["gamma"])]
                for _, row in saved.iterrows()
            ] if parameter_prior != "uniform" else saved[
                ["f_low", "f_high", "window_size", "C", "gamma"]
            ].values.tolist()
            y0 = (-saved["score"]).tolist()
            print(f"Resuming {len(history)}/{N_CALLS} saved evaluations.", flush=True)

    def objective(point):
        preprocessing = (
            dict(zip(PREPROCESSING_BOUNDS, point[:3]))
            if parameter_prior == "uniform"
            else transform_preprocessing_parameters(point[:3], parameter_prior)
        )
        params = {**preprocessing, "C": float(point[3]), "gamma": float(point[4])}
        score, metrics, feature_shape, _ = _cross_validate_training_segments(
            train_signals,
            train_labels,
            train_segment_ids,
            params,
            objective_mode=objective_mode,
            random_state=random_state,
            feature_cache=feature_cache,
        )
        row = _flatten_history(
            len(history) + 1,
            params,
            score,
            metrics,
            feature_shape,
        )
        if parameter_prior != "uniform":
            row["parameter_prior"] = parameter_prior
        history.append(row)
        pd.DataFrame(history).to_csv(running_path, index=False)
        print(
            f"Grouped BO {row['iteration']:02d}/{N_CALLS} | "
            f"score={score:.4f} | macro_f1={metrics['macro_f1']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"recall_N={metrics['per_class_recall']['N']:.4f} | "
            f"recall_F={metrics['per_class_recall']['F']:.4f}"
        )
        return -score

    if history and len(history) < N_INITIAL_POINTS:
        planned_initial = _initial_random_points(search_space, random_state)
        if not np.allclose(
            np.asarray(x0, dtype=float),
            np.asarray(planned_initial[:len(history)], dtype=float),
            rtol=0, atol=1e-12,
        ):
            raise ValueError("Saved initial BO candidates do not match the random seed")
        for point in planned_initial[len(history):]:
            objective(point)
        x0 = planned_initial
        y0 = [-row["score"] for row in history]

    remaining = N_CALLS - len(history)
    result = None
    if remaining:
        result = gp_minimize(
            objective,
            search_space,
            n_calls=remaining,
            n_initial_points=max(0, N_INITIAL_POINTS - len(history)),
            random_state=random_state,
            acq_func="EI",
            x0=x0,
            y0=y0,
        )

    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    best_params = {
        name: float(best_row[name])
        for name in ["f_low", "f_high", "window_size", "C", "gamma"]
    }

    # This is the first and only point where test segments are transformed or scored.
    window_metrics, segment_metrics, train_shape, test_shape = _final_test_once(
        train_signals,
        train_labels,
        train_segment_ids,
        test_signals,
        test_labels,
        test_segment_ids,
        best_params,
    )

    history_df.to_csv(results_dir / "grouped_bo_history.csv", index=False)
    best_output = {
        "objective_mode": objective_mode,
        "cv_split_protocol": "recording_fixed_v2",
        "parameter_prior": parameter_prior,
        "bo_evaluations_used": N_CALLS,
        "random_state": random_state,
        "cv_folds": CV_FOLDS,
        "n_initial_points": N_INITIAL_POINTS,
        "best_iteration": int(best_row["iteration"]),
        "best_validation_score": float(best_row["score"]),
        "best_validation_accuracy": float(best_row["validation_accuracy"]),
        "best_validation_macro_f1": float(best_row["validation_macro_f1"]),
        **({
            "best_validation_segment_accuracy": float(best_row["validation_segment_accuracy"]),
            "best_validation_segment_macro_f1": float(best_row["validation_segment_macro_f1"]),
        } if objective_mode == "segment_accuracy" else {}),
        "best_parameters": best_params,
        "class_weight": CLASS_WEIGHT,
        "feature_setting": FEATURE_SETTING,
        "feature_columns": 55,
        "window_overlap": WINDOW_OVERLAP,
        "train_feature_shape": list(train_shape),
        "test_feature_shape": list(test_shape),
        "window_level_test_metrics": _json_metrics(window_metrics),
        "segment_level_test_metrics": _json_metrics(segment_metrics),
        "test_evaluations": 1,
    }
    (results_dir / "grouped_bo_final_result.json").write_text(
        json.dumps(best_output, indent=2),
        encoding="utf-8",
    )

    print("\nGrouped BO completed.")
    print("Best iteration:", best_output["best_iteration"])
    print("Best parameters:", best_params)
    print("Window-level held-out test accuracy:", round(window_metrics["accuracy"], 4))
    print("Window-level held-out test macro F1:", round(window_metrics["macro_f1"], 4))
    print("Segment-level held-out test accuracy:", round(segment_metrics["accuracy"], 4))
    print("Segment-level held-out test macro F1:", round(segment_metrics["macro_f1"], 4))
    print("Held-out test evaluations: 1")
    return result, best_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate grouped CV using training segments only; never access test features.",
    )
    parser.add_argument(
        "--finalize-first",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N saved BO evaluations, then evaluate test once.",
    )
    parser.add_argument("--objective-mode", choices=OBJECTIVE_MODES, default="segment_accuracy")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--parameter-prior", choices=SUPPORTED_PRIORS, default="uniform")
    arguments = parser.parse_args()
    run_grouped_bo(
        smoke_test=arguments.smoke_test,
        finalize_first=arguments.finalize_first,
        objective_mode=arguments.objective_mode,
        output_dir=arguments.output_dir,
        random_state=arguments.random_state,
        parameter_prior=arguments.parameter_prior,
    )

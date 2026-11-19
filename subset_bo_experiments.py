"""
subset_bo_experiments.py

Full Bayesian Optimization for Bonn EEG subset classification tasks:
    1. Z vs S
    2. F vs S
    3. Z vs F vs S

For each subset task, BO searches:
    f_low, f_high, window_size, overlap, select_k, C, gamma
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Categorical, Real
from skopt.utils import use_named_args

from data_loader import load_bonn_dataset
from features import extract_features_by_setting
from preprocessing import bandpass_filter_signals, segment_windows


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "01_subset_tasks"
FS = 173.61
RANDOM_STATE = 42
N_CALLS = 20
N_INITIAL_POINTS = 16
VALIDATION_SIZE = 0.2
FIXED_FEATURE_SETTING = "entropy_hjorth"

LABEL_NAMES = {
    0: "Z",
    1: "O",
    2: "N",
    3: "F",
    4: "S",
}

TASKS = [
    {"name": "Z_vs_S", "labels": [0, 4]},
    {"name": "F_vs_S", "labels": [3, 4]},
    {"name": "Z_vs_F_vs_S", "labels": [0, 3, 4]},
]

SEARCH_SPACE = [
    Real(0.5, 5.0, name="f_low"),
    Real(30.0, 60.0, name="f_high"),
    Real(3.0, 5.0, name="window_size"),
    Categorical([0.0, 0.5], name="overlap"),
    Categorical([20, 35, 55], name="select_k"),
    Real(0.1, 100.0, prior="log-uniform", name="C"),
    Real(1e-4, 1.0, prior="log-uniform", name="gamma"),
]


def _subset_data(raw_signals, raw_labels, labels):
    """Return signals and labels for one subset task."""
    selected_labels = np.asarray(labels, dtype=int)
    keep_mask = np.isin(raw_labels, selected_labels)
    signals = [signal for signal, keep in zip(raw_signals, keep_mask) if keep]
    y = raw_labels[keep_mask]
    return signals, y, selected_labels.tolist()


def _make_pipeline(select_k, n_features, C, gamma):
    """Build a SelectKBest + RBF-SVM pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "select",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(int(select_k), int(n_features)),
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=float(C),
                    gamma=float(gamma),
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _metrics(y_true, y_pred, labels):
    """Return accuracy, recall dictionary, and confusion matrix."""
    class_names = [LABEL_NAMES[label] for label in labels]
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": {
            class_name: float(recall_value)
            for class_name, recall_value in zip(class_names, recalls)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def _score_candidate(signals, labels, class_labels, params):
    """Evaluate one BO candidate with CV plus held-out test metrics."""
    filtered = bandpass_filter_signals(
        signals,
        FS,
        params["f_low"],
        params["f_high"],
    )
    windows, y = segment_windows(
        filtered,
        labels,
        FS,
        params["window_size"],
        overlap=params["overlap"],
    )
    X = extract_features_by_setting(windows, FS, params["feature_setting"])

    train_val_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y[train_val_idx],
    )

    validation_model = _make_pipeline(
        params["select_k"],
        X.shape[1],
        params["C"],
        params["gamma"],
    )
    validation_model.fit(X[train_idx], y[train_idx])
    val_pred = validation_model.predict(X[val_idx])
    validation_score = accuracy_score(y[val_idx], val_pred)

    final_model = _make_pipeline(
        params["select_k"],
        X.shape[1],
        params["C"],
        params["gamma"],
    )
    final_model.fit(X[train_idx], y[train_idx])
    test_pred = final_model.predict(X[test_idx])
    test_metrics = _metrics(y[test_idx], test_pred, class_labels)

    return float(validation_score), test_metrics, X.shape, {
        int(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))
    }


def _history_row(task_name, iteration, score, params, test_metrics, feature_shape, class_distribution):
    """Flatten one BO candidate result for CSV."""
    row = {
        "task": task_name,
        "iteration": iteration,
        "score": score,
        **params,
        "feature_rows": int(feature_shape[0]),
        "feature_columns": int(feature_shape[1]),
        "class_distribution": json.dumps(class_distribution),
        "test_accuracy": test_metrics["accuracy"],
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"].tolist()),
    }
    for class_name, recall_value in test_metrics["per_class_recall"].items():
        row[f"test_recall_{class_name}"] = recall_value
    return row


def _run_one_task(task, raw_signals, raw_labels):
    """Run full BO for one subset task."""
    signals, labels, class_labels = _subset_data(raw_signals, raw_labels, task["labels"])
    class_names = [LABEL_NAMES[label] for label in class_labels]
    running_output = RESULTS_DIR / f"subset_bo_{task['name']}_running.csv"

    if running_output.exists():
        existing = pd.read_csv(running_output)
        if len(existing) >= N_CALLS:
            print(f"Skipping {task['name']}; existing BO history has {len(existing)} rows.")
            best_row = existing.loc[existing["score"].idxmax()].copy()
            return existing, best_row

    print("\n" + "=" * 72)
    print(f"Full BO task: {task['name']} ({' / '.join(class_names)})")
    print("=" * 72)

    history = []

    @use_named_args(SEARCH_SPACE)
    def objective(f_low, f_high, window_size, overlap, select_k, C, gamma):
        params = {
            "f_low": float(f_low),
            "f_high": float(f_high),
            "window_size": float(window_size),
            "overlap": float(overlap),
            "feature_setting": FIXED_FEATURE_SETTING,
            "select_k": int(select_k),
            "C": float(C),
            "gamma": float(gamma),
        }
        try:
            score, test_metrics, feature_shape, class_distribution = _score_candidate(
                signals,
                labels,
                class_labels,
                params,
            )
        except Exception as exc:
            print(f"{task['name']} candidate failed: {exc}")
            score = 0.0
            test_metrics = {
                "accuracy": 0.0,
                "per_class_recall": {name: 0.0 for name in class_names},
                "confusion_matrix": np.zeros((len(class_labels), len(class_labels)), dtype=int),
            }
            feature_shape = (0, 0)
            class_distribution = {}

        row = _history_row(
            task["name"],
            len(history) + 1,
            score,
            params,
            test_metrics,
            feature_shape,
            class_distribution,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(running_output, index=False)
        print(
            f"{task['name']} BO {row['iteration']:02d}/{N_CALLS} | "
            f"cv_accuracy={score:.4f} | test_accuracy={row['test_accuracy']:.4f} | "
            f"overlap={params['overlap']}"
        )
        return -score

    gp_minimize(
        objective,
        SEARCH_SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=RANDOM_STATE,
        acq_func="EI",
    )

    history_df = pd.DataFrame(history)
    best_row = history_df.loc[history_df["score"].idxmax()].copy()
    history_df.to_csv(running_output, index=False)
    return history_df, best_row


def run_subset_bo_experiments():
    """Run BO for all subset tasks and save detailed results."""
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading five-class Bonn EEG dataset...")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")

    all_history = []
    best_rows = []
    for task in TASKS:
        history_df, best_row = _run_one_task(task, raw_signals, raw_labels)
        all_history.append(history_df)
        best_rows.append(best_row)

    history_output = RESULTS_DIR / "subset_bo_history.csv"
    best_output = RESULTS_DIR / "subset_bo_results.csv"
    pd.concat(all_history, ignore_index=True).to_csv(history_output, index=False)
    pd.DataFrame(best_rows).to_csv(best_output, index=False)

    print("\n" + "=" * 72)
    print("Subset BO experiments completed.")
    print(f"Saved history to {history_output}")
    print(f"Saved best results to {best_output}")
    print(pd.DataFrame(best_rows)[["task", "score", "test_accuracy", "feature_setting", "overlap"]].to_string(index=False))

    return pd.DataFrame(best_rows)


if __name__ == "__main__":
    run_subset_bo_experiments()

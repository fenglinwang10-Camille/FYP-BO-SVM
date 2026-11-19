"""
random_search.py

Random Search baseline for the current project stage.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from data_loader import load_bonn_dataset
from objective import CLASS_NAMES, evaluate_eeg_svm_params


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"
FS = 173.61

N_CALLS = 80
RANDOM_STATE = 42
SCORE_MODE = "nf_focused"
USE_CV = True
CV_FOLDS = 5
FIXED_FEATURE_SETTING = "entropy_hjorth"


def _latest_incomplete_running_history():
    """
    Find the newest unfinished Random Search running-history file.

    This lets a long Random Search run continue after a Jupyter interruption.
    """
    running_files = sorted(RESULTS_DIR.glob("random_search_history_running_*.csv"))
    for path in reversed(running_files):
        try:
            dataframe = pd.read_csv(path)
        except Exception:
            continue
        if 0 < len(dataframe) < N_CALLS and "overlap" in dataframe.columns:
            return path, dataframe
    return None, None


def sample_log_uniform(rng, low, high):
    """Sample a positive value evenly on a logarithmic scale."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _metrics_columns(prefix, metrics):
    """Flatten metrics for CSV output."""
    row = {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_macro_recall": metrics["macro_recall"],
    }
    for class_name in CLASS_NAMES:
        row[f"{prefix}_recall_{class_name}"] = metrics["per_class_recall"][class_name]
    row[f"{prefix}_seizure_sensitivity"] = metrics["seizure_sensitivity"]
    row[f"{prefix}_confusion_matrix"] = json.dumps(
        metrics["confusion_matrix"].tolist()
    )
    return row


def _history_row(iteration, details):
    """Build one Random Search history row."""
    return {
        "iteration": iteration,
        "score": details["score"],
        **details["parameters"],
        **_metrics_columns("validation", details["validation_metrics"]),
        **_metrics_columns("test", details["test_metrics"]),
    }


def _format_recall_values(row, prefix):
    """Format per-class recall values for terminal progress output."""
    values = [
        f"{class_name}={row[f'{prefix}_recall_{class_name}']:.4f}"
        for class_name in CLASS_NAMES
    ]
    return ", ".join(values)


def run_random_search():
    """
    Run Random Search using the same search space and number of calls as BO.

    Random Search is added as a baseline optimization method. It samples
    parameter combinations independently and selects the best one using
    validation accuracy, then reports test performance for that selected set.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    running_path, resume_df = _latest_incomplete_running_history()
    if running_path is None:
        running_path = RESULTS_DIR / "random_search_history_running_current.csv"
        history = []
        completed = 0
        print("Starting a new Random Search run.")
    else:
        history = resume_df.to_dict("records")
        completed = len(history)
        print(f"Resuming Random Search from {running_path}")
        print(f"Already completed {completed}/{N_CALLS} evaluations.")

    print("Loading five-class Bonn EEG dataset...")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")

    rng = np.random.default_rng(RANDOM_STATE)

    # Advance the RNG so resumed runs use the same parameter sequence as one
    # uninterrupted run, without repeating completed evaluations.
    for _ in range(completed):
        rng.uniform(0.5, 5.0)
        rng.uniform(30.0, 60.0)
        rng.uniform(1.0, 5.0)
        rng.choice([0.0, 0.5])
        sample_log_uniform(rng, 0.1, 100.0)
        sample_log_uniform(rng, 1e-4, 1.0)
        rng.choice(["none", "balanced"])

    for iteration in range(completed + 1, N_CALLS + 1):
        params = {
            "f_low": float(rng.uniform(0.5, 5.0)),
            "f_high": float(rng.uniform(30.0, 60.0)),
            "window_size": float(rng.uniform(1.0, 5.0)),
            "overlap": float(rng.choice([0.0, 0.5])),
            "C": sample_log_uniform(rng, 0.1, 100.0),
            "gamma": sample_log_uniform(rng, 1e-4, 1.0),
            "class_weight": str(rng.choice(["none", "balanced"])),
            "feature_setting": FIXED_FEATURE_SETTING,
        }

        details = evaluate_eeg_svm_params(
            raw_signals,
            raw_labels,
            FS,
            score_mode=SCORE_MODE,
            use_cv=USE_CV,
            cv_folds=CV_FOLDS,
            return_details=True,
            **params,
        )

        row = _history_row(iteration, details)
        history.append(row)
        pd.DataFrame(history).to_csv(running_path, index=False)

        print(
            f"RS {iteration:02d}/{N_CALLS} | "
            f"validation_accuracy={row['validation_accuracy']:.4f} | "
            f"validation_recall_rate=[{_format_recall_values(row, 'validation')}] | "
            f"validation_seizure_sensitivity={row['validation_seizure_sensitivity']:.4f} | "
            f"overlap={row['overlap']}"
        )

    history_df = pd.DataFrame(history)
    best_index = history_df["score"].idxmax()
    best_row = history_df.loc[best_index].copy()
    history_df.to_csv(RESULTS_DIR / "random_search_history.csv", index=False)
    pd.DataFrame([best_row]).to_csv(
        RESULTS_DIR / "random_search_best_params.csv",
        index=False,
    )

    print("\nRandom Search completed.")
    print("Best validation accuracy:", round(float(best_row["validation_accuracy"]), 4))
    print("Best validation macro recall:", round(float(best_row["validation_macro_recall"]), 4))
    print("Best test accuracy:", round(float(best_row["test_accuracy"]), 4))
    print("Best test macro recall:", round(float(best_row["test_macro_recall"]), 4))
    print("Best test seizure sensitivity:", round(float(best_row["test_seizure_sensitivity"]), 4))
    print("Best parameters:")
    for column in [
        "f_low",
        "f_high",
        "window_size",
        "overlap",
        "C",
        "gamma",
        "class_weight",
        "feature_setting",
    ]:
        print(f"  {column}: {best_row[column]}")

    return history_df


if __name__ == "__main__":
    run_random_search()

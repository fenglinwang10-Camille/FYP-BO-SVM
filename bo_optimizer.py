"""
bo_optimizer.py

Bayesian Optimization for Bonn EEG RBF-SVM parameter tuning.
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import shutil

import pandas as pd
from skopt import gp_minimize
from skopt.space import Categorical, Real

from data_loader import load_bonn_dataset
from objective import CLASS_NAMES, evaluate_eeg_svm_params
from parameter_priors import (
    CDF_EPSILON,
    PREPROCESSING_BOUNDS,
    SUPPORTED_PRIORS,
    parameter_to_latent,
    transform_preprocessing_parameters,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"
FS = 173.61

N_CALLS = 80
N_INITIAL_POINTS = 16
RANDOM_STATE = 42
FIXED_CLASS_WEIGHT = "balanced"
FIXED_FEATURE_SETTING = "entropy_hjorth"
SCORE_MODE = "nf_focused"
USE_CV = True
CV_FOLDS = 5
SEEDED_INITIAL_POINT = [
    4.296087282189888,
    57.07959431782851,
    4.918282722346371,
    0.0,
    25.472856018371225,
    0.1311928489172758,
]

PARAMETER_COLUMNS = [
    "f_low",
    "f_high",
    "window_size",
    "overlap",
    "C",
    "gamma",
]


def _results_dir(parameter_prior):
    if parameter_prior == "uniform":
        return RESULTS_DIR
    return PROJECT_DIR / "results" / "02_prior_experiments" / parameter_prior


def _search_space(parameter_prior):
    if parameter_prior == "uniform":
        preprocessing_space = [
            Real(bounds.low, bounds.high, name=name)
            for name, bounds in PREPROCESSING_BOUNDS.items()
        ]
    else:
        preprocessing_space = [
            Real(CDF_EPSILON, 1.0 - CDF_EPSILON, name=f"u_{name}")
            for name in PREPROCESSING_BOUNDS
        ]
    return preprocessing_space + [
        Categorical([0.0, 0.5], name="overlap"),
        Real(0.1, 100.0, prior="log-uniform", name="C"),
        Real(1e-4, 1.0, prior="log-uniform", name="gamma"),
    ]


def _optimizer_point_to_parameters(point, parameter_prior):
    if parameter_prior == "uniform":
        preprocessing = dict(zip(PREPROCESSING_BOUNDS, point[:3]))
    else:
        preprocessing = transform_preprocessing_parameters(point[:3], parameter_prior)
    return {
        **preprocessing,
        "overlap": point[3],
        "C": point[4],
        "gamma": point[5],
    }


def _parameters_to_optimizer_point(row, parameter_prior):
    if parameter_prior == "uniform":
        preprocessing = [float(row[name]) for name in PREPROCESSING_BOUNDS]
    else:
        preprocessing = [
            parameter_to_latent(row[name], parameter_prior, bounds)
            for name, bounds in PREPROCESSING_BOUNDS.items()
        ]
    return preprocessing + [float(row["overlap"]), float(row["C"]), float(row["gamma"])]


def _latest_incomplete_running_history(results_dir, parameter_prior):
    """
    Find the newest unfinished focused BO running-history file.

    This lets a long BO run continue after a Jupyter interruption instead of
    wasting all previous evaluations.
    """
    running_files = sorted(results_dir.glob("bo_cv5_nf_focused_history_running_*.csv"))
    for path in reversed(running_files):
        try:
            dataframe = pd.read_csv(path)
        except Exception:
            continue
        has_current_columns = all(column in dataframe.columns for column in PARAMETER_COLUMNS)
        prior_matches = (
            "parameter_prior" not in dataframe.columns
            and parameter_prior == "uniform"
        ) or (
            "parameter_prior" in dataframe.columns
            and dataframe["parameter_prior"].eq(parameter_prior).all()
        )
        if 0 < len(dataframe) < N_CALLS and has_current_columns and prior_matches:
            return path, dataframe
    return None, None


def _metrics_columns(prefix, metrics):
    """Flatten accuracy, per-class recall, and seizure sensitivity for CSV."""
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


def _history_row(iteration, details, parameter_prior):
    """Build one BO history row."""
    return {
        "iteration": iteration,
        "parameter_prior": parameter_prior,
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


def run_bayesian_optimization(
    parameter_prior="uniform", output_dir=None, feature_setting=FIXED_FEATURE_SETTING
):
    """
    Run focused Bayesian Optimization with a Gaussian Process surrogate.

    This experiment fixes class_weight and feature_setting, then lets BO search
    signal processing and SVM parameters. The first point is seeded from the
    best Random Search configuration.
    """
    if parameter_prior not in SUPPORTED_PRIORS:
        raise ValueError(f"parameter_prior must be one of {SUPPORTED_PRIORS}")

    results_dir = Path(output_dir) if output_dir is not None else _results_dir(parameter_prior)
    results_dir.mkdir(parents=True, exist_ok=True)
    search_space = _search_space(parameter_prior)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    official_history_path = results_dir / "bo_history.csv"
    official_best_path = results_dir / "bo_best_params.csv"
    resume_path, resume_df = _latest_incomplete_running_history(
        results_dir, parameter_prior
    )
    running_history_path = (
        resume_path or results_dir / f"bo_cv5_nf_focused_history_running_{run_id}.csv"
    )

    if resume_df is None and official_history_path.exists():
        backup_path = results_dir / f"bo_history_backup_{run_id}.csv"
        shutil.copy2(official_history_path, backup_path)
        print(f"Backed up previous BO history to {backup_path}")

    print("Loading five-class Bonn EEG dataset...")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")

    history = []
    x0 = None
    y0 = None

    if resume_df is not None:
        history = resume_df.to_dict("records")
        x0 = [
            _parameters_to_optimizer_point(row, parameter_prior)
            for _, row in resume_df.iterrows()
        ]
        y0 = (-resume_df["score"]).tolist()
        print(f"Resuming BO from {running_history_path}")
        print(f"Already completed {len(history)}/{N_CALLS} evaluations.")
    else:
        seed_row = dict(zip(PARAMETER_COLUMNS, SEEDED_INITIAL_POINT))
        x0 = [_parameters_to_optimizer_point(seed_row, parameter_prior)]
        print("Starting a new N/F-focused seeded BO run.")
        print(f"Preprocessing parameter prior={parameter_prior}")
        print(f"Fixed class_weight={FIXED_CLASS_WEIGHT}")
        print(f"Fixed feature_setting={feature_setting}")
        print(f"Objective score = {CV_FOLDS}-fold CV nf_focused validation score")
        print(f"Initial point from Random Search best: {SEEDED_INITIAL_POINT}")

    def objective_for_bo(point):
        parameters = _optimizer_point_to_parameters(point, parameter_prior)
        details = evaluate_eeg_svm_params(
            raw_signals,
            raw_labels,
            FS,
            **parameters,
            class_weight=FIXED_CLASS_WEIGHT,
            feature_setting=feature_setting,
            score_mode=SCORE_MODE,
            use_cv=USE_CV,
            cv_folds=CV_FOLDS,
            return_details=True,
        )

        row = _history_row(len(history) + 1, details, parameter_prior)
        history.append(row)
        pd.DataFrame(history).to_csv(running_history_path, index=False)

        print(
            f"BO {row['iteration']:02d}/{N_CALLS} | "
            f"score={row['score']:.4f} | "
            f"validation_accuracy={row['validation_accuracy']:.4f} | "
            f"validation_macro_recall={row['validation_macro_recall']:.4f} | "
            f"validation_recall_rate=[{_format_recall_values(row, 'validation')}] | "
            f"validation_seizure_sensitivity={row['validation_seizure_sensitivity']:.4f} | "
            f"overlap={row['overlap']}"
        )

        # gp_minimize minimizes, so negative score maximizes it.
        return -row["score"]

    remaining_calls = N_CALLS - len(history)
    if remaining_calls <= 0:
        print(f"BO already has {N_CALLS} evaluations. Using existing running history.")
    else:
        gp_minimize(
            objective_for_bo,
            search_space,
            n_calls=remaining_calls,
            n_initial_points=max(0, N_INITIAL_POINTS - len(history)),
            random_state=RANDOM_STATE,
            acq_func="EI",
            x0=x0,
            y0=y0,
        )

    history_df = pd.DataFrame(history)
    best_index = history_df["score"].idxmax()
    best_row = history_df.loc[best_index].copy()

    history_df.to_csv(official_history_path, index=False)
    pd.DataFrame([best_row]).to_csv(official_best_path, index=False)

    print("\nBayesian Optimization completed.")
    print(f"Saved BO history to {official_history_path}")
    print(f"Saved best BO parameters to {official_best_path}")
    print("Best validation accuracy:", round(float(best_row["validation_accuracy"]), 4))
    print("Best validation macro recall:", round(float(best_row["validation_macro_recall"]), 4))
    print("Best objective score:", round(float(best_row["score"]), 4))
    print("Best test accuracy:", round(float(best_row["test_accuracy"]), 4))
    print("Best test macro recall:", round(float(best_row["test_macro_recall"]), 4))
    print("Best test seizure sensitivity:", round(float(best_row["test_seizure_sensitivity"]), 4))
    print("Best parameters:")
    for column in PARAMETER_COLUMNS:
        print(f"  {column}: {best_row[column]}")

    return history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run direct BO-SVM with a selectable preprocessing prior."
    )
    parser.add_argument(
        "--parameter-prior",
        choices=SUPPORTED_PRIORS,
        default="uniform",
        help="Inverse-CDF prior for f_low, f_high, and window_size.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional result directory. Useful for fair comparison runs.",
    )
    parser.add_argument(
        "--feature-setting",
        choices=("original", "entropy_hjorth", "expanded", "nf_enhanced"),
        default=FIXED_FEATURE_SETTING,
        help="Feature extraction setting held fixed for this BO run.",
    )
    args = parser.parse_args()
    run_bayesian_optimization(
        args.parameter_prior,
        output_dir=args.output_dir,
        feature_setting=args.feature_setting,
    )

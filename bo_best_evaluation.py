"""
bo_best_evaluation.py

Re-run EEG-SVM with the best Bayesian Optimization parameters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_bonn_dataset
from display_utils import display_image
from objective import CLASS_NAMES, evaluate_eeg_svm_params


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"
FS = 173.61


def save_confusion_matrix_plot(matrix, output_path):
    """Save a simple matplotlib confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("BO-SVM Best Test Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _print_metrics(title, metrics):
    """Print accuracy, per-class recall, and seizure sensitivity."""
    print(f"\n{title}")
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro recall:", round(metrics["macro_recall"], 4))
    print("Per-class recall rate:")
    for class_name in CLASS_NAMES:
        print(f"  Recall_{class_name}: {metrics['per_class_recall'][class_name]:.4f}")
    print("Seizure sensitivity (Recall_S):", round(metrics["seizure_sensitivity"], 4))


def run_best_evaluation():
    """Evaluate and report the best BO parameter set."""
    RESULTS_DIR.mkdir(exist_ok=True)

    best_path = RESULTS_DIR / "bo_best_params.csv"
    if not best_path.exists():
        raise FileNotFoundError(
            "Missing results/00_main_bo_svm/bo_best_params.csv. Run python bo_optimizer.py first."
        )

    best_row = pd.read_csv(best_path).iloc[0]
    params = {
        "f_low": float(best_row["f_low"]),
        "f_high": float(best_row["f_high"]),
        "window_size": float(best_row["window_size"]),
        "overlap": float(best_row.get("overlap", 0.0)),
        "C": float(best_row["C"]),
        "gamma": float(best_row["gamma"]),
        "class_weight": best_row.get("class_weight", "none"),
        "feature_setting": best_row.get("feature_setting", "entropy_hjorth"),
    }

    print("Loading five-class Bonn EEG dataset...")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")

    details = evaluate_eeg_svm_params(
        raw_signals,
        raw_labels,
        FS,
        return_details=True,
        **params,
    )

    print("\nBest BO parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    _print_metrics("Validation metrics", details["validation_metrics"])
    _print_metrics("Test metrics", details["test_metrics"])
    print("\nTest confusion matrix:")
    print(details["test_metrics"]["confusion_matrix"])

    output_path = RESULTS_DIR / "bo_best_confusion_matrix.png"
    save_confusion_matrix_plot(details["test_metrics"]["confusion_matrix"], output_path)
    print(f"\nSaved confusion matrix plot to {output_path}")

    display_image(output_path)

    return details


if __name__ == "__main__":
    run_best_evaluation()

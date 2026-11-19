"""
baseline_manual.py

Fixed-parameter five-class Bonn EEG RBF-SVM baseline.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_bonn_dataset
from display_utils import display_dataframe, display_image
from objective import CLASS_NAMES, evaluate_eeg_svm_params


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset"
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"
FS = 173.61

BASELINE_PARAMS = {
    "f_low": 0.5,
    "f_high": 40.0,
    "window_size": 2.0,
    "C": 10.0,
    "gamma": 0.1,
}


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


def _print_metrics(title, metrics):
    """Print accuracy, class recall rates, and seizure sensitivity."""
    print(f"\n{title}")
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro recall:", round(metrics["macro_recall"], 4))
    print("Per-class recall rate:")
    for class_name in CLASS_NAMES:
        value = metrics["per_class_recall"][class_name]
        print(f"  Recall_{class_name}: {value:.4f}")
    print("Seizure sensitivity (Recall_S):", round(metrics["seizure_sensitivity"], 4))


def save_confusion_matrix_plot(matrix, output_path, title):
    """Save a simple confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_baseline():
    """Run the manual SVM baseline and save validation/test metrics."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading five-class Bonn EEG dataset...")
    raw_signals, raw_labels = load_bonn_dataset(DATASET_PATH, classification="five-class")

    details = evaluate_eeg_svm_params(
        raw_signals,
        raw_labels,
        FS,
        return_details=True,
        **BASELINE_PARAMS,
    )

    print("\nFeature matrix shape:", details.get("feature_matrix_shape", "not available"))
    print("Class distribution:", details.get("class_distribution", "not available"))

    _print_metrics("Validation metrics", details["validation_metrics"])
    _print_metrics("Test metrics", details["test_metrics"])
    print("\nTest confusion matrix:")
    print(details["test_metrics"]["confusion_matrix"])

    row = {
        "method": "SVM without BO",
        **BASELINE_PARAMS,
        **_metrics_columns("validation", details["validation_metrics"]),
        **_metrics_columns("test", details["test_metrics"]),
    }
    metrics_df = pd.DataFrame([row])
    metrics_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

    confusion_path = RESULTS_DIR / "baseline_confusion_matrix.png"
    save_confusion_matrix_plot(
        details["test_metrics"]["confusion_matrix"],
        confusion_path,
        "Baseline SVM Test Confusion Matrix",
    )

    print(f"\nSaved baseline metrics to {RESULTS_DIR / 'baseline_metrics.csv'}")
    print(f"Saved baseline confusion matrix to {confusion_path}")

    display_dataframe(metrics_df)
    display_image(confusion_path)

    return details


if __name__ == "__main__":
    run_baseline()

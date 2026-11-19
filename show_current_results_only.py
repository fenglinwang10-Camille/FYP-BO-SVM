"""
show_current_results_only.py

Display saved current-stage results in Jupyter without training models.
"""

from pathlib import Path

import pandas as pd

from display_utils import display_dataframe, display_image


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"


def show_current_results_only():
    """Show saved baseline, BO, Random Search, tables, and plots."""
    print("Current stage: SVM without BO vs BO-SVM vs Random Search SVM.")
    print("This script only displays saved results; it does not train models.\n")

    files = {
        "Baseline metrics": RESULTS_DIR / "baseline_metrics.csv",
        "BO best parameters": RESULTS_DIR / "bo_best_params.csv",
        "Random Search best parameters": RESULTS_DIR / "random_search_best_params.csv",
        "Comparison summary": RESULTS_DIR / "comparison_summary.csv",
    }

    for label, path in files.items():
        if path.exists():
            print(label)
            display_dataframe(pd.read_csv(path))
        else:
            print(f"Missing {path}")

    image_paths = [
        RESULTS_DIR / "baseline_confusion_matrix.png",
        RESULTS_DIR / "bo_best_confusion_matrix.png",
        RESULTS_DIR / "convergence_bo_vs_random.png",
        RESULTS_DIR / "test_accuracy_comparison.png",
        RESULTS_DIR / "class_wise_recall_comparison.png",
    ]

    for image_path in image_paths:
        if image_path.exists():
            print(image_path.name)
            display_image(image_path)


if __name__ == "__main__":
    show_current_results_only()

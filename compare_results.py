"""
compare_results.py

Compare SVM without BO, BO-SVM, and Random Search SVM using final test metrics.
"""

import json
from pathlib import Path

import pandas as pd

from display_utils import display_dataframe
from objective import CLASS_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"


def _params_to_json(row):
    """Format parameters as one compact JSON string."""
    params = {
        "f_low": float(row["f_low"]),
        "f_high": float(row["f_high"]),
        "window_size": float(row["window_size"]),
        "C": float(row["C"]),
        "gamma": float(row["gamma"]),
    }
    if "overlap" in row and pd.notna(row["overlap"]):
        params["overlap"] = float(row["overlap"])
    if "class_weight" in row and pd.notna(row["class_weight"]):
        params["class_weight"] = row["class_weight"]
    if "feature_setting" in row and pd.notna(row["feature_setting"]):
        params["feature_setting"] = row["feature_setting"]
    return json.dumps(params)


def _macro_recall_from_row(row):
    """Read macro recall or derive it from saved per-class recall columns."""
    if "test_macro_recall" in row:
        return float(row["test_macro_recall"])
    return float(
        sum(float(row[f"test_recall_{class_name}"]) for class_name in CLASS_NAMES)
        / len(CLASS_NAMES)
    )


def _comparison_row(method, row):
    """Create one row for the comparison summary."""
    output = {
        "method": method,
        "test_accuracy": float(row["test_accuracy"]),
        "test_macro_recall": _macro_recall_from_row(row),
    }
    for class_name in CLASS_NAMES:
        output[f"test_recall_{class_name}"] = float(row[f"test_recall_{class_name}"])
    output["test_seizure_sensitivity"] = float(row["test_seizure_sensitivity"])
    output["test_confusion_matrix"] = row.get("test_confusion_matrix", "")
    output["best_parameters"] = _params_to_json(row)
    return output


def compare_results():
    """Read saved results and compare all three methods."""
    baseline_path = RESULTS_DIR / "baseline_metrics.csv"
    bo_best_path = RESULTS_DIR / "bo_best_params.csv"
    random_best_path = RESULTS_DIR / "random_search_best_params.csv"

    for path in [baseline_path, bo_best_path, random_best_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Follow README_RUN_STEPS.md first.")

    baseline_row = pd.read_csv(baseline_path).iloc[0]
    bo_best_row = pd.read_csv(bo_best_path).iloc[0]
    random_best_row = pd.read_csv(random_best_path).iloc[0]

    comparison = pd.DataFrame(
        [
            _comparison_row("SVM without BO", baseline_row),
            _comparison_row("SVM with BO", bo_best_row),
            _comparison_row("Random Search SVM", random_best_row),
        ]
    )

    output_path = RESULTS_DIR / "comparison_summary.csv"
    comparison.to_csv(output_path, index=False)

    print("\nFinal Test Set Comparison")
    display_columns = [
        "method",
        "test_accuracy",
        "test_macro_recall",
        "test_recall_Z",
        "test_recall_O",
        "test_recall_N",
        "test_recall_F",
        "test_recall_S",
        "test_seizure_sensitivity",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved comparison summary to {output_path}")

    display_dataframe(comparison)

    return comparison


if __name__ == "__main__":
    compare_results()

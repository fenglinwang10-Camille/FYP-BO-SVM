"""
make_subset_table3_metrics.py

Create clearly defined metrics for selected Bonn subset BO results:
    A/E   = Z/S
    D/E   = F/S
    A/D/E = Z/F/S

All tasks report macro recall. Seizure recall and seizure specificity treat
E/S as positive and every non-S class as negative.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SUBSET_DIR = PROJECT_DIR / "results" / "01_subset_tasks"
INPUT_PATH = SUBSET_DIR / "subset_bo_results.csv"
OUTPUT_CSV = SUBSET_DIR / "subset_bo_table3_metrics.csv"
OUTPUT_MD = SUBSET_DIR / "subset_bo_table3_metrics.md"


CASE_MAP = {
    "Z_vs_S": ("A-E", ["A", "E"]),
    "F_vs_S": ("D-E", ["D", "E"]),
    "Z_vs_F_vs_S": ("A-D-E", ["A", "D", "E"]),
}


def _binary_metrics(matrix):
    """Return accuracy, macro recall, seizure recall, and seizure specificity."""
    tn, fp = matrix[0, 0], matrix[0, 1]
    fn, tp = matrix[1, 0], matrix[1, 1]
    accuracy = (tp + tn) / matrix.sum()
    seizure_recall = tp / (tp + fn) if tp + fn else 0.0
    seizure_specificity = tn / (tn + fp) if tn + fp else 0.0
    nonseizure_recall = seizure_specificity
    macro_recall = (seizure_recall + nonseizure_recall) / 2.0
    return accuracy, macro_recall, seizure_recall, seizure_specificity


def _multiclass_metrics(matrix):
    """Return accuracy, macro recall, and S-vs-non-S seizure metrics."""
    total = matrix.sum()
    accuracy = np.trace(matrix) / total
    recalls = []

    for index in range(matrix.shape[0]):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)

    seizure_index = matrix.shape[0] - 1
    seizure_tp = matrix[seizure_index, seizure_index]
    seizure_fn = matrix[seizure_index, :].sum() - seizure_tp
    seizure_fp = matrix[:, seizure_index].sum() - seizure_tp
    seizure_tn = total - seizure_tp - seizure_fn - seizure_fp
    seizure_recall = seizure_tp / (seizure_tp + seizure_fn)
    seizure_specificity = seizure_tn / (seizure_tn + seizure_fp)
    return accuracy, float(np.mean(recalls)), seizure_recall, seizure_specificity


def _percent(value):
    """Convert a decimal metric to percent."""
    return float(value) * 100.0


def _markdown_table(dataframe):
    columns = list(dataframe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in dataframe.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def build_subset_table3_metrics():
    """Build CSV and Markdown table outputs."""
    source = pd.read_csv(INPUT_PATH)
    rows = []
    for task, (case_name, _) in CASE_MAP.items():
        match = source[source["task"] == task]
        if match.empty:
            continue
        row = match.iloc[0]
        matrix = np.asarray(json.loads(row["test_confusion_matrix"]), dtype=float)
        if matrix.shape == (2, 2):
            metrics = _binary_metrics(matrix)
        else:
            metrics = _multiclass_metrics(matrix)

        rows.append(
            {
                "Case": case_name,
                "Accuracy": f"{_percent(metrics[0]):.2f}",
                "Macro Recall": f"{_percent(metrics[1]):.2f}",
                "Seizure Recall": f"{_percent(metrics[2]):.2f}",
                "Seizure Specificity": f"{_percent(metrics[3]):.2f}",
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_MD.open("w", encoding="utf-8") as handle:
        handle.write("# Subset BO Table-3-Style Metrics\n\n")
        handle.write("Seizure recall and specificity treat E/S as positive. ")
        handle.write("Macro recall is the unweighted mean of class recalls.\n\n")
        handle.write(_markdown_table(output))
        handle.write("\n")

    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_MD}")
    print(output.to_string(index=False))
    return output


if __name__ == "__main__":
    build_subset_table3_metrics()

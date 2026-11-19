"""
meeting_results_summary.py

Create a readable meeting summary from saved comparison results.
"""

from pathlib import Path
import json

import pandas as pd

from objective import CLASS_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def _find_comparison_summary():
    """Find the current comparison_summary.csv in common result locations."""
    candidates = [
        RESULTS_DIR / "00_main_bo_svm" / "comparison_summary.csv",
        RESULTS_DIR / "comparison_summary.csv",
        RESULTS_DIR / "bo_random_search" / "comparison_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find comparison_summary.csv. Run compare_results.py first."
    )


def _percent(value):
    """Format a decimal metric as a presentation-friendly percentage."""
    return f"{float(value) * 100:.2f}%"


def _macro_recall(row):
    """Read macro recall or calculate it from per-class recall columns."""
    if "test_macro_recall" in row and pd.notna(row["test_macro_recall"]):
        return float(row["test_macro_recall"])
    return sum(float(row[f"test_recall_{name}"]) for name in CLASS_NAMES) / len(
        CLASS_NAMES
    )


def _short_params(row):
    """Keep parameter output short enough for a meeting table."""
    if "best_parameters" not in row or pd.isna(row["best_parameters"]):
        return ""
    try:
        params = json.loads(row["best_parameters"])
        return ", ".join(f"{key}: {value}" for key, value in params.items())
    except Exception:
        return str(row["best_parameters"]).replace('"', "")


def _markdown_table(dataframe):
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    columns = list(dataframe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_meeting_summary():
    """Build CSV and Markdown files that are easier to present than raw CSV."""
    source_path = _find_comparison_summary()
    df = pd.read_csv(source_path)
    best_method = df.loc[df["test_accuracy"].idxmax(), "method"]

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Method": row["method"],
                "Test Accuracy": _percent(row["test_accuracy"]),
                "Recall Z": _percent(row["test_recall_Z"]),
                "Recall O": _percent(row["test_recall_O"]),
                "Recall N": _percent(row["test_recall_N"]),
                "Recall F": _percent(row["test_recall_F"]),
                "Recall S": _percent(row["test_recall_S"]),
                "Seizure Sensitivity": _percent(row["test_seizure_sensitivity"]),
                "Best": "YES" if row["method"] == best_method else "",
                "Key Parameters": _short_params(row),
            }
        )

    readable = pd.DataFrame(rows)
    output_dir = source_path.parent
    csv_path = output_dir / "meeting_comparison_summary.csv"
    md_path = output_dir / "meeting_comparison_summary.md"

    readable.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Meeting Comparison Summary\n\n")
        handle.write(f"Source: `{source_path.relative_to(PROJECT_DIR)}`\n\n")
        handle.write(f"Best method by test accuracy: **{best_method}**\n\n")
        handle.write(_markdown_table(readable.drop(columns=["Key Parameters"])))
        handle.write("\n\n## Best Parameters\n\n")
        handle.write(
            _markdown_table(readable[["Method", "Key Parameters"]])
        )
        handle.write("\n")

    print(f"Saved readable CSV to {csv_path}")
    print(f"Saved Markdown summary to {md_path}")
    print(readable.drop(columns=["Key Parameters"]).to_string(index=False))
    return readable


if __name__ == "__main__":
    build_meeting_summary()

"""Compare grouped BO objective functions on the same held-out recordings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_bo_optimizer import N_CALLS, N_INITIAL_POINTS, OBJECTIVE_COMPARISON_DIR, RESULTS_DIR


MACRO_DIR = OBJECTIVE_COMPARISON_DIR / "macro_f1"
PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_comparable(weighted, macro):
    weighted_split = _read_json(RESULTS_DIR / "grouped_split_manifest.json")
    macro_split = _read_json(MACRO_DIR / "grouped_split_manifest.json")
    if weighted_split != macro_split:
        raise ValueError("The two runs use different outer splits")
    for field in ("class_weight", "feature_setting", "feature_columns", "window_overlap"):
        if weighted[field] != macro[field]:
            raise ValueError(f"The two runs differ in {field}")
    if weighted["bo_evaluations_used"] != N_CALLS or macro["bo_evaluations_used"] != N_CALLS:
        raise ValueError("The BO budgets differ")
    if macro["n_initial_points"] != N_INITIAL_POINTS:
        raise ValueError("Macro-F1 run does not have the planned BO budget")

    weighted_history = pd.read_csv(RESULTS_DIR / "grouped_bo_history.csv")
    macro_history = pd.read_csv(MACRO_DIR / "grouped_bo_history.csv")
    if len(weighted_history) != N_CALLS or len(macro_history) != N_CALLS:
        raise ValueError("Expected complete histories for both BO runs")
    for field in ("cv_folds", "class_weight", "feature_setting", "window_overlap"):
        if not weighted_history[field].equals(macro_history[field]):
            raise ValueError(f"The BO histories differ in {field}")
    first_weighted = weighted_history.loc[: N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy()
    first_macro = macro_history.loc[: N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy()
    if not np.allclose(first_weighted, first_macro, rtol=0, atol=1e-12):
        raise ValueError("The initial BO candidates differ")


def _row(label, result):
    row = {
        "objective": label,
        "best_iteration": result["best_iteration"],
        "validation_macro_f1": result["best_validation_macro_f1"],
    }
    for level in ("window", "segment"):
        metrics = result[f"{level}_level_test_metrics"]
        row[f"{level}_accuracy"] = metrics["accuracy"]
        row[f"{level}_macro_f1"] = metrics["macro_f1"]
        for name in ("N", "F", "S"):
            row[f"{level}_recall_{name}"] = metrics["per_class_recall"][name]
    return row


def main():
    weighted = _read_json(RESULTS_DIR / "grouped_bo_final_result.json")
    macro = _read_json(MACRO_DIR / "grouped_bo_final_result.json")
    _check_comparable(weighted, macro)
    comparison = pd.DataFrame([
        _row("N/F weighted", weighted),
        _row("macro-F1", macro),
    ])
    OBJECTIVE_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output = OBJECTIVE_COMPARISON_DIR / "objective_comparison.csv"
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"Saved {output}")
    print("Single outer split: use repeated splits before claiming a stable difference.")


if __name__ == "__main__":
    main()

"""Compare the old weighted objective with grouped segment-level accuracy."""

import argparse
import json

import numpy as np
import pandas as pd

from grouped_bo_optimizer import (
    CLASS_WEIGHT, CV_FOLDS, FEATURE_SETTING, N_CALLS, N_INITIAL_POINTS,
    OBJECTIVE_COMPARISON_DIR, RESULTS_DIR, SEGMENT_OBJECTIVE_DIR,
    WINDOW_OVERLAP,
)


PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")


def compare(seed=42):
    old_dir = (RESULTS_DIR if seed == 42 else
               OBJECTIVE_COMPARISON_DIR / "repeated" / f"seed_{seed}" / "nf_focused")
    new_dir = SEGMENT_OBJECTIVE_DIR / f"seed_{seed}" / "segment_accuracy"
    final_path = new_dir / "grouped_bo_final_result.json"
    if not final_path.exists():
        running = new_dir / "grouped_bo_history_running.csv"
        completed = len(pd.read_csv(running)) if running.exists() else 0
        print(f"Segment-accuracy run incomplete: {completed}/{N_CALLS}")
        return None

    old = json.loads((old_dir / "grouped_bo_final_result.json").read_text())
    new = json.loads(final_path.read_text())
    old_split = json.loads((old_dir / "grouped_split_manifest.json").read_text())
    new_split = json.loads((new_dir / "grouped_split_manifest.json").read_text())
    if old_split != new_split:
        raise ValueError("The outer recording split differs")
    if new["objective_mode"] != "segment_accuracy" or new["parameter_prior"] != "uniform":
        raise ValueError("Unexpected objective or parameter prior")
    if new["random_state"] != seed or new["cv_folds"] != CV_FOLDS:
        raise ValueError("Seed or CV folds differ")
    if new["n_initial_points"] != N_INITIAL_POINTS:
        raise ValueError("Initial evaluation budget differs")
    for result in (old, new):
        if result["bo_evaluations_used"] != N_CALLS or result["test_evaluations"] != 1:
            raise ValueError("BO or test evaluation budget differs")
        for key, expected in (("class_weight", CLASS_WEIGHT),
                              ("feature_setting", FEATURE_SETTING),
                              ("window_overlap", WINDOW_OVERLAP)):
            if result[key] != expected:
                raise ValueError(f"Unexpected {key}")

    old_history = pd.read_csv(old_dir / "grouped_bo_history.csv")
    new_history = pd.read_csv(new_dir / "grouped_bo_history.csv")
    if len(old_history) != N_CALLS or len(new_history) != N_CALLS:
        raise ValueError("An evaluation history is incomplete")
    for history in (old_history, new_history):
        if not history["cv_folds"].eq(CV_FOLDS).all():
            raise ValueError("CV fold count differs")
    if not np.allclose(
        old_history.loc[:N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy(),
        new_history.loc[:N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy(),
        rtol=0, atol=1e-12,
    ):
        raise ValueError("The first 16 BO candidates differ")

    rows = []
    for name, result in (("nf_focused", old), ("segment_accuracy", new)):
        segment = result["segment_level_test_metrics"]
        rows.append({
            "seed": seed,
            "objective": name,
            "best_iteration": result["best_iteration"],
            "test_segment_accuracy": segment["accuracy"],
            "test_segment_macro_f1": segment["macro_f1"],
            "test_recall_N": segment["per_class_recall"]["N"],
            "test_recall_F": segment["per_class_recall"]["F"],
            "test_recall_S": segment["per_class_recall"]["S"],
        })
    table = pd.DataFrame(rows)
    output = SEGMENT_OBJECTIVE_DIR / f"seed_{seed}_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"Saved {output}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compare(args.seed)

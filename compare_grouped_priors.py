"""Compare completed grouped BO preprocessing priors on one outer split."""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from grouped_bo_optimizer import (
    CLASS_WEIGHT, CV_FOLDS, FEATURE_SETTING, N_CALLS, N_INITIAL_POINTS,
    FIXED_FOLD_DIR, OBJECTIVE_COMPARISON_DIR, PROJECT_DIR, RESULTS_DIR,
    WINDOW_OVERLAP,
)
from parameter_priors import SUPPORTED_PRIORS


def _directory(seed, mode, prior):
    if mode == "segment_accuracy":
        return FIXED_FOLD_DIR / f"seed_{seed}" / prior
    if prior == "uniform":
        if seed == 42:
            return RESULTS_DIR if mode == "nf_focused" else OBJECTIVE_COMPARISON_DIR / mode
        return OBJECTIVE_COMPARISON_DIR / "repeated" / f"seed_{seed}" / mode
    return PROJECT_DIR / "results" / "06_grouped_prior_comparison" / f"seed_{seed}" / mode / prior


def compare(seed=42, mode="segment_accuracy"):
    baseline_dir = _directory(seed, mode, "uniform")
    baseline_manifest = json.loads((baseline_dir / "grouped_split_manifest.json").read_text())
    baseline_folds = (
        json.loads((baseline_dir / "grouped_cv_folds.json").read_text())
        if mode == "segment_accuracy" else None
    )
    rows = []
    for prior in SUPPORTED_PRIORS:
        directory = _directory(seed, mode, prior)
        final_path = directory / "grouped_bo_final_result.json"
        running_path = directory / "grouped_bo_history_running.csv"
        if not final_path.exists():
            completed = len(pd.read_csv(running_path)) if running_path.exists() else 0
            rows.append({"prior": prior, "status": f"incomplete ({completed}/{N_CALLS})"})
            continue
        result = json.loads(final_path.read_text())
        manifest = json.loads((directory / "grouped_split_manifest.json").read_text())
        history = pd.read_csv(directory / "grouped_bo_history.csv")
        if manifest != baseline_manifest:
            raise ValueError(f"Outer split differs: {directory}")
        if mode == "segment_accuracy" and (
            json.loads((directory / "grouped_cv_folds.json").read_text()) != baseline_folds
            or result.get("cv_split_protocol") != "recording_fixed_v2"
            or "cv_split_protocol" not in history
            or not history["cv_split_protocol"].eq("recording_fixed_v2").all()
            or not np.allclose(history["score"], history["validation_segment_accuracy"])
        ):
            raise ValueError(f"CV fold protocol or objective differs: {directory}")
        if len(history) != N_CALLS or result["bo_evaluations_used"] != N_CALLS:
            raise ValueError(f"BO budget differs: {directory}")
        if not history["cv_folds"].eq(CV_FOLDS).all():
            raise ValueError(f"CV folds differ: {directory}")
        for key, expected in (("class_weight", CLASS_WEIGHT),
                              ("feature_setting", FEATURE_SETTING),
                              ("window_overlap", WINDOW_OVERLAP)):
            if result[key] != expected:
                raise ValueError(f"{key} differs: {directory}")
        if not (seed == 42 and mode == "nf_focused" and prior == "uniform") and (
            result.get("parameter_prior") != prior
            or result.get("n_initial_points") != N_INITIAL_POINTS
            or result.get("objective_mode") != mode
            or result.get("random_state") != seed
            or result.get("test_evaluations") != 1
        ):
            raise ValueError(f"Prior, initial budget, or objective differs: {directory}")
        segment = result["segment_level_test_metrics"]
        rows.append({
            "prior": prior,
            "status": "complete",
            "best_cv_score": result["best_validation_score"],
            "segment_accuracy": segment["accuracy"],
            "segment_macro_f1": segment["macro_f1"],
            "segment_recall_N": segment["per_class_recall"]["N"],
            "segment_recall_F": segment["per_class_recall"]["F"],
            "segment_recall_S": segment["per_class_recall"]["S"],
        })
    table = pd.DataFrame(rows)
    output = (
        FIXED_FOLD_DIR if mode == "segment_accuracy"
        else PROJECT_DIR / "results" / "06_grouped_prior_comparison"
    ) / f"seed_{seed}_{mode}_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"Saved: {output}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-mode", choices=("nf_focused", "macro_f1", "segment_accuracy"), default="segment_accuracy")
    args = parser.parse_args()
    compare(args.seed, args.objective_mode)

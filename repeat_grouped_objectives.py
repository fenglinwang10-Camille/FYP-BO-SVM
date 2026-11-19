"""Repeat the paired grouped-BO objective comparison across outer splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from grouped_bo_optimizer import (
    CLASS_WEIGHT,
    CV_FOLDS,
    FEATURE_SETTING,
    N_CALLS,
    N_INITIAL_POINTS,
    OBJECTIVE_COMPARISON_DIR,
    RESULTS_DIR,
    WINDOW_OVERLAP,
    run_grouped_bo,
)


SEEDS = (42, 43, 44, 45, 46)
MODES = ("nf_focused", "macro_f1")
PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")
METRICS = (
    "segment_accuracy", "segment_macro_f1", "segment_recall_N",
    "segment_recall_F", "segment_recall_S", "window_accuracy",
    "window_macro_f1", "window_recall_N", "window_recall_F", "window_recall_S",
)


def _directory(seed, mode):
    if seed == 42:
        return RESULTS_DIR if mode == "nf_focused" else OBJECTIVE_COMPARISON_DIR / mode
    return OBJECTIVE_COMPARISON_DIR / "repeated" / f"seed_{seed}" / mode


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _completed_result(seed, mode):
    directory = _directory(seed, mode)
    final_path = directory / "grouped_bo_final_result.json"
    if not final_path.exists():
        return None
    result = _read_json(final_path)
    manifest = _read_json(directory / "grouped_split_manifest.json")
    history = pd.read_csv(directory / "grouped_bo_history.csv")
    if manifest["random_state"] != seed or len(history) != N_CALLS:
        raise ValueError(f"Incomplete or mismatched saved run: {directory}")
    if result.get("bo_evaluations_used") != N_CALLS or result.get("test_evaluations") != 1:
        raise ValueError(f"Unexpected evaluation budget: {directory}")
    if seed != 42 and result.get("objective_mode") != mode:
        raise ValueError(f"Unexpected objective mode: {directory}")
    for key, expected in (
        ("feature_setting", FEATURE_SETTING),
        ("class_weight", CLASS_WEIGHT),
        ("window_overlap", WINDOW_OVERLAP),
    ):
        if result[key] != expected:
            raise ValueError(f"Unexpected {key}: {directory}")
    if not history["cv_folds"].eq(CV_FOLDS).all():
        raise ValueError(f"Unexpected CV folds: {directory}")
    if seed != 42 and result.get("n_initial_points") != N_INITIAL_POINTS:
        raise ValueError(f"Unexpected initial budget: {directory}")
    return result


def _check_pair(seed):
    left_dir = _directory(seed, MODES[0])
    right_dir = _directory(seed, MODES[1])
    if _read_json(left_dir / "grouped_split_manifest.json") != _read_json(
        right_dir / "grouped_split_manifest.json"
    ):
        raise ValueError(f"Outer splits differ for seed {seed}")
    left = pd.read_csv(left_dir / "grouped_bo_history.csv")
    right = pd.read_csv(right_dir / "grouped_bo_history.csv")
    for field in ("cv_folds", "class_weight", "feature_setting", "window_overlap"):
        if not left[field].equals(right[field]):
            raise ValueError(f"Histories differ in {field} for seed {seed}")
    if not np.allclose(
        left.loc[: N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy(),
        right.loc[: N_INITIAL_POINTS - 1, list(PARAMETERS)].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"Initial BO candidates differ for seed {seed}")


def _row(seed, mode, result):
    row = {
        "seed": seed,
        "objective": mode,
        "best_iteration": result["best_iteration"],
        "validation_macro_f1": result["best_validation_macro_f1"],
    }
    for level in ("segment", "window"):
        metrics = result[f"{level}_level_test_metrics"]
        row[f"{level}_accuracy"] = metrics["accuracy"]
        row[f"{level}_macro_f1"] = metrics["macro_f1"]
        for name in ("N", "F", "S"):
            row[f"{level}_recall_{name}"] = metrics["per_class_recall"][name]
    return row


def _summary(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"n": n, "mean": mean, "sd": np.nan,
                "ci95_lower": np.nan, "ci95_upper": np.nan}
    sd = float(np.std(values, ddof=1))
    half_width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_lower": mean - half_width, "ci95_upper": mean + half_width}


def _save_summaries(rows):
    output = OBJECTIVE_COMPARISON_DIR / "repeated"
    output.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows).sort_values(["seed", "objective"])
    per_seed.to_csv(output / "per_seed.csv", index=False)

    paired = []
    for seed, group in per_seed.groupby("seed"):
        if set(group["objective"]) != set(MODES):
            continue
        by_mode = group.set_index("objective")
        paired.append({
            "seed": seed,
            **{metric: float(by_mode.loc["macro_f1", metric]
                             - by_mode.loc["nf_focused", metric]) for metric in METRICS},
        })
    paired_df = pd.DataFrame(paired)
    if paired_df.empty:
        return
    paired_df.to_csv(output / "paired_differences.csv", index=False)

    summaries = []
    for mode, group in per_seed.groupby("objective"):
        for metric in METRICS:
            summaries.append({"objective": mode, "metric": metric,
                              **_summary(group[metric])})
    pd.DataFrame(summaries).to_csv(output / "summary_by_objective.csv", index=False)
    paired_summaries = [
        {"metric": metric, "difference": "macro_f1 - nf_focused",
         **_summary(paired_df[metric])}
        for metric in METRICS
    ]
    pd.DataFrame(paired_summaries).to_csv(output / "paired_summary.csv", index=False)
    print(f"Saved summaries for {len(paired_df)} paired outer splits.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    arguments = parser.parse_args()
    rows = []
    for seed in arguments.seeds:
        results = {}
        feature_cache = {}
        for mode in MODES:
            result = _completed_result(seed, mode)
            if result is None:
                if seed == 42:
                    raise FileNotFoundError(f"Archived seed 42 run is missing: {mode}")
                print(f"Starting seed {seed}, objective {mode}", flush=True)
                run_grouped_bo(
                    objective_mode=mode,
                    output_dir=_directory(seed, mode),
                    random_state=seed,
                    feature_cache=feature_cache,
                )
                result = _completed_result(seed, mode)
            else:
                print(f"Using completed seed {seed}, objective {mode}", flush=True)
            results[mode] = result
        _check_pair(seed)
        rows.extend(_row(seed, mode, results[mode]) for mode in MODES)
        _save_summaries(rows)
        feature_cache.clear()
    print("Repeated objective comparison completed.", flush=True)


if __name__ == "__main__":
    main()

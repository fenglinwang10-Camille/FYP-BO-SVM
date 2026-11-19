"""Nested recording-level CV with paired BO/RS random initialization.

Only the training recordings are passed to candidate evaluation. An immutable
protocol and split manifest are written before searching; finished outer-test
results are reused on restart. Original 09_* results are never overwritten.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from skopt import Optimizer

from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    CLASS_NAMES, CLASS_WEIGHT, DATASET_PATH, FEATURE_SETTING, FS, PROJECT_DIR,
    SEARCH_SPACE, WINDOW_OVERLAP, _build_feature_matrix,
    _cross_validate_training_segments, _flatten_history, _json_metrics,
    _majority_vote_by_segment, _make_model, _metrics, _select_segments,
)
from grouped_method_comparison import MANUAL_PARAMS


PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "10_nested_cv_shared_initialization"
N_FOLDS = 5


def _json_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv_save(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Protocol or saved inputs changed: {path}. Use a new output directory.")
    else:
        _json_save(path, value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_splits(labels, outer_seed=42, inner_seed=1729):
    """Split original recordings before any windows or features are created."""
    labels = np.asarray(labels, dtype=int)
    counts = Counter(labels.tolist())
    if counts != Counter({i: 100 for i in range(5)}):
        raise ValueError(f"Expected 500 recordings, 100 per class; got {counts}")
    outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=outer_seed)
    splits = []
    test_coverage = []
    for number, (train, test) in enumerate(outer.split(np.zeros(len(labels)), labels), 1):
        actual_inner_seed = inner_seed + number - 1
        inner = StratifiedKFold(N_FOLDS, shuffle=True, random_state=actual_inner_seed)
        folds = []
        validation_coverage = []
        for k, (a, b) in enumerate(inner.split(np.zeros(len(train)), labels[train]), 1):
            fit_ids, validation_ids = train[a], train[b]
            assert not set(fit_ids) & set(validation_ids)
            assert not (set(fit_ids) | set(validation_ids)) & set(test)
            assert Counter(labels[validation_ids]) == Counter({i: 16 for i in range(5)})
            folds.append({"fold": k, "train_segment_ids": fit_ids.tolist(),
                          "validation_segment_ids": validation_ids.tolist()})
            validation_coverage.extend(validation_ids.tolist())
        assert sorted(validation_coverage) == sorted(train.tolist())
        assert Counter(labels[test]) == Counter({i: 20 for i in range(5)})
        splits.append({"outer_fold": number, "inner_split_seed": actual_inner_seed,
                       "train_segment_ids": train.tolist(), "test_segment_ids": test.tolist(),
                       "inner_folds": folds})
        test_coverage.extend(test.tolist())
    assert sorted(test_coverage) == list(range(len(labels)))
    return splits


def random_points(seed, count):
    """Independent uniform preprocessing and log-uniform SVM parameters."""
    unit = np.random.default_rng(seed).random((count, len(PARAMETERS)))
    values = np.empty_like(unit)
    for column, dimension in enumerate(SEARCH_SPACE):
        if dimension.prior == "log-uniform":
            values[:, column] = np.exp(np.log(dimension.low) + unit[:, column]
                                       * np.log(dimension.high / dimension.low))
        else:
            values[:, column] = dimension.low + unit[:, column] * (dimension.high - dimension.low)
    return values.tolist()


def make_protocol(signals, labels, args):
    data_hash = hashlib.sha256()
    for signal, label in zip(signals, labels):
        values = np.asarray(signal, dtype="<f8")
        data_hash.update(np.asarray([len(values), int(label)], dtype="<i8").tobytes())
        data_hash.update(values.tobytes())
    sources = ("nested_bo_rs.py", "grouped_bo_optimizer.py", "grouped_method_comparison.py",
               "features.py", "preprocessing.py", "data_loader.py", "parameter_priors.py")
    return {
        "protocol": "nested_recording_cv_shared_initialization_v1",
        "outer_folds": 5, "inner_folds": 5,
        "outer_split_seed": args.outer_split_seed, "inner_split_seed": args.inner_split_seed,
        "search_seed": args.search_seed, "evaluations_per_method": 50,
        "shared_initial_evaluations": 16, "parameter_mapping": "uniform",
        "search_space": {d.name: {"low": d.low, "high": d.high, "prior": d.prior}
                         for d in SEARCH_SPACE},
        "acquisition_function": "EI", "xi": 0.01,
        "acquisition_optimizer": "lbfgs", "feature_setting": FEATURE_SETTING,
        "feature_columns": 55, "sampling_frequency": FS,
        "class_weight": CLASS_WEIGHT, "window_overlap": WINDOW_OVERLAP,
        "vote_rule": "majority_vote; ties use smallest class index, unchanged from original study",
        "objective": "mean inner-fold recording accuracy after window majority vote",
        "selection_ties": "earliest candidate with the highest inner CV accuracy",
        "fixed_reference_parameters": dict(MANUAL_PARAMS),
        "dataset_sha256": data_hash.hexdigest(),
        "source_sha256": {s: hashlib.sha256((PROJECT_DIR / s).read_bytes()).hexdigest() for s in sources},
        "versions": {p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn",
                                             "scikit-optimize", "PyWavelets", "joblib")},
        "python": platform.python_version(),
        "limitations": "Recording IDs are not subject IDs. Data were inspected in earlier development; "
                       "this is a prospectively fixed reanalysis of the existing dataset, not new external data.",
    }


@contextmanager
def run_lock(directory):
    """The OS releases this byte lock even if Python is interrupted."""
    lock_path = Path(directory) / "run.lock"
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"Another process is already running this experiment: {directory}") from None
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _save_state(directory, state):
    temporary = directory / "search_state.joblib.tmp"
    joblib.dump(state, temporary, compress=0)
    temporary.replace(directory / "search_state.joblib")
    shared = state["shared"]
    for method in ("bo", "rs"):
        rows = [dict(row, method=method.upper()) for row in shared + state[method]]
        if rows:
            _csv_save(directory / f"{method}_history.csv", rows)
    if shared:
        _csv_save(directory / "shared_initial_history.csv", shared)
    _json_save(directory / "progress.json", {
        "shared_completed": len(shared), "bo_completed": len(shared) + len(state["bo"]),
        "rs_completed": len(shared) + len(state["rs"]),
        "unique_training_cv_evaluations": len(shared) + len(state["bo"]) + len(state["rs"]),
        "phase": "searching" if len(state["bo"]) + len(shared) < state["budget"]
                 or len(state["rs"]) + len(shared) < state["budget"] else "search_complete",
    })


def _evaluate_candidate(train_data, folds, point, iteration, phase):
    params = dict(zip(PARAMETERS, map(float, point)))
    started = perf_counter()
    score, metrics, shape, _ = _cross_validate_training_segments(
        *train_data, params, objective_mode="segment_accuracy", fixed_folds=folds,
    )
    row = _flatten_history(iteration, params, score, metrics, shape)
    row.update(phase=phase, candidate_seconds=perf_counter() - started,
               cv_split_protocol="nested_recording_fixed_inner_v1")
    return row


def run_searches(train_data, split, protocol, directory, max_new=None):
    """Test recordings are deliberately absent from this interface."""
    budget, n_initial = protocol["evaluations_per_method"], protocol["shared_initial_evaluations"]
    points = random_points(protocol["search_seed"], budget)
    signature = _digest({"protocol": protocol, "split": split})
    checkpoint = directory / "search_state.joblib"
    state = joblib.load(checkpoint) if checkpoint.exists() else {
        "signature": signature, "budget": budget, "shared": [], "bo": [], "rs": [],
        "optimizer": None,
    }
    if state["signature"] != signature:
        raise ValueError("Checkpoint belongs to a different experiment")
    folds = [(np.asarray(f["train_segment_ids"]), np.asarray(f["validation_segment_ids"]))
             for f in split["inner_folds"]]
    _immutable_json(directory / "shared_initial_candidates.json", [
        dict(iteration=i + 1, **dict(zip(PARAMETERS, point)))
        for i, point in enumerate(points[:n_initial])
    ])
    completed_now = 0
    while len(state["shared"]) < n_initial:
        if max_new is not None and completed_now >= max_new:
            return state, False
        iteration = len(state["shared"]) + 1
        row = _evaluate_candidate(train_data, folds, points[iteration - 1], iteration, "shared_initial")
        state["shared"].append(row)
        _save_state(directory, state)
        completed_now += 1
        print(f"Fold {split['outer_fold']} shared {iteration}/{n_initial}: "
              f"CV={row['score']:.4f}, {row['candidate_seconds']:.1f}s", flush=True)
    if state["optimizer"] is None:
        started = perf_counter()
        optimizer = Optimizer(
            SEARCH_SPACE, n_initial_points=n_initial, random_state=protocol["search_seed"],
            acq_func="EI", acq_func_kwargs={"xi": protocol["xi"]},
            acq_optimizer="lbfgs", model_queue_size=1,
        )
        optimizer.tell(points[:n_initial], [-row["score"] for row in state["shared"]])
        state["optimizer"] = optimizer
        state["initial_optimizer_seconds"] = perf_counter() - started
        _save_state(directory, state)
    for offset in range(budget - n_initial):
        for method in ("bo", "rs"):
            if len(state[method]) > offset:
                continue
            if max_new is not None and completed_now >= max_new:
                return state, False
            iteration = n_initial + offset + 1
            started = perf_counter()
            point = state["optimizer"].ask() if method == "bo" else points[iteration - 1]
            ask_seconds = perf_counter() - started
            row = _evaluate_candidate(train_data, folds, point, iteration,
                                      "guided" if method == "bo" else "random_continuation")
            started = perf_counter()
            if method == "bo":
                state["optimizer"].tell(point, -row["score"])
            row["optimizer_seconds"] = ask_seconds + perf_counter() - started
            state[method].append(row)
            _save_state(directory, state)
            completed_now += 1
            print(f"Fold {split['outer_fold']} {method.upper()} {iteration}/{budget}: "
                  f"CV={row['score']:.4f}, {row['candidate_seconds']:.1f}s", flush=True)
    # Export again in case a previous process exited between checkpoint and CSV writes.
    _save_state(directory, state)
    return state, True


def selected_configurations(state, protocol):
    selected = {}
    for method in ("bo", "rs"):
        best = max(state["shared"] + state[method], key=lambda row: row["score"])
        selected[method] = {"best_iteration": best["iteration"], "best_inner_cv_accuracy": best["score"],
                            "parameters": {name: best[name] for name in PARAMETERS}}
    selected["fixed_reference"] = {"parameters": protocol["fixed_reference_parameters"],
                                    "best_iteration": None, "best_inner_cv_accuracy": None}
    return selected


def evaluate_outer(train_data, test_data, params):
    started = perf_counter()
    X_train, y_train, train_groups = _build_feature_matrix(*train_data, params)
    model = _make_model(params)
    model.fit(X_train, y_train)
    X_test, y_test, test_groups = _build_feature_matrix(*test_data, params)
    if set(train_groups) & set(test_groups):
        raise RuntimeError("Outer training and test recordings overlap")
    predictions = model.predict(X_test)
    truth, voted = _majority_vote_by_segment(predictions, y_test, test_groups)
    recording_ids = np.unique(test_groups)
    return {
        "segment_level_test_metrics": _json_metrics(_metrics(truth, voted)),
        "window_level_test_metrics": _json_metrics(_metrics(y_test, predictions)),
        "outer_evaluation_seconds": perf_counter() - started,
        "recording_predictions": [dict(recording_id=int(r), true_label=int(y), predicted_label=int(p))
                                  for r, y, p in zip(recording_ids, truth, voted)],
    }


def summarize(output, protocol):
    rows, predictions = [], []
    for number in range(1, 6):
        directory = output / f"outer_fold_{number}"
        for method in ("bo", "rs", "fixed_reference"):
            path = directory / f"{method}_outer_result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            metrics = result["segment_level_test_metrics"]
            row = dict(outer_fold=number, method=method, search_seed=protocol["search_seed"],
                       best_inner_cv_accuracy=result["selection"]["best_inner_cv_accuracy"],
                       best_iteration=result["selection"]["best_iteration"],
                       outer_accuracy=metrics["accuracy"], outer_macro_recall=metrics["macro_recall"],
                       outer_seizure_sensitivity=metrics["per_class_recall"]["S"],
                       **result["selection"]["parameters"])
            row.update({f"outer_recall_{name}": metrics["per_class_recall"][name] for name in CLASS_NAMES})
            rows.append(row)
            predictions.extend(dict(p, outer_fold=number, method=method) for p in result["recording_predictions"])
    if rows:
        _csv_save(output / "outer_fold_results.csv", rows)
        _csv_save(output / "outer_recording_predictions.csv", predictions)
    if len(rows) != 15:
        return False
    summaries = []
    for method in ("bo", "rs", "fixed_reference"):
        selected_rows = [r for r in rows if r["method"] == method]
        selected_predictions = sorted([p for p in predictions if p["method"] == method],
                                      key=lambda p: p["recording_id"])
        assert [p["recording_id"] for p in selected_predictions] == list(range(500))
        metrics = _json_metrics(_metrics(
            [p["true_label"] for p in selected_predictions],
            [p["predicted_label"] for p in selected_predictions],
        ))
        summaries.append(dict(method=method, folds=5,
                              mean_outer_accuracy=float(np.mean([r["outer_accuracy"] for r in selected_rows])),
                              sd_outer_accuracy=float(np.std([r["outer_accuracy"] for r in selected_rows], ddof=1)),
                              pooled_outer_metrics=metrics))
    table = pd.DataFrame(rows).pivot(index="outer_fold", columns="method", values="outer_accuracy")
    differences = [dict(outer_fold=int(k), bo_accuracy=float(r["bo"]), rs_accuracy=float(r["rs"]),
                        bo_minus_rs_percentage_points=float(100 * (r["bo"] - r["rs"])))
                   for k, r in table.iterrows()]
    _csv_save(output / "paired_bo_minus_rs.csv", differences)
    _json_save(output / "summary.json", {"protocol": protocol, "methods": summaries,
                                         "fold_variability_is_descriptive": True})
    return True


def run(args):
    signals, labels = load_bonn_dataset(DATASET_PATH)
    splits = make_splits(labels, args.outer_split_seed, args.inner_split_seed)
    protocol = make_protocol(signals, labels, args)
    output = Path(args.output_dir) / f"split_{args.outer_split_seed}_inner_{args.inner_split_seed}" / f"search_{args.search_seed}"
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        _immutable_json(output / "protocol.json", protocol)
        _immutable_json(output / "nested_folds.json", splits)
        if args.prepare_only:
            print(f"Verified all outer/inner folds; no model was fitted. Output: {output}", flush=True)
            return
        _json_save(output / "run_status.json", {"status": "running", "pid": os.getpid()})
        try:
            for split in splits:
                if args.folds and split["outer_fold"] not in args.folds:
                    continue
                directory = output / f"outer_fold_{split['outer_fold']}"
                directory.mkdir(exist_ok=True)
                _immutable_json(directory / "fold_manifest.json", split)
                train_data = _select_segments(signals, labels, split["train_segment_ids"])
                state, complete = run_searches(train_data, split, protocol, directory, args.max_new_evaluations)
                if not complete:
                    _json_save(output / "run_status.json", {"status": "paused_at_evaluation_limit", "pid": os.getpid()})
                    return
                selected = selected_configurations(state, protocol)
                _immutable_json(directory / "selected_configurations.json", selected)
                # Selection is now frozen. Only this step receives outer test data.
                for method, selection in selected.items():
                    final_path = directory / f"{method}_outer_result.json"
                    signature = _digest({"protocol": protocol, "split": split, "selection": selection})
                    if final_path.exists():
                        existing = json.loads(final_path.read_text(encoding="utf-8"))
                        if existing["signature"] != signature:
                            raise ValueError(f"Saved outer result no longer matches selection: {final_path}")
                        continue
                    result = evaluate_outer(train_data, _select_segments(signals, labels, split["test_segment_ids"]),
                                            selection["parameters"])
                    result.update(signature=signature, method=method, outer_fold=split["outer_fold"],
                                  selection=selection, completed_outer_evaluations=1)
                    _json_save(final_path, result)
                    print(f"Fold {split['outer_fold']} {method}: outer prediction saved", flush=True)
                summarize(output, protocol)
            completed = summarize(output, protocol)
            _json_save(output / "run_status.json", {"status": "complete" if completed else "partial_folds_complete",
                                                   "pid": os.getpid()})
            print(f"{'COMPLETE' if completed else 'PARTIAL'}: {output}", flush=True)
        except BaseException as error:
            _json_save(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   "error": str(error), "pid": os.getpid()})
            raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-split-seed", type=int, default=42)
    parser.add_argument("--inner-split-seed", type=int, default=1729)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, 6))
    parser.add_argument("--max-new-evaluations", type=int)
    args = parser.parse_args()
    if args.max_new_evaluations is not None and args.max_new_evaluations < 1:
        parser.error("--max-new-evaluations must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())

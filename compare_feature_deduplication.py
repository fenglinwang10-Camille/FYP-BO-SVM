"""Paired training-only 55/53 feature screen with equal SVM search budgets.

CV is used for tuning and comparison, so selected scores are exploratory,
not unbiased generalisation estimates. No held-out test features are built.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from analyze_grouped_features import REFERENCE_PARAMS, _saved_training_folds
from data_loader import load_bonn_dataset
from grouped_bo_optimizer import (
    DATASET_PATH, FIXED_FOLD_DIR, _build_feature_matrix, _make_model,
    _majority_vote_by_segment, _metrics, _select_segments,
)

# nf_enhanced ordering: powers 0:5, relative powers 5:10, ratios 10:16,
# statistics 16:29, entropies 29:32, Hjorth 32:35, wavelets 35:55.
REMOVED = {15: "near-duplicate gamma relative power", 32: "Hjorth activity duplicates variance"}
KEEP = [i for i in range(55) if i not in REMOVED]
OUTPUT = FIXED_FOLD_DIR / "feature_deduplication_tuned_cv"


def evaluate(X, y, groups, folds, params):
    fold_scores, predictions = [], []
    seen = set()
    for number, fold in enumerate(folds, 1):
        tr = np.asarray(fold["train_segment_ids"])
        va = np.asarray(fold["validation_segment_ids"])
        assert not np.intersect1d(tr, va).size
        train, valid = np.isin(groups, tr), np.isin(groups, va)
        model = _make_model(params)
        model.fit(X[train], y[train])
        truth, pred = _majority_vote_by_segment(model.predict(X[valid]), y[valid], groups[valid])
        fold_scores.append(_metrics(truth, pred)["accuracy"])
        for rid, actual, predicted in zip(np.unique(groups[valid]), truth, pred):
            assert int(rid) not in seen
            seen.add(int(rid))
            predictions.append(dict(fold=number, recording_id=int(rid),
                                    true_label=int(actual), predicted_label=int(predicted)))
    assert seen == set(np.unique(groups).tolist())
    metrics = _metrics(np.array([r["true_label"] for r in predictions]),
                       np.array([r["predicted_label"] for r in predictions]))
    return float(np.mean(fold_scores)), metrics, predictions


def run(seeds, budget, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=list(seeds), budget_per_version_per_seed=budget,
        preprocessing={k: REFERENCE_PARAMS[k] for k in ("f_low", "f_high", "window_size")},
        versions={"original_55": list(range(55)), "deduplicated_53": KEEP},
        removed_columns=REMOVED, svm_search="reference candidate plus paired log-uniform random candidates",
        C_bounds=[0.1, 100], gamma_bounds=[0.0001, 1],
        objective="mean fold recording accuracy after majority vote",
        held_out_test_evaluations=0,
        limitations="Tuning and comparison reuse training CV; scores are selection-biased. Outer training splits overlap.",
        source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                       for name in ("features.py", "preprocessing.py", "grouped_bo_optimizer.py", __file__)})
    protocol = json.loads(json.dumps(protocol))
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different protocol; use a new output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    signals, labels = load_bonn_dataset(DATASET_PATH, classification="five-class")
    labels = np.asarray(labels, dtype=int)
    summaries = []
    for seed in seeds:
        train_ids, folds = _saved_training_folds(seed, labels)
        train_signals, train_labels, ids = _select_segments(signals, labels, train_ids)
        start = perf_counter()
        X, y, groups = _build_feature_matrix(train_signals, train_labels, ids, REFERENCE_PARAMS)
        extraction_seconds = perf_counter() - start
        assert X.shape[1] == 55 and np.isfinite(X).all()
        np.testing.assert_array_equal(X[:, 18], X[:, 32])
        # Relative gamma differs only in placement of the numerical epsilon.
        np.testing.assert_allclose(X[:, 9], X[:, 15], rtol=1e-6, atol=1e-12)
        audit = dict(extraction_seconds=extraction_seconds, rows=len(X),
                     gamma_max_difference=float(np.max(np.abs(X[:, 9]-X[:, 15]))),
                     variance_max_difference=float(np.max(np.abs(X[:, 18]-X[:, 32]))))
        (output / f"seed_{seed}_audit.json").write_text(json.dumps(audit, indent=2))
        rng = np.random.default_rng(seed)
        candidates = [(REFERENCE_PARAMS["C"], REFERENCE_PARAMS["gamma"])]
        candidates += [(float(np.exp(rng.uniform(np.log(.1), np.log(100)))),
                        float(np.exp(rng.uniform(np.log(1e-4), 0)))) for _ in range(budget-1)]
        for version, indices in protocol["versions"].items():
            path = output / f"seed_{seed}_{version}_history.csv"
            history = pd.read_csv(path).to_dict("records") if path.exists() else []
            assert len(history) <= budget
            for row, (C, gamma) in zip(history, candidates):
                np.testing.assert_allclose([row["C"], row["gamma"]], [C, gamma], rtol=1e-12)
            for iteration in range(len(history), budget):
                C, gamma = candidates[iteration]
                start = perf_counter()
                score, metrics, _ = evaluate(X[:, indices], y, groups, folds,
                                            {**REFERENCE_PARAMS, "C": C, "gamma": gamma})
                history.append(dict(iteration=iteration+1, C=C, gamma=gamma,
                    cv_accuracy=score, cv_macro_f1=metrics["macro_f1"],
                    seconds=perf_counter()-start))
                pd.DataFrame(history).to_csv(path, index=False)
                print(f"seed={seed} {version} {iteration+1}/{budget} CV={score:.4f}", flush=True)
            best = max(history, key=lambda row: row["cv_accuracy"])
            score, metrics, predictions = evaluate(X[:, indices], y, groups, folds,
                {**REFERENCE_PARAMS, "C": best["C"], "gamma": best["gamma"]})
            assert np.isclose(score, best["cv_accuracy"])
            pd.DataFrame(predictions).to_csv(output / f"seed_{seed}_{version}_selected_cv_predictions.csv", index=False)
            summaries.append(dict(seed=seed, version=version, n_features=len(indices),
                **best, search_seconds=sum(r["seconds"] for r in history),
                cv_recall_F=metrics["per_class_recall"]["F"]))
            pd.DataFrame(summaries).to_csv(output / "per_seed.csv", index=False)
    frame = pd.DataFrame(summaries)
    summary = frame.groupby("version").agg(n=("seed", "size"),
        cv_accuracy_mean=("cv_accuracy", "mean"), cv_accuracy_sd=("cv_accuracy", "std"),
        cv_macro_f1_mean=("cv_macro_f1", "mean"), search_seconds_mean=("search_seconds", "mean"))
    summary.to_csv(output / "summary.csv")
    paired = frame.pivot(index="seed", columns="version", values="cv_accuracy")
    paired["difference_53_minus_55"] = paired["deduplicated_53"]-paired["original_55"]
    paired.to_csv(output / "paired_differences.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,43,44,45,46])
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.budget < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive budget and unique seeds required")
    run(args.seeds, args.budget, args.output)

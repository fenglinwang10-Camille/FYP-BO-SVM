from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

from data_loader import load_bonn_dataset
import experiment_2_notch_extended_handcrafted_features_runner as runner


def load_existing_payload(method_key: str, display_name: str):
    seg = pd.read_csv(runner.RESULTS_DIR / f"{method_key}_segment_level_metrics.csv").iloc[0].to_dict()
    win = pd.read_csv(runner.RESULTS_DIR / f"{method_key}_window_level_metrics.csv").iloc[0].to_dict()
    params = json.loads((runner.RESULTS_DIR / f"{method_key}_best_parameters.json").read_text(encoding="utf-8"))
    return {
        "method": display_name,
        "parameters": params,
        "best_grouped_cv_objective": float(pd.read_csv(runner.RESULTS_DIR / "final_summary_table.csv").set_index("method").loc[display_name, "best_grouped_cv_objective"]) if (runner.RESULTS_DIR / "final_summary_table.csv").exists() and display_name in pd.read_csv(runner.RESULTS_DIR / "final_summary_table.csv")["method"].tolist() else float("nan"),
        "segment_test_metrics": {
            "accuracy": float(seg["accuracy"]),
            "macro_recall": float(seg["macro_recall"]),
            "macro_f1": float(seg["macro_f1"]),
            "seizure_sensitivity": float(seg["seizure_sensitivity"]),
            "per_class_recall": {cls: float(seg[f"recall_{cls}"]) for cls in runner.CLASS_NAMES},
            "confusion_matrix": pd.read_csv(runner.RESULTS_DIR / f"confusion_matrix_{method_key}_segment_level.csv").drop(columns=["true_label"]).to_numpy(dtype=int).tolist(),
        },
        "window_test_metrics": {
            "accuracy": float(win["accuracy"]),
            "macro_recall": float(win["macro_recall"]),
            "macro_f1": float(win["macro_f1"]),
            "seizure_sensitivity": float(win["seizure_sensitivity"]),
            "per_class_recall": {cls: float(win[f"recall_{cls}"]) for cls in runner.CLASS_NAMES},
        },
    }


def main() -> None:
    raw_signals, raw_labels = load_bonn_dataset(runner.DATASET_PATH, classification="five-class")
    raw_signals = np.asarray(raw_signals, dtype=object)
    raw_labels = np.asarray(raw_labels, dtype=int)
    train_ids, test_ids, _ = runner.load_outer_split()
    train_ids = train_ids.astype(int)
    test_ids = test_ids.astype(int)
    train_signals, train_labels = raw_signals[train_ids], raw_labels[train_ids]
    test_signals, test_labels = raw_signals[test_ids], raw_labels[test_ids]

    fold_df = pd.read_csv(runner.RESULTS_DIR / runner.RESULT_FILENAMES["grouped_fold_assignments"])
    fold_map = {int(row.segment_id): int(row.fold_id) for row in fold_df.itertuples(index=False)}

    manual_config, manual_cv_result, manual_params = runner.run_manual(train_signals, train_labels, train_ids, fold_map)
    manual_holdout = runner.holdout_evaluate(
        runner.apply_notch(train_signals),
        train_labels,
        train_ids,
        runner.apply_notch(test_signals),
        test_labels,
        test_ids,
        manual_params,
    )
    manual_payload = runner.write_metric_artifacts(
        "Fixed conventional RBF-SVM baseline",
        "manual",
        manual_cv_result,
        manual_holdout,
        manual_params,
        manual_cv_result["objective"],
        output_dir=runner.RESULTS_DIR,
    )
    manual_payload["manual_config"] = manual_config

    bo_payload = load_existing_payload("bo", "BO")
    random_payload = load_existing_payload("random", "Random Search")
    manual_payload["best_grouped_cv_objective"] = float(manual_cv_result["objective"])

    method_payloads = [bo_payload, random_payload, manual_payload]
    plot_payloads = [
        bo_payload,
        random_payload,
        {**manual_payload, "method": "Manual SVM"},
    ]
    runner.save_method_comparison(method_payloads)
    runner.save_recall_comparison([
        {
            "method": payload["method"],
            **{f"segment_recall_{cls}": payload["segment_test_metrics"]["per_class_recall"][cls] for cls in runner.CLASS_NAMES},
        }
        for payload in plot_payloads
    ])
    runner.save_selected_parameters_summary(method_payloads, runner.FIGURES_DIR / runner.RESULT_FILENAMES["selected_parameters_summary"])

    summary_rows = []
    for payload in method_payloads:
        seg = payload["segment_test_metrics"]
        win = payload["window_test_metrics"]
        summary_rows.append({
            "method": payload["method"],
            "best_grouped_cv_objective": payload["best_grouped_cv_objective"],
            "segment_accuracy": seg["accuracy"],
            "segment_macro_recall": seg["macro_recall"],
            "segment_recall_Z": seg["per_class_recall"]["Z"],
            "segment_recall_O": seg["per_class_recall"]["O"],
            "segment_recall_N": seg["per_class_recall"]["N"],
            "segment_recall_F": seg["per_class_recall"]["F"],
            "segment_recall_S": seg["per_class_recall"]["S"],
            "segment_seizure_sensitivity": seg["seizure_sensitivity"],
            "window_accuracy": win["accuracy"],
            "window_macro_recall": win["macro_recall"],
            "window_recall_Z": win["per_class_recall"]["Z"],
            "window_recall_O": win["per_class_recall"]["O"],
            "window_recall_N": win["per_class_recall"]["N"],
            "window_recall_F": win["per_class_recall"]["F"],
            "window_recall_S": win["per_class_recall"]["S"],
            "window_seizure_sensitivity": win["seizure_sensitivity"],
            "parameters": json.dumps(payload["parameters"]),
        })
    pd.DataFrame(summary_rows).to_csv(runner.RESULTS_DIR / "final_summary_table.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    import numpy as np

    main()

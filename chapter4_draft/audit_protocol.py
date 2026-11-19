"""Check the manuscript settings against existing current-run manifests."""
from pathlib import Path
import json
import csv
ROOT = Path(__file__).resolve().parents[1]
report = []
for seed in range(42, 47):
    reference = None
    for prior in ("uniform", "normal", "exponential", "gamma"):
        directory = ROOT / "results/09_fixed_fold_segment_accuracy" / f"seed_{seed}" / prior
        result = json.loads((directory / "grouped_bo_final_result.json").read_text())
        folds = json.loads((directory / "grouped_cv_folds.json").read_text())
        split = json.loads((directory / "grouped_split_manifest.json").read_text())
        assert result["cv_split_protocol"] == "recording_fixed_v2"
        assert result["bo_evaluations_used"] == 50
        assert result["n_initial_points"] == 16
        assert result["feature_columns"] == 55
        assert len(folds) == 5
        for fold in folds:
            a, b = fold["train_segment_ids"], fold["validation_segment_ids"]
            assert len(a) == 320 and len(b) == 80 and not set(a) & set(b)
        if reference is not None:
            assert (folds, split) == reference
        reference = folds, split
        with (directory / "grouped_bo_history.csv").open(newline="", encoding="utf-8-sig") as handle:
            assert len(list(csv.DictReader(handle))) == 50
        report.append({"seed": seed, "mapping": prior, "evaluations": 50,
                       "initial_points": 16, "feature_columns": 55})
payload = {"runs": report, "shared_folds_and_splits_within_seed": True,
           "fold_sizes": [320, 80], "feature_column_sum": 10+6+13+3+3+20,
           "source": "Existing manifests and metadata; no experiment rerun"}
(Path(__file__).parent / "protocol_audit.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8")
print("Verified all 20 runs: matched splits/folds within seed, 320/80 folds,")
print("50 history rows per run, 16 initial points and 55 feature columns.")

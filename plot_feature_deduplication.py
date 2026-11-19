"""Plot and validate completed paired feature comparison outputs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_feature_deduplication import OUTPUT


def main():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    results = pd.read_csv(OUTPUT / "per_seed.csv")
    assert len(results) == 2*len(protocol["seeds"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    colors = {"original_55": "#446a97", "deduplicated_53": "#c66d39"}
    labels = {"original_55": "Original (55)", "deduplicated_53": "Deduplicated (53)"}
    curves = {name: [] for name in colors}
    for seed in protocol["seeds"]:
        reference = None
        for version in colors:
            h = pd.read_csv(OUTPUT / f"seed_{seed}_{version}_history.csv")
            assert len(h) == protocol["budget_per_version_per_seed"]
            candidates = h[["C", "gamma"]].to_numpy()
            if reference is None:
                reference = candidates
            else:
                np.testing.assert_array_equal(candidates, reference)
            row = results[(results.seed == seed) & (results.version == version)].iloc[0]
            assert np.isclose(row.cv_accuracy, h.cv_accuracy.max())
            pred = pd.read_csv(OUTPUT / f"seed_{seed}_{version}_selected_cv_predictions.csv")
            assert len(pred) == 400 and pred.recording_id.nunique() == 400
            run_dir = OUTPUT.parent / f"seed_{seed}" / "uniform"
            split = json.loads((run_dir / "grouped_split_manifest.json").read_text())
            assert set(pred.recording_id) == set(split["train_segment_ids"])
            assert not set(pred.recording_id).intersection(split["test_segment_ids"])
            curves[version].append(h.cv_accuracy.cummax().to_numpy()*100)
        pair = results[results.seed == seed].set_index("version")
        axes[0].plot([0, 1], [pair.loc[v, "cv_accuracy"]*100 for v in colors],
                     marker="o", alpha=.7, label=f"Seed {seed}")
    axes[0].set_xticks([0, 1], list(labels.values()))
    axes[0].set_ylabel("Selected training-CV accuracy (%)")
    axes[0].legend(fontsize=8, frameon=False)
    for version, paths in curves.items():
        paths = np.array(paths)
        mean, sd = paths.mean(axis=0), paths.std(axis=0, ddof=1)
        x = np.arange(1, paths.shape[1]+1)
        axes[1].plot(x, mean, color=colors[version], label=labels[version])
        axes[1].fill_between(x, mean-sd, mean+sd, color=colors[version], alpha=.15)
    axes[1].set_xlabel("SVM configurations evaluated")
    axes[1].set_ylabel("Best training-CV accuracy so far (%)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT / "feature_comparison.png", dpi=200)
    fig.savefig(OUTPUT / "feature_comparison.pdf")
    print("Validated paired candidates, complete histories, scores and training-only prediction IDs.")


if __name__ == "__main__":
    main()

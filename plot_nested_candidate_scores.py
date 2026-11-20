"""Plot individual BO and RS candidate scores from one saved outer fold."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(fold=5):
    root = Path(__file__).resolve().parent
    run = root / "results/10_nested_cv_shared_initialization/split_42_inner_1729/search_42"
    frames = {method: pd.read_csv(run / f"outer_fold_{fold}" / f"{method}_history.csv")
              for method in ("bo", "rs")}
    params = ["f_low", "f_high", "window_size", "C", "gamma", "score"]
    np.testing.assert_allclose(frames["bo"].loc[:15, params],
                               frames["rs"].loc[:15, params], rtol=0, atol=1e-12)
    for frame in frames.values():
        assert frame.iteration.tolist() == list(range(1, 51))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7.15, 3.2), layout="constrained")
        initial = frames["bo"].iloc[:16]
        ax.scatter(initial.iteration, initial.score * 100, color="#697580",
                   marker="D", s=24, label="Shared initial candidates", zorder=4)
        for method, color, label, marker in [
            ("bo", "#096398", "BO", "s"),
            ("rs", "#AD4E24", "Random search", "o")]:
            guided = frames[method].iloc[16:]
            ax.scatter(guided.iteration, guided.score * 100,
                       edgecolors=color, facecolors=color if method == "bo" else "white",
                       marker=marker, s=25, linewidths=1, label=label, zorder=3)
        ax.axvline(16.5, color="#67707A", linewidth=1, linestyle=(0, (3, 3)))
        scores = np.concatenate([frame.score.to_numpy() * 100 for frame in frames.values()])
        ymin = min(20, np.floor(scores.min() / 5) * 5 - 2)
        ax.set(xlim=(0.5, 50.5), ylim=(ymin, 95),
               xticks=[1, 10, 16, 20, 30, 40, 50],
               yticks=np.arange(20, 100, 10),
               xlabel="Candidate evaluations", ylabel="Candidate inner-CV accuracy (%)",
               title=f"Individual candidate scores: outer fold {fold}")
        ax.grid(axis="y", color="#DFE6EC", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", ncols=3, frameon=False, fontsize=8)
        target = root / "figures/chapter5" / f"nested_fold_{fold}_candidate_scores.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, facecolor="white")
        plt.close(fig)
    for method, frame in frames.items():
        guided = frame.iloc[16:].score * 100
        print(f"{method}: evaluations 17-50 median={guided.median():.3f}%, n={len(guided)}")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=5)
    main(parser.parse_args().fold)

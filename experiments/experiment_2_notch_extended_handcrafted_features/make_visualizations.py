from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(RESULTS / "final_summary_table.csv")
    methods = summary["method"].tolist()
    bo = pd.read_csv(RESULTS / "bo_search_history.csv")
    random = pd.read_csv(RESULTS / "random_search_history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "objective", "Search History: Objective Value"),
        (axes[1], "best_objective_so_far", "Search History: Best-so-far Objective"),
    ]:
        ax.plot(bo["evaluation_number"], bo[metric], marker="o", linewidth=1.6, markersize=3.5, label="BO", color="#1f77b4")
        ax.plot(random["evaluation_number"], random[metric], marker="o", linewidth=1.6, markersize=3.5, label="Random", color="#ff7f0e")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Objective")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
    out = FIGURES / "search_history_objective_value.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    x = np.arange(len(methods))
    width = 0.25

    axes[0, 0].bar(x - width, summary["best_grouped_cv_objective"], width=width, label="Best CV objective", color="#1f77b4")
    axes[0, 0].bar(x, summary["segment_accuracy"], width=width, label="Segment accuracy", color="#ff7f0e")
    axes[0, 0].bar(x + width, summary["segment_macro_recall"], width=width, label="Segment macro recall", color="#2ca02c")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Segment-level Summary")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x - width / 2, summary["window_accuracy"], width=width, label="Window accuracy", color="#9467bd")
    axes[0, 1].bar(x + width / 2, summary["window_macro_recall"], width=width, label="Window macro recall", color="#8c564b")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_title("Window-level Summary")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x, summary["segment_recall_F"], color="#d62728", width=0.55)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_title("Segment Recall for F")
    axes[1, 0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(summary["segment_recall_F"]):
        axes[1, 0].text(i, float(v) + 0.02, f"{float(v):.2f}", ha="center", va="bottom", fontsize=9)

    axes[1, 1].bar(x, summary["segment_recall_S"], color="#17becf", width=0.55)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].set_title("Segment Recall for S")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(summary["segment_recall_S"]):
        axes[1, 1].text(i, float(v) + 0.02, f"{float(v):.2f}", ha="center", va="bottom", fontsize=9)

    out = FIGURES / "method_metrics_overview.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(out)
    print(FIGURES / "search_history_objective_value.png")


if __name__ == "__main__":
    main()

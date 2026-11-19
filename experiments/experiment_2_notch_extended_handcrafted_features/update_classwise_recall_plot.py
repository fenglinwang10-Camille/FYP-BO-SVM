from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main() -> None:
    df = pd.read_csv(RESULTS / "method_comparison_segment_level.csv")

    methods = ["Manual", "Random Search", "BO"]
    labels = ["Z", "O", "N", "F", "S"]
    columns = [
        "segment_recall_Z",
        "segment_recall_O",
        "segment_recall_N",
        "segment_recall_F",
        "segment_recall_S",
    ]
    names = {
        "Manual": "Manual SVM",
        "Random Search": "Random Search",
        "BO": "BO-SVM",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    values = {
        method: df.loc[df["method"] == method, columns].iloc[0].to_numpy(dtype=float) * 100.0
        for method in methods
    }

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "classwise_recall_comparison_segment_level.png"

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10.4, 4.8), facecolor="#0f1b3d")
    ax.set_facecolor("#0f1b3d")

    x = np.arange(len(labels))
    width = 0.24

    for idx, method in enumerate(methods):
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values[method],
            width=width,
            label=names[method],
            color=colors[idx],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values[method]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color="white")
    ax.set_xlabel("Bonn EEG class", fontsize=11, color="white")
    ax.set_ylabel("Recall (%)", fontsize=11, color="white")
    ax.set_ylim(0, 105)
    ax.set_title("Class-wise Recall on the Held-out Test Set", fontsize=13, pad=10, color="white")
    ax.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        labelcolor="white",
    )
    ax.grid(axis="y", alpha=0.18, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="x", colors="white")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

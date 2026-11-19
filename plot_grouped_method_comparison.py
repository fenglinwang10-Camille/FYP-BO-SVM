"""Plot leakage-controlled grouped method comparison results."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "09_fixed_fold_segment_accuracy" / "method_comparison_seed_42"
BO_RESULTS = ROOT / "results" / "09_fixed_fold_segment_accuracy" / "seed_42" / "uniform"


def plot_accuracy_comparison(summary):
    method_keys = ["Manual SVM", "Random Search SVM", "BO-SVM"]
    labels = ["Manual SVM", "Random Search", "BO-SVM"]
    rows = summary.set_index("method").loc[method_keys]
    window_accuracy = 100 * rows["window_test_accuracy"].to_numpy(float)
    segment_accuracy = 100 * rows["segment_test_accuracy"].to_numpy(float)

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars1 = ax.bar(x - width / 2, window_accuracy, width, label="Window level")
    bars2 = ax.bar(x + width / 2, segment_accuracy, width, label="Segment level")
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "grouped_method_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def plot_convergence():
    bo = pd.read_csv(BO_RESULTS / "grouped_bo_history.csv").head(50)
    random_search = pd.read_csv(RESULTS / "grouped_random_search_history.csv").head(50)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(
        bo["iteration"],
        bo["score"].cummax(),
        linewidth=2,
        label="Bayesian Optimisation",
    )
    ax.plot(
        random_search["iteration"],
        random_search["score"].cummax(),
        linewidth=2,
        label="Random Search",
    )
    ax.set_xlabel("Candidate evaluation")
    ax.set_ylabel("Best grouped-CV objective so far")
    ax.set_xlim(1, 50)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "grouped_bo_vs_random_convergence.png", dpi=300)
    plt.close(fig)


def plot_bo_confusion_matrix():
    result = json.loads(
        (BO_RESULTS / "grouped_bo_final_result.json").read_text(encoding="utf-8")
    )
    matrix = np.asarray(
        result["window_level_test_metrics"]["confusion_matrix"],
        dtype=int,
    )
    class_names = ["Z", "O", "N", "F", "S"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(5), class_names)
    ax.set_yticks(range(5), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Grouped BO-SVM Held-Out Test")
    threshold = matrix.max() / 2
    for row in range(5):
        for column in range(5):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(RESULTS / "grouped_bo_test_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_classwise_recall_comparison():
    result_files = {
        "Manual SVM": RESULTS / "grouped_manual_final_result.json",
        "Random Search": RESULTS / "grouped_random_search_final_result.json",
        "BO-SVM": BO_RESULTS / "grouped_bo_final_result.json",
    }
    classes = ["Z", "O", "N", "F", "S"]
    recall_values = {}
    for method, path in result_files.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        recalls = result["window_level_test_metrics"]["per_class_recall"]
        recall_values[method] = 100 * np.asarray(
            [recalls[class_name] for class_name in classes],
            dtype=float,
        )

    x = np.arange(len(classes))
    width = 0.25
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for index, (method, values) in enumerate(recall_values.items()):
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=method,
            color=colors[index],
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_xlabel("EEG class")
    ax.set_ylabel("Held-out test recall (%)")
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULTS / "grouped_classwise_recall_comparison.png", dpi=300)
    plt.close(fig)


def main():
    summary = pd.read_csv(RESULTS / "grouped_method_comparison.csv")
    plot_accuracy_comparison(summary)
    plot_convergence()
    plot_bo_confusion_matrix()
    plot_classwise_recall_comparison()
    print("Saved grouped comparison plots to", RESULTS)


if __name__ == "__main__":
    main()

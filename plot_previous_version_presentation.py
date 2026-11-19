"""Generate presentation figures for the earlier five-class experiment."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN_DIR = ROOT / "results" / "00_main_bo_svm"
ARCHIVE = (
    ROOT
    / "results"
    / "90_bo_archives"
    / "bo_nf_focused_history_running_20260611_203307.csv"
)
SUBSET_SOURCE = ROOT / "results" / "01_subset_tasks" / "subset_svm_results_no_overlap.csv"
OUTPUT_DIR = ROOT / "results" / "03_previous_version"
CLASS_NAMES = ["Z", "O", "N", "F", "S"]


def nf_score(dataframe):
    return (
        0.35 * dataframe["validation_accuracy"].astype(float)
        + 0.25 * dataframe["validation_macro_recall"].astype(float)
        + 0.20 * dataframe["validation_recall_N"].astype(float)
        + 0.20 * dataframe["validation_recall_F"].astype(float)
    )


def load_main_rows():
    comparison = pd.read_csv(MAIN_DIR / "comparison_summary.csv")
    baseline = comparison[comparison["method"] == "SVM without BO"].iloc[0]
    random_row = pd.read_csv(MAIN_DIR / "random_search_best_params.csv").iloc[0]
    bo_history = pd.read_csv(ARCHIVE)
    bo_row = bo_history.loc[bo_history["score"].astype(float).idxmax()]
    return baseline, random_row, bo_history, bo_row


def accuracy_plot(baseline, random_row, bo_row):
    labels = ["Manual SVM", "Random Search", "Bayesian Optimization"]
    values = [
        float(baseline["test_accuracy"]),
        float(random_row["test_accuracy"]),
        float(bo_row["test_accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=["#6b7280", "#f59e0b", "#2563eb"])
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("Reported window-level accuracy")
    ax.set_title("Five-Class Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value * 100:.2f}%",
            ha="center",
            fontsize=10,
        )
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def convergence_plot(bo_history):
    random_history = pd.read_csv(MAIN_DIR / "random_search_history.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        bo_history["iteration"],
        bo_history["score"].astype(float).cummax(),
        label="Bayesian Optimization",
        color="#2563eb",
    )
    ax.plot(
        random_history["iteration"],
        nf_score(random_history).cummax(),
        label="Random Search",
        color="#f59e0b",
    )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best validation objective so far")
    ax.set_title("Validation Objective Search Progress")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_bo_vs_random.png", dpi=300)
    plt.close(fig)


def recall_plot(baseline, random_row, bo_row):
    rows = [
        ("Manual SVM", baseline),
        ("Random Search", random_row),
        ("Bayesian Optimization", bo_row),
    ]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, (label, row) in enumerate(rows):
        values = [float(row[f"test_recall_{name}"]) for name in CLASS_NAMES]
        ax.bar(x + (index - 1) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("Reported window-level recall")
    ax.set_title("Class-Wise Recall Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_wise_recall_comparison.png", dpi=300)
    plt.close(fig)


def confusion_plot(matrix, labels, title, filename):
    matrix = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def subset_plots():
    subset = pd.read_csv(SUBSET_SOURCE)
    task_order = ["Z_vs_S", "F_vs_S", "Z_vs_F_vs_S"]
    task_labels = ["Z vs S", "F vs S", "Z vs F vs S"]
    ordered = subset.set_index("task").loc[task_order]
    accuracies = ordered["test_accuracy"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(task_labels, accuracies, color="#2563eb")
    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel("Reported window-level accuracy")
    ax.set_title("Subset Classification Accuracy")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value * 100:.2f}%",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subset_accuracy.png", dpi=300)
    plt.close(fig)

    macro_recalls = [1.0, 0.99, (0.97 + 0.99 + 1.0) / 3]
    seizure_recalls = [1.0, 0.98, 1.0]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - 0.18, macro_recalls, width=0.36, label="Macro recall")
    ax.bar(x + 0.18, seizure_recalls, width=0.36, label="Seizure recall")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel("Recall")
    ax.set_title("Subset Recall Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subset_recall.png", dpi=300)
    plt.close(fig)

    for task, labels in [("F_vs_S", ["F", "S"]), ("Z_vs_F_vs_S", ["Z", "F", "S"])]:
        matrix = json.loads(ordered.loc[task, "confusion_matrix"])
        confusion_plot(
            matrix,
            labels,
            f"{task.replace('_vs_', ' vs ')} Confusion Matrix",
            f"{task}_confusion_matrix.png",
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline, random_row, bo_history, bo_row = load_main_rows()
    accuracy_plot(baseline, random_row, bo_row)
    convergence_plot(bo_history)
    recall_plot(baseline, random_row, bo_row)
    confusion_plot(
        json.loads(bo_row["test_confusion_matrix"]),
        CLASS_NAMES,
        "BO-SVM Reported Confusion Matrix",
        "bo_best_confusion_matrix.png",
    )
    subset_plots()


if __name__ == "__main__":
    main()

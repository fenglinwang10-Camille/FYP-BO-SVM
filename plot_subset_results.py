"""
plot_subset_results.py

Generate presentation-ready plots for Bonn EEG subset classification tasks:
    A/E  = Z/S
    D/E  = F/S
    A/D/E = Z/F/S
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SUBSET_DIR = PROJECT_DIR / "results" / "01_subset_tasks"

TASK_LABELS = {
    "Z_vs_S": "A vs E\n(Z vs S)",
    "F_vs_S": "D vs E\n(F vs S)",
    "Z_vs_F_vs_S": "A vs D vs E\n(Z vs F vs S)",
}

TASK_CLASSES = {
    "Z_vs_S": ["Z", "S"],
    "F_vs_S": ["F", "S"],
    "Z_vs_F_vs_S": ["Z", "F", "S"],
}


def _load_csv(name):
    """Load one subset result CSV."""
    path = SUBSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def plot_subset_accuracy_comparison():
    """Plot full-BO subset test accuracy for each task."""
    bo = _load_csv("subset_bo_results.csv")

    rows = []
    for _, row in bo.iterrows():
        rows.append(
            {
                "task": row["task"],
                "accuracy": float(row["test_accuracy"]),
            }
        )

    dataframe = pd.DataFrame(rows)
    tasks = [task for task in TASK_LABELS if task in dataframe["task"].unique()]
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(8, 5))
    values = [
        dataframe[dataframe["task"] == task]["accuracy"].iloc[0]
        for task in tasks
    ]
    bars = ax.bar(x, values, width=0.45, color="#2563eb")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in tasks])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.90, 1.02)
    ax.set_title("Bonn EEG Subset Classification Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output = SUBSET_DIR / "subset_bo_accuracy.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


def plot_bo_convergence():
    """Plot the best validation objective observed for each subset task."""
    history = _load_csv("subset_bo_history.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    for task, group in history.groupby("task"):
        group = group.sort_values("iteration")
        best_so_far = group["score"].cummax()
        ax.plot(
            group["iteration"],
            best_so_far,
            marker="o",
            label=TASK_LABELS.get(task, task).replace("\n", " "),
        )

    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best validation objective so far")
    ax.set_ylim(0.80, 1.02)
    ax.set_title("Subset BO Validation Progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = SUBSET_DIR / "subset_bo_convergence.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


def plot_bo_confusion_matrices():
    """Plot confusion matrices from the full-BO subset results."""
    bo = _load_csv("subset_bo_results.csv")

    for _, row in bo.iterrows():
        matrix = np.asarray(json.loads(row["test_confusion_matrix"]))
        task = row["task"]
        class_names = TASK_CLASSES.get(task, task.split("_vs_"))

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{TASK_LABELS.get(task, task)} Confusion Matrix")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center")

        fig.tight_layout()
        output = SUBSET_DIR / f"{task}_confusion_matrix.png"
        fig.savefig(output, dpi=300)
        plt.close(fig)
        print(f"Saved {output}")


def main():
    """Generate all subset result plots."""
    plot_subset_accuracy_comparison()
    plot_bo_convergence()
    plot_bo_confusion_matrices()


if __name__ == "__main__":
    main()

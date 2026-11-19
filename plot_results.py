"""
plot_results.py

Plot BO/Random Search validation-objective convergence and reported metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from display_utils import display_image
from objective import CLASS_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results" / "00_main_bo_svm"


def plot_convergence():
    """Plot best-so-far N/F-focused validation objective for both searches."""
    bo_path = RESULTS_DIR / "bo_history.csv"
    random_path = RESULTS_DIR / "random_search_history.csv"

    for path in [bo_path, random_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run BO and Random Search first.")

    bo_df = pd.read_csv(bo_path)
    random_df = pd.read_csv(random_path)

    def nf_focused_score(dataframe):
        return (
            0.35 * dataframe["validation_accuracy"]
            + 0.25 * dataframe["validation_macro_recall"]
            + 0.20 * dataframe["validation_recall_N"]
            + 0.20 * dataframe["validation_recall_F"]
        )

    bo_scores = bo_df["score"] if "score" in bo_df else nf_focused_score(bo_df)
    random_scores = nf_focused_score(random_df)
    bo_best = bo_scores.cummax()
    random_best = random_scores.cummax()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bo_df["iteration"], bo_best, marker="o", label="Bayesian Optimization")
    ax.plot(random_df["iteration"], random_best, marker="s", label="Random Search")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best validation objective so far")
    ax.set_title("Validation Objective: BO vs Random Search")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = RESULTS_DIR / "convergence_bo_vs_random.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    display_image(output_path)
    print(f"Saved {output_path}")


def plot_test_accuracy_bar():
    """Plot final test accuracy comparison across the three methods."""
    comparison_path = RESULTS_DIR / "comparison_summary.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Missing comparison_summary.csv. Run compare_results.py first.")

    comparison = pd.read_csv(comparison_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        comparison["method"],
        comparison["test_accuracy"],
        color=["#6b7280", "#2563eb", "#f59e0b"],
    )
    ax.set_ylabel("Reported window-level accuracy")
    ax.set_ylim(0, min(1.0, comparison["test_accuracy"].max() + 0.1))
    ax.set_title("Reported Window-Level Accuracy")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, comparison["test_accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=15)
    fig.tight_layout()

    output_path = RESULTS_DIR / "test_accuracy_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    display_image(output_path)
    print(f"Saved {output_path}")


def plot_class_wise_recall():
    """Plot class-wise recall comparison for Z, O, N, F, and S."""
    comparison_path = RESULTS_DIR / "comparison_summary.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Missing comparison_summary.csv. Run compare_results.py first.")

    comparison = pd.read_csv(comparison_path)
    x_positions = range(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (_, row) in enumerate(comparison.iterrows()):
        values = [row[f"test_recall_{class_name}"] for class_name in CLASS_NAMES]
        offsets = [x + (index - 1) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=row["method"])

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Reported window-level recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Class-wise Recall Rate Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = RESULTS_DIR / "class_wise_recall_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    display_image(output_path)
    print(f"Saved {output_path}")


def generate_plots():
    """Generate all requested plots."""
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_convergence()
    plot_test_accuracy_bar()
    plot_class_wise_recall()


if __name__ == "__main__":
    generate_plots()

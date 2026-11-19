"""Plot all five BO parameters against observed training-CV accuracy.

The plot uses completed BO history only; no EEG or held-out test data is read.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

from grouped_bo_optimizer import FIXED_FOLD_DIR, N_CALLS, SEARCH_SPACE
from parameter_priors import SUPPORTED_PRIORS


PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")
LOG_PARAMETERS = {"C", "gamma"}


def _normalise_history(history):
    columns = []
    for dimension in SEARCH_SPACE:
        values = history[dimension.name].to_numpy(dtype=float)
        low, high = map(float, dimension.bounds)
        if dimension.name in LOG_PARAMETERS:
            values = np.log10(values)
            low, high = np.log10([low, high])
        columns.append((values - low) / (high - low))
    coordinates = np.column_stack(columns)
    if not np.isfinite(coordinates).all() or np.any(coordinates < -1e-8) or np.any(coordinates > 1 + 1e-8):
        raise ValueError("BO history contains an out-of-bounds parameter")
    return np.clip(coordinates, 0.0, 1.0)


def _axis_ticks(dimension):
    low, high = map(float, dimension.bounds)
    if dimension.name in LOG_PARAMETERS:
        values = np.geomspace(low, high, 4 if dimension.name == "C" else 5)
        positions = (np.log10(values) - np.log10(low)) / (np.log10(high) - np.log10(low))
        labels = [f"{value:g}" for value in values]
    else:
        positions = np.linspace(0.0, 1.0, 5)
        values = low + positions * (high - low)
        labels = [f"{value:g}" for value in values]
    return positions, labels


def plot(seed=42, prior="uniform"):
    if prior not in SUPPORTED_PRIORS:
        raise ValueError(f"Unknown parameter prior: {prior}")
    run_dir = FIXED_FOLD_DIR / f"seed_{seed}" / prior
    result = json.loads((run_dir / "grouped_bo_final_result.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "grouped_bo_history.csv")
    if (result.get("objective_mode") != "segment_accuracy"
            or result.get("cv_split_protocol") != "recording_fixed_v2"
            or result.get("parameter_prior") != prior
            or int(result.get("random_state", -1)) != seed):
        raise ValueError("Saved run does not match the requested fixed-fold experiment")
    if (len(history) != N_CALLS
            or history["iteration"].tolist() != list(range(1, N_CALLS + 1))
            or not np.allclose(history["score"], history["validation_segment_accuracy"])):
        raise ValueError("Expected 50 ordered segment-accuracy BO evaluations")

    coordinates = _normalise_history(history)
    scores = 100.0 * history["score"].to_numpy(dtype=float)
    best_index = int(np.argmax(scores))
    if (int(history.iloc[best_index]["iteration"]) != int(result["best_iteration"])
            or not np.isclose(scores[best_index], 100 * result["best_validation_score"])):
        raise ValueError("BO history and saved best result disagree")

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(PARAMETERS))
    norm = colors.Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    cmap = plt.get_cmap("viridis")
    for index in np.argsort(scores):
        if index == best_index:
            continue
        ax.plot(x, coordinates[index], color=cmap(norm(scores[index])),
                linewidth=1.1, alpha=0.55, zorder=2)

    for index, dimension in enumerate(SEARCH_SPACE):
        ax.vlines(index, 0, 1, color="#4b5563", linewidth=1.0, zorder=3)
        positions, labels = _axis_ticks(dimension)
        for position, label in zip(positions, labels):
            ax.plot([index - 0.025, index + 0.025], [position, position],
                    color="#4b5563", linewidth=0.8, zorder=4)
            ax.text(index - 0.045, position, label, ha="right", va="center",
                    fontsize=8, color="#374151")

    ax.plot(x, coordinates[best_index], color="#d1492e", linewidth=3.0,
            marker="o", markersize=5.5, markerfacecolor="white",
            markeredgewidth=1.8, label=f"Best observed CV: {scores[best_index]:.2f}% "
            f"(evaluation {best_index + 1})", zorder=6)
    ax.set_xticks(x, ["f_low (Hz)", "f_high (Hz)", "window_size (s)",
                      "SVM C (log)", "SVM gamma (log)"], fontsize=10)
    ax.set_xlim(-0.40, 4.18)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            fraction=0.025, pad=0.03)
    colorbar.set_label("Observed grouped-CV segment accuracy (%)")
    fig.suptitle("BO search across all five parameters", fontsize=15, y=0.97)
    fig.text(0.08, 0.035,
             f"Seed {seed} | {prior} | 50 BO evaluations. Each line is one complete parameter setting. "
             "Axis positions are normalized; C and gamma use log scales.\n"
             "Color is training-CV accuracy, not held-out test accuracy. "
             "The red line is the best observed evaluation, not a global optimum.",
             fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.81, bottom=0.16)
    output_dir = FIXED_FOLD_DIR / "visualizations" / f"seed_{seed}_{prior}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "bo_five_parameter_parallel_coordinates.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior", choices=SUPPORTED_PRIORS, default="uniform")
    args = parser.parse_args()
    print(plot(args.seed, args.prior))

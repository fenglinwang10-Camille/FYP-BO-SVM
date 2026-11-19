"""Plot the five BO parameters, random initial evaluations versus BO-guided ones.

The plot uses completed BO history only; no EEG or held-out test data is read.
Top panel: the random initial evaluations. Bottom panel: the BO-guided
evaluations. Both panels share the same axes and colour scale, and the best
evaluation of each phase is drawn in red.
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
AXIS_LABELS = [r"$f_{\mathrm{low}}$ (Hz)", r"$f_{\mathrm{high}}$ (Hz)",
               r"Window length $T$ (s)", r"$C$ (log scale)",
               r"$\gamma$ (log scale)"]
TICK_VALUES = {"f_low": [0.5, 2.0, 3.5, 5.0], "f_high": [30, 40, 50, 60],
               "window_size": [1, 2, 3, 4, 5], "C": [0.1, 1, 10, 100],
               "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
BEST_COLOUR = "#d1492e"
AXIS_COLOUR = "#4b5563"


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
    values = [v for v in TICK_VALUES[dimension.name] if low - 1e-12 <= v <= high + 1e-12]
    if len(values) < 2:
        values = list(np.linspace(low, high, 4))
    values = np.asarray(values, dtype=float)
    if dimension.name in LOG_PARAMETERS:
        positions = (np.log10(values) - np.log10(low)) / (np.log10(high) - np.log10(low))
    else:
        positions = (values - low) / (high - low)
    if dimension.name == "gamma":
        labels = [rf"$10^{{{int(round(np.log10(v)))}}}$" for v in values]
    else:
        labels = [f"{v:g}" for v in values]
    return positions, labels


def _draw_panel(ax, coordinates, scores, indices, norm, cmap, title):
    x = np.arange(len(PARAMETERS))
    best_index = int(indices[np.argmax(scores[indices])])
    for index in indices[np.argsort(scores[indices])]:
        if index == best_index:
            continue
        ax.plot(x, coordinates[index], color=cmap(norm(scores[index])),
                linewidth=1.0, alpha=0.65, zorder=2)
    for position, dimension in enumerate(SEARCH_SPACE):
        ax.vlines(position, 0, 1, color=AXIS_COLOUR, linewidth=0.9, zorder=3)
        positions, labels = _axis_ticks(dimension)
        for tick, label in zip(positions, labels):
            ax.plot([position - 0.025, position + 0.025], [tick, tick],
                    color=AXIS_COLOUR, linewidth=0.7, zorder=4)
            ax.text(position - 0.05, tick, label, ha="right", va="center",
                    fontsize=8, color="#374151")
    ax.plot(x, coordinates[best_index], color=BEST_COLOUR, linewidth=2.4,
            marker="o", markersize=4.5, markerfacecolor="white",
            markeredgewidth=1.5, zorder=6,
            label=f"Best in this phase: {scores[best_index]:.2f}% "
                  f"(evaluation {best_index + 1})")
    ax.set_xticks(x, AXIS_LABELS, fontsize=9)
    ax.set_xlim(-0.40, 4.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9, pad=10)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False,
              fontsize=8, handlelength=1.5)


def plot(seed=44, prior="uniform", n_initial=16, vmin=75.0):
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
    overall_best = int(np.argmax(scores))
    if (int(history.iloc[overall_best]["iteration"]) != int(result["best_iteration"])
            or not np.isclose(scores[overall_best], 100 * result["best_validation_score"])):
        raise ValueError("BO history and saved best result disagree")

    initial = np.arange(0, n_initial)
    guided = np.arange(n_initial, len(history))
    vmax = float(np.ceil(scores.max()))
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#9ca3af")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.8), layout="constrained")
    _draw_panel(axes[0], coordinates, scores, initial, norm, cmap,
                f"(a) Random initial evaluations (1\u2013{n_initial})")
    _draw_panel(axes[1], coordinates, scores, guided, norm, cmap,
                f"(b) BO-guided evaluations ({n_initial + 1}\u2013{len(history)})")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                            fraction=0.04, pad=0.02, extend="min")
    colorbar.set_label("Mean validation accuracy (%)", fontsize=9)
    colorbar.set_ticks(np.arange(vmin, vmax + 1, 5))
    colorbar.ax.tick_params(labelsize=8)

    output_dir = FIXED_FOLD_DIR / "visualizations" / f"seed_{seed}_{prior}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "bo_parallel_initial_vs_guided"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--prior", choices=SUPPORTED_PRIORS, default="uniform")
    parser.add_argument("--n_initial", type=int, default=16)
    parser.add_argument("--vmin", type=float, default=75.0,
                        help="scores below this value are drawn in grey")
    args = parser.parse_args()
    print(plot(args.seed, args.prior, args.n_initial, args.vmin))

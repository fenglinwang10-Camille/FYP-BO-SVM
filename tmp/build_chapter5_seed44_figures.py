"""Plot the chapter 5 comparisons from the saved segment-level results."""

import json
from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from analyze_search_behaviour import (
    MAPPINGS, N_INITIAL, PREVIEW, RESULTS, SEEDS, read_history,
)


SEED = 44
COMPARISON = RESULTS / f"method_comparison_seed_{SEED}"
LABELS = ("Z", "O", "N", "F", "S")
EVALUATIONS = np.arange(1, 51)
BO_COLOR = "#096398"
RS_COLOR = "#AD4E24"
NEUTRAL = "#5F6874"
GRID = "#DCE4EA"
NAMES = {"RS": "Random search", "BO": "Bayesian optimisation"}
METHOD_COLORS = {"RS": RS_COLOR, "BO": BO_COLOR}
MAP_LABELS = ("Uniform", "Truncated normal", "Truncated exponential", "Truncated Gamma")
MAP_COLORS = (BO_COLOR, "#B14D28", "#14806C", "#835691")
SUFFIX = "_review.png"


def save(fig, name):
    fig.savefig(PREVIEW / (name + SUFFIX), dpi=320, facecolor="white")
    plt.close(fig)


def grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=.65)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=10.8, pad=8)


def search_axis(ax):
    ax.axvline(N_INITIAL, color=NEUTRAL, lw=1, linestyle=(0, (3, 3)), zorder=1.5)
    ax.set(xlim=(.5, 50.5), xticks=[1, 10, 16, 20, 30, 40, 50],
           xlabel="Candidate evaluations")
    grid(ax)


def candidate_axis(fig, outer):
    ax = fig.add_subplot(outer)
    ax.set(ylim=(15, 95), yticks=range(20, 91, 10),
           ylabel="Candidate CV accuracy (%)")
    grid(ax)
    return ax


def candidate_boxes(ax, groups):
    centers = np.arange(1, 6)
    rng = np.random.default_rng(44)
    for (runs, color, fill), shift in zip(groups, (-.19, .19), strict=True):
        jitter = rng.uniform(-.10, .10, runs.shape)
        ax.boxplot(runs.T, positions=centers + shift, patch_artist=True,
                   manage_ticks=False, widths=.30, showfliers=False,
                   boxprops={"facecolor": fill, "edgecolor": color},
                   medianprops={"color": color, "linewidth": 1.4},
                   whiskerprops={"color": color}, capprops={"color": color})
        for x, run, noise in zip(centers + shift, runs, jitter, strict=True):
            ax.scatter(x + noise, run, s=9, alpha=.45, color=color,
                       linewidths=0, zorder=2.7)
    ax.set(xlim=(.49, 5.51), xticks=centers, xticklabels=list(SEEDS),
           xlabel="Seed")


def figure_51(uniform):
    fig = plt.figure(figsize=(7.25, 7.05), layout="constrained")
    layout = fig.add_gridspec(2, 1, height_ratios=[1, 1.10])
    history = fig.add_subplot(layout[0])
    candidate_layout = layout[1].subgridspec(2, 1, height_ratios=[4.6, 1], hspace=.035)
    candidates = fig.add_subplot(candidate_layout[0])
    low_scores = fig.add_subplot(candidate_layout[1], sharex=candidates)
    candidates.set(ylim=(40, 95), yticks=range(40, 91, 10),
                   ylabel="Candidate CV accuracy (%)")
    low_scores.set(ylim=(15, 40), yticks=[20, 30])
    for axis, edge in ((candidates, 0), (low_scores, 1)):
        grid(axis)
        axis.plot([0], [edge], transform=axis.transAxes,
                  marker=[(-1, -.6), (1, .6)], markersize=7,
                  color="#333333", linestyle="none", clip_on=False)
    candidates.tick_params(axis="x", bottom=False, labelbottom=False)
    candidates.spines["bottom"].set_visible(False)
    best = np.maximum.accumulate(uniform, axis=1)
    mean, sd = best.mean(axis=0), best.std(axis=0, ddof=1)
    for index, run in enumerate(best):
        history.step(EVALUATIONS, run, where="post", lw=.85, alpha=.30,
                     color=BO_COLOR, label="Individual seed" if index == 0 else None)
    history.fill_between(EVALUATIONS, mean - sd, mean + sd, step="post", alpha=.13,
                         color=BO_COLOR, linewidth=0, label="Mean ± sample SD")
    history.step(EVALUATIONS, mean, where="post", color=BO_COLOR, lw=2, label="Mean")
    search_axis(history)
    history.set_ylabel("Best CV accuracy (%)")
    title(history, "(a) Uniform BO search histories")
    history.legend(loc="lower right", ncols=2, frameon=False, fontsize=8.2)

    phases = ((uniform[:, :N_INITIAL], RS_COLOR, "#F4E2D8"),
              (uniform[:, N_INITIAL:], BO_COLOR, "#DDECF6"))
    candidate_boxes(candidates, phases)
    candidate_boxes(low_scores, phases)
    candidates.set_xlabel("")
    title(candidates, "(b) BO candidate scores by phase")
    candidates.legend(
        handles=[Patch(facecolor=fill, edgecolor=color, label=label)
                 for (_, color, fill), label in zip(
                     phases, ("Evaluations 1–16", "Evaluations 17–50"), strict=True)],
        loc="lower right", bbox_to_anchor=(1, 1.005), ncols=2,
        frameon=False, fontsize=8.4,
    )
    save(fig, "fig5_1_uniform_bo_history_and_phases")


def figure_52(scores, baseline):
    fig = plt.figure(figsize=(7.25, 6.7), layout="constrained")
    layout = fig.add_gridspec(2, 1, height_ratios=[1, 1.12])
    history = fig.add_subplot(layout[0])
    candidates = candidate_axis(fig, layout[1])
    for method, values in scores.items():
        best = np.maximum.accumulate(values)
        history.step(EVALUATIONS, best, where="post", label=NAMES[method],
                     color=METHOD_COLORS[method], lw=1.9,
                     linestyle="--" if method == "RS" else "-")
    history.axhline(baseline, lw=1.1, color=NEUTRAL, linestyle="-.",
                    label=f"Fixed reference ({baseline:.2f}%)", zorder=1.8)
    search_axis(history)
    history.set(ylabel="Best CV accuracy (%)", ylim=(67.5, 92),
                yticks=[70, 75, 80, 85, 90])
    title(history, "(a) Best-so-far validation accuracy on seed 44")
    history.legend(loc="lower right", frameon=False, fontsize=8.4)

    search_axis(candidates)
    candidates.set(ylim=(30, 93), yticks=range(30, 91, 10))
    phases = ((1, N_INITIAL), (N_INITIAL + 1, len(EVALUATIONS)))
    for (method, values), shift, marker in zip(
        scores.items(), (-.14, .14), ("o", "s"), strict=True
    ):
        color = METHOD_COLORS[method]
        candidates.scatter(EVALUATIONS + shift, values, alpha=.47, s=16,
                           color=color, marker=marker, linewidths=0, zorder=2.5)
        for start, end in phases:
            median = np.median(values[start - 1:end])
            candidates.hlines(median, start, end, color=color, linewidth=1.65,
                              linestyle="--" if method == "RS" else "-", zorder=3)
            median_text = Decimal(str(median)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            center = (start + end) / 2
            candidates.text(center, 47.5 if method == "RS" else 43.5,
                            f"{method} {median_text}%", color=color, fontsize=8.8,
                            ha="center", va="center")
    for start, end in phases:
        candidates.text((start + end) / 2, 52.2, f"Median, evaluations {start}–{end}",
                        color=NEUTRAL, fontsize=8.3, ha="center", va="center")
    title(candidates, "(b) Candidate scores on seed 44")
    candidates.legend(
        handles=[Line2D([], [], color=METHOD_COLORS[method], marker=marker,
                        linestyle="--" if method == "RS" else "-", markersize=4,
                        label=NAMES[method])
                 for method, marker in zip(scores, ("o", "s"), strict=True)],
        loc="lower right", bbox_to_anchor=(1, 1.005), ncols=2,
        frameon=False, fontsize=8.4,
    )
    save(fig, "fig5_2_seed44_search_comparison")


def paired_history_figure(rs, bo):
    fig = plt.figure(figsize=(7.25, 7.05), layout="constrained")
    layout = fig.add_gridspec(2, 1, height_ratios=[1, 1.10])
    history = fig.add_subplot(layout[0])
    candidates = candidate_axis(fig, layout[1])
    for method, runs in (("RS", rs), ("BO", bo)):
        color = METHOD_COLORS[method]
        best = np.maximum.accumulate(runs, axis=1)
        q1, median, q3 = np.percentile(best, [25, 50, 75], axis=0)
        history.fill_between(EVALUATIONS, q1, q3, step="post", color=color,
                             alpha=.16, linewidth=0)
        history.step(EVALUATIONS, median, where="post", color=color, lw=1.8,
                     linestyle="--" if method == "RS" else "-", label=NAMES[method])
    search_axis(history)
    history.set_ylabel("Best CV accuracy (%)")
    title(history, "Best-so-far validation accuracy")
    history.text(16.65, .11, "End of BO random\ninitialisation",
                 transform=history.get_xaxis_transform(), fontsize=8.4,
                 color=NEUTRAL, va="bottom")
    history.legend(loc="lower right", frameon=False, fontsize=8.5)
    candidate_boxes(candidates, ((rs[:, N_INITIAL:], RS_COLOR, "#F4E2D8"),
                                (bo[:, N_INITIAL:], BO_COLOR, "#DDECF6")))
    title(candidates, "Candidate scores during evaluations 17–50")
    save(fig, "paired_search_median_iqr_and_rs_scores")


def figure_53(results):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), layout="constrained",
                             sharex=True, sharey=True)
    for index, (ax, (method, result)) in enumerate(zip(axes, results.items(), strict=True)):
        metrics = result["segment_level_test_metrics"]
        matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        assert matrix.shape == (5, 5) and np.all(matrix.sum(axis=1) == 20)
        assert np.isclose(matrix.trace() / matrix.sum(), metrics["accuracy"])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=20, interpolation="none")
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(column, row, str(value), ha="center", va="center",
                    color="white" if value >= 12 else "#15263A", fontsize=9)
        ax.set_xticks(np.arange(5), LABELS)
        ax.set_yticks(np.arange(5), LABELS)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="#DEE7F0", linewidth=.55)
        ax.tick_params(which="both", length=0, pad=4)
        ax.spines[:].set_visible(False)
        ax.set_title(method, fontsize=10, pad=9)
        if index == 0:
            ax.set_ylabel("True class")
        if index == 1:
            ax.set_xlabel("Predicted class")
    scale = fig.colorbar(image, ax=axes, shrink=.8, pad=.02, ticks=[0, 5, 10, 15, 20])
    scale.set_label("Test segments", labelpad=7)
    scale.outline.set_visible(False)
    scale.ax.tick_params(length=0)
    save(fig, "fig5_3_seed44_segment_confusion_matrices")


def figure_54(mappings):
    fig, (history, final) = plt.subplots(2, 1, figsize=(7.25, 5.9),
                                        layout="constrained", height_ratios=[1.28, 1])
    starts = []
    for mapping, label, color in zip(MAPPINGS, MAP_LABELS, MAP_COLORS, strict=True):
        best = np.maximum.accumulate(mappings[mapping])
        starts.append(best[0])
        history.step(EVALUATIONS, best, color=color, where="post", lw=1.8, label=label)
    search_axis(history)
    history.set(ylabel="Best CV accuracy (%)", ylim=(min(starts) - 1.5, 91.8))
    title(history, f"(a) Best-so-far validation accuracy on seed {SEED}")
    history.legend(loc="lower right", ncols=2, frameon=False, fontsize=8.2)

    for position, (mapping, color) in enumerate(zip(MAPPINGS, MAP_COLORS, strict=True)):
        best = mappings[mapping].max()
        final.scatter(position, best, s=46, color=color, edgecolor="white",
                      linewidth=.65, zorder=3)
        final.annotate(f"{best:.2f}%", (position, best), xytext=(0, 7),
                       textcoords="offset points", color=color, fontsize=9,
                       ha="center", va="bottom")
    final.set(ylabel="Final best CV accuracy (%)",
              ylim=(88.9, 91.2), yticks=np.arange(89, 91.1, .5),
              xlim=(-.56, 3.56), xticks=np.arange(4),
              xticklabels=MAP_LABELS)
    title(final, f"(b) Final best validation accuracy on seed {SEED}")
    grid(final)
    save(fig, "fig5_4_mapping_history_and_final_cv")


def main():
    mappings = {
        mapping: np.stack([read_history(seed, "BO", mapping)[1] / 4 for seed in SEEDS])
        for mapping in MAPPINGS
    }
    rs = np.stack([read_history(seed, "RS")[1] / 4 for seed in SEEDS])
    scores = {
        "RS": rs[SEED - min(SEEDS)],
        "BO": mappings["uniform"][SEED - min(SEEDS)],
    }
    paths = {
        "Fixed reference": COMPARISON / "grouped_manual_final_result.json",
        "Random search": COMPARISON / "grouped_random_search_final_result.json",
        "Bayesian optimisation": RESULTS / f"seed_{SEED}" / "uniform/grouped_bo_final_result.json",
    }
    results = {method: json.loads(path.read_text(encoding="utf-8"))
               for method, path in paths.items()}
    fixed = results["Fixed reference"]
    assert fixed["search_evaluations"] == 1 and fixed["test_evaluations"] == 1
    assert all(result["random_state"] == SEED for result in results.values())
    assert np.isclose(scores["BO"].max(), 90.50) and np.isclose(scores["RS"].max(), 88.00)
    assert np.isclose(mappings["uniform"].max(), scores["BO"].max())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "axes.labelcolor": "#263849",
                         "text.color": "#263849"}):
        figure_51(mappings["uniform"])
        figure_52(scores, fixed["validation_segment_accuracy"] * 100)
        figure_53(results)
        figure_54({name: runs[SEED - min(SEEDS)] for name, runs in mappings.items()})
        paired_history_figure(rs, mappings["uniform"])


if __name__ == "__main__":
    main()

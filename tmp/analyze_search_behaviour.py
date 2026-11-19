"""Summarise saved 55-feature search histories without training or test scoring."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "09_fixed_fold_segment_accuracy"
ARTIFACTS = RESULTS / "search_behaviour_analysis"
PREVIEW = ROOT / "tmp"
SEEDS = range(42, 47)
MAPPINGS = ("uniform", "normal", "exponential", "gamma")
PARAMETERS = ["f_low", "f_high", "window_size", "C", "gamma"]
LOWER = np.array([0.5, 30., 1., 0.1, 1e-4])
UPPER = np.array([5., 60., 5., 100., 1.])
N_INITIAL = 16
N_EVALUATIONS = 50
PHASES = (("1_16", slice(0, N_INITIAL)), ("17_50", slice(N_INITIAL, N_EVALUATIONS)))
THRESHOLDS = {"87_5": 350, "88_0": 352, "88_75": 355}
COLORS = {"RS": "#AD4E24", "BO": "#0B6798"}
NAMES = {"RS": "Random search", "BO": "Bayesian optimisation"}


def read_history(seed, method, mapping="uniform"):
    if method == "RS":
        directory = RESULTS / f"method_comparison_seed_{seed}"
        prefix = "grouped_random_search"
    else:
        directory = RESULTS / f"seed_{seed}" / mapping
        prefix = "grouped_bo"
    history = pd.read_csv(directory / f"{prefix}_history.csv", float_precision="round_trip")
    result = json.loads((directory / f"{prefix}_final_result.json").read_text(encoding="utf-8"))
    assert history.iteration.tolist() == list(range(1, N_EVALUATIONS + 1))
    assert (history.feature_columns == 55).all() and (history.cv_folds == 5).all()
    assert np.allclose(history.score, history.validation_segment_accuracy)
    counts = np.rint(history.score.to_numpy() * 400).astype(int)
    assert np.allclose(counts / 400, history.score)
    selected = result["best_iteration"] - 1
    params = result["best_parameters"] if method == "BO" else result["parameters"]
    assert np.allclose(history.iloc[selected][PARAMETERS].to_numpy(dtype=float),
                       [params[name] for name in PARAMETERS])
    assert counts[selected] == counts.max()
    if method == "BO":
        assert result["n_initial_points"] == N_INITIAL
        assert result["parameter_prior"] == mapping and result["random_state"] == seed
        assert result["cv_split_protocol"] == "recording_fixed_v2"
    return history, counts, selected


def normalised(values):
    scaled = (values - LOWER) / (UPPER - LOWER)
    scaled[:, 3:] = (
        (np.log10(values[:, 3:]) - np.log10(LOWER[3:]))
        / (np.log10(UPPER[3:]) - np.log10(LOWER[3:]))
    )
    assert (scaled >= -1e-10).all() and (scaled <= 1 + 1e-10).all()
    return scaled


def learning_data(histories):
    frames = []
    for (method, seed), (history, counts, selected) in histories.items():
        coordinates = normalised(history[PARAMETERS].to_numpy())
        distance = np.linalg.norm(coordinates - coordinates[selected], axis=1)
        at_boundary = (np.isclose(coordinates, 0, atol=1e-10, rtol=0)
                       | np.isclose(coordinates, 1, atol=1e-10, rtol=0))
        frames.append(pd.DataFrame({
            "method": method,
            "seed": seed,
            "evaluation": history.iteration,
            "phase": np.where(history.iteration <= N_INITIAL, "1_16", "17_50"),
            "candidate_cv_pct": counts / 4,
            "distance_to_selected": distance,
            "selected_evaluation": np.arange(N_EVALUATIONS) == selected,
            "candidate_has_boundary_value": at_boundary.any(axis=1),
            "selected_boundary_values": at_boundary[selected].sum(),
            **{parameter: history[parameter] for parameter in PARAMETERS},
        }))
    return pd.concat(frames, ignore_index=True)


def phase_statistics(data, groups):
    records = []
    for key, part in data.groupby(groups, sort=False):
        without_selected = part.loc[~part.selected_evaluation]
        interior = without_selected.loc[~without_selected.candidate_has_boundary_value]
        records.append({
            **dict(zip(groups, key, strict=True)),
            "candidates": len(part),
            "candidate_cv_median_pct": part.candidate_cv_pct.median(),
            "candidate_cv_mean_pct": part.candidate_cv_pct.mean(),
            "candidates_below_70": int((part.candidate_cv_pct < 70).sum()),
            "median_distance": part.distance_to_selected.median(),
            "median_distance_excluding_selected": without_selected.distance_to_selected.median(),
            "interior_candidates_excluding_selected": len(interior),
            "median_distance_interior_excluding_selected": interior.distance_to_selected.median(),
            "candidates_with_boundary_value": int(part.candidate_has_boundary_value.sum()),
        })
    return pd.DataFrame(records)


def mapping_statistics(histories):
    by_seed = []
    for mapping in MAPPINGS:
        for seed in SEEDS:
            _, counts, selected = (histories["BO", seed] if mapping == "uniform"
                                  else read_history(seed, "BO", mapping))
            best = np.maximum.accumulate(counts)
            threshold_hits = {}
            for name, threshold in THRESHOLDS.items():
                evaluation = np.flatnonzero(best >= threshold)
                threshold_hits[f"first_eval_at_{name}_pct"] = (
                    int(evaluation[0] + 1) if len(evaluation) else None
                )
            by_seed.append({
                "mapping": mapping,
                "seed": seed,
                "best_cv_at_16_pct": best[N_INITIAL - 1] / 4,
                "best_cv_at_50_pct": best[-1] / 4,
                "best_so_far_mean_pct": best.mean() / 4,
                "best_gain_eval_17_50_pp": (best[-1] - best[N_INITIAL - 1]) / 4,
                "first_eval_of_final_best": int(np.argmax(counts) + 1),
                "selected_evaluation_in_saved_result": selected + 1,
                "candidates_below_70_eval_1_16": int((counts[:N_INITIAL] < 280).sum()),
                "candidates_below_70_eval_17_50": int((counts[N_INITIAL:] < 280).sum()),
                **threshold_hits,
            })
    by_seed = pd.DataFrame(by_seed)
    for threshold in THRESHOLDS:
        key = f"first_eval_at_{threshold}_pct"
        by_seed[key] = by_seed[key].astype("Int64")

    summary = []
    for mapping, runs in by_seed.groupby("mapping", sort=False):
        row = {
            "mapping": mapping,
            "seeds": len(runs),
            "mean_best_cv_at_16_pct": runs.best_cv_at_16_pct.mean(),
            "mean_best_cv_at_50_pct": runs.best_cv_at_50_pct.mean(),
            "sample_sd_final_best_cv_pp": runs.best_cv_at_50_pct.std(),
            "mean_best_so_far_over_50_evals_pct": runs.best_so_far_mean_pct.mean(),
            "mean_best_gain_eval_17_50_pp": runs.best_gain_eval_17_50_pp.mean(),
            "last_improvement_eval_min": runs.first_eval_of_final_best.min(),
            "last_improvement_eval_max": runs.first_eval_of_final_best.max(),
            "last_improvement_eval_median": runs.first_eval_of_final_best.median(),
            "candidates_below_70_eval_17_50": runs.candidates_below_70_eval_17_50.sum(),
            "candidate_count_eval_17_50": len(runs) * (N_EVALUATIONS - N_INITIAL),
            "candidate_pct_below_70_eval_17_50": (
                100 * runs.candidates_below_70_eval_17_50.sum()
                / (len(runs) * (N_EVALUATIONS - N_INITIAL))
            ),
        }
        for threshold in THRESHOLDS:
            hits = runs[f"first_eval_at_{threshold}_pct"].dropna()
            row[f"runs_reaching_{threshold}_pct"] = len(hits)
            row[f"median_eval_at_{threshold}_pct_among_reached"] = hits.median()
        summary.append(row)
    return by_seed, pd.DataFrame(summary)


def save_figure(fig, filename):
    fig.savefig(PREVIEW / filename, dpi=320, facecolor="white")
    plt.close(fig)


def history_axis(ax, title, ylabel):
    ax.axvline(N_INITIAL, color="#687484", linewidth=1, linestyle=(0, (3.5, 3)), zorder=2)
    ax.set(xlabel="Candidate evaluation", ylabel=ylabel, xlim=(.5, 50.5),
           xticks=[1, 10, 16, 20, 30, 40, 50])
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    ax.grid(axis="y", color="#D7DFE5", linewidth=.65)


def score_figure(data):
    fig, ax = plt.subplots(figsize=(7.05, 4.3), layout="constrained")
    history_axis(ax, "Candidate validation scores", "Segment-level CV accuracy (%)")
    handles = []
    for method, offset, marker in (("RS", -.14, "o"), ("BO", .14, "s")):
        group = data.loc[data.method.eq(method)]
        ax.scatter(group.evaluation + offset, group.candidate_cv_pct,
                   s=14, marker=marker, color=COLORS[method], alpha=.35,
                   linewidths=0, zorder=3)
        daily = group.groupby("evaluation").candidate_cv_pct.median()
        for _, stage in PHASES:
            rolling = daily.iloc[stage].rolling(5, center=True, min_periods=5).median().dropna()
            ax.plot(rolling.index, rolling, color=COLORS[method], lw=2.0, zorder=4)
        handles.append(Line2D([], [], color=COLORS[method], lw=2, marker=marker,
                              markersize=4, label=NAMES[method]))
    ax.set(ylim=(15, 95), yticks=range(20, 91, 10))
    ax.legend(handles=handles, frameon=False, loc="lower right",
              bbox_to_anchor=(1, 1.008), ncols=2, fontsize=8.5)
    save_figure(fig, "candidate_scores_over_evaluations.png")


def distance_figure(data):
    fig, ax = plt.subplots(figsize=(7.05, 4.15), layout="constrained")
    history_axis(ax, "Distance to the configuration selected in each run",
                 "Euclidean distance in normalised search space")
    for method in ("RS", "BO"):
        group = data.loc[data.method.eq(method)]
        values = group.groupby("evaluation").distance_to_selected
        medians = values.median()
        quartiles = values.quantile([.25, .75]).unstack()
        ax.fill_between(medians.index, quartiles[.25], quartiles[.75],
                        color=COLORS[method], alpha=.12, linewidth=0)
        ax.plot(medians.index, medians, color=COLORS[method], lw=1.65,
                linestyle="--" if method == "RS" else "-", label=NAMES[method])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "distance_to_selected_configuration.png")


def parameter_figure(data):
    sample = data.loc[data.seed.eq(44)]
    phase_colors = {"1_16": "#A34B22", "17_50": "#0B6798"}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.55), sharex=True, sharey=True,
                             layout="constrained")
    fig.suptitle(r"Seed 44 in the $C$–$\gamma$ plane", x=.013, ha="left", fontsize=11)
    for method, ax in zip(("RS", "BO"), axes, strict=True):
        run = sample.loc[sample.method.eq(method)]
        for phase, part in run.groupby("phase", sort=False):
            ax.scatter(part.C, part.gamma, s=12 + part.candidate_cv_pct, color=phase_colors[phase],
                       alpha=.61, edgecolor="white", linewidth=.45, zorder=3)
        selected = run.loc[run.selected_evaluation].iloc[0]
        ax.scatter([selected.C], [selected.gamma], s=180, marker="*",
                   facecolors="none", edgecolors="#23374A", linewidths=1.25, zorder=4)
        ax.set(xlabel=r"$C$", xscale="log", yscale="log",
               xlim=(.075, 140), ylim=(7e-5, 1.35))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_title(NAMES[method], loc="left", fontsize=10)
        ax.grid(color="#D7DFE5", linewidth=.65)
    axes[0].set_ylabel(r"$\gamma$")
    legend = [Line2D([], [], linestyle="none", marker="o", markersize=6,
                     color=phase_colors[phase], label=f"Evaluations {phase.replace('_', '–')}")
              for phase in ("1_16", "17_50")]
    legend.extend([
        Line2D([], [], linestyle="none", marker="*", markersize=10, markerfacecolor="none",
               markeredgecolor="#23374A", label="Selected configuration"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
               markeredgecolor="#687484", label="Larger area = higher CV accuracy"),
    ])
    fig.legend(handles=legend, loc="outside lower center", ncols=2, frameon=False,
               fontsize=8.6, columnspacing=2.4)
    save_figure(fig, "seed44_c_gamma_search_phases.png")


def main():
    histories = {(method, seed): read_history(seed, method)
                 for method in ("RS", "BO") for seed in SEEDS}
    data = learning_data(histories)
    phase_by_seed = phase_statistics(data, ["method", "seed", "phase"])
    phase_pooled = phase_statistics(data, ["method", "phase"])
    mapping_by_seed, mapping_summary = mapping_statistics(histories)
    ARTIFACTS.mkdir(exist_ok=True)
    outputs = {
        "learning_by_evaluation.csv": data,
        "learning_by_seed_and_phase.csv": phase_by_seed,
        "learning_pooled_by_phase.csv": phase_pooled,
        "mapping_speed_stability_by_seed.csv": mapping_by_seed,
        "mapping_speed_stability_summary.csv": mapping_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(ARTIFACTS / name, index=False, float_format="%.8g")
    print(phase_pooled.to_string(index=False))
    print(mapping_summary.round(3).to_string(index=False))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "axes.labelcolor": "#263849",
                         "text.color": "#263849"}):
        score_figure(data)
        distance_figure(data)
        parameter_figure(data)


if __name__ == "__main__":
    main()

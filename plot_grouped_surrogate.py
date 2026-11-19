"""Refit and visualize the final GP surrogate from a completed grouped BO run.

Plots are conditional slices of the five-dimensional surrogate, not observed
accuracy curves or held-out test predictions. No EEG data is loaded.
"""

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from scipy.stats import norm
from skopt import Optimizer
from skopt.space import Real

from grouped_bo_optimizer import FIXED_FOLD_DIR, N_CALLS, N_INITIAL_POINTS, SEARCH_SPACE
from parameter_priors import (
    CDF_EPSILON,
    PREPROCESSING_BOUNDS,
    SUPPORTED_PRIORS,
    parameter_to_latent,
)


ROOT = Path(__file__).resolve().parent
PARAMETERS = ("f_low", "f_high", "window_size", "C", "gamma")


def _run_dir(seed, prior, legacy=False):
    if not legacy:
        return FIXED_FOLD_DIR / f"seed_{seed}" / prior
    if prior == "uniform":
        return ROOT / "results" / "07_grouped_segment_objective" / f"seed_{seed}" / "segment_accuracy"
    return (
        ROOT / "results" / "06_grouped_prior_comparison"
        / f"seed_{seed}" / "segment_accuracy" / prior
    )


def _space(prior):
    if prior == "uniform":
        return SEARCH_SPACE
    return [
        Real(CDF_EPSILON, 1.0 - CDF_EPSILON, name=f"u_{name}")
        for name in PREPROCESSING_BOUNDS
    ] + SEARCH_SPACE[3:]


def _to_optimizer_points(frame, prior):
    points = frame.loc[:, PARAMETERS].to_numpy(dtype=float)
    if prior != "uniform":
        for column, bounds in enumerate(PREPROCESSING_BOUNDS.values()):
            points[:, column] = [
                parameter_to_latent(value, prior, bounds)
                for value in points[:, column]
            ]
    return points.tolist()


def _refit(run_dir, prior, seed, legacy=False):
    result = json.loads((run_dir / "grouped_bo_final_result.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "grouped_bo_history.csv")
    if result.get("objective_mode") != "segment_accuracy" or result.get("parameter_prior") != prior:
        raise ValueError("The saved run does not match segment_accuracy and the requested prior")
    if int(result.get("random_state", -1)) != seed:
        raise ValueError("The saved run has a different random seed")
    if not legacy and result.get("cv_split_protocol") != "recording_fixed_v2":
        raise ValueError("The saved run does not use recording-fixed CV folds")
    if len(history) != N_CALLS or history["iteration"].tolist() != list(range(1, N_CALLS + 1)):
        raise ValueError("A complete, ordered 50-evaluation history is required")
    if not np.allclose(history["score"], history["validation_segment_accuracy"]):
        raise ValueError("History scores are not segment-level validation accuracy")

    optimizer = Optimizer(
        dimensions=_space(prior),
        base_estimator="GP",
        n_initial_points=N_INITIAL_POINTS,
        acq_func="EI",
        random_state=seed,
    )
    optimizer.tell(_to_optimizer_points(history, prior), (-history["score"]).tolist())
    best = {name: float(result["best_parameters"][name]) for name in PARAMETERS}
    return optimizer, history, best


def _predict(optimizer, best, prior, varying):
    count = len(next(iter(varying.values())))
    points = np.tile([best[name] for name in PARAMETERS], (count, 1))
    for name, values in varying.items():
        points[:, PARAMETERS.index(name)] = values
    frame = pd.DataFrame(points, columns=PARAMETERS)
    transformed = optimizer.space.transform(_to_optimizer_points(frame, prior))
    cost_mean, sd = optimizer.models[-1].predict(transformed, return_std=True)
    return -cost_mean, np.maximum(sd, 0.0)


def _expected_improvement(mean, sd, incumbent):
    # skopt's EI uses xi=0.01 on the 0-1 objective scale by default.
    improvement = mean - incumbent - 0.01
    z = np.divide(improvement, sd, out=np.zeros_like(improvement), where=sd > 0)
    return np.where(
        sd > 0,
        improvement * norm.cdf(z) + sd * norm.pdf(z),
        np.maximum(improvement, 0.0),
    )


def _caption(best, prior, seed):
    return (
        f"Seed {seed} | {prior} | final GP refit from 50 training-CV scores\n"
        "Other parameters fixed at the best observed setting. "
        "GP uncertainty is not a test-accuracy confidence interval."
    )


def plot_one_dimensional(optimizer, history, best, prior, seed, output):
    grid = np.geomspace(0.1, 100.0, 220)
    mean, sd = _predict(optimizer, best, prior, {"C": grid})
    incumbent = float(history["score"].max())
    ei = _expected_improvement(mean, sd, incumbent)

    fig, (ax, ax_ei) = plt.subplots(
        2, 1, figsize=(8.3, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    ax.fill_between(grid, 100 * (mean - 1.96 * sd), 100 * (mean + 1.96 * sd),
                    color="#7ca8a2", alpha=0.35, label="GP mean +/- 1.96 SD")
    ax.plot(grid, 100 * mean, color="#176c66", linewidth=2, label="GP predicted CV accuracy")
    ax.axhline(100 * incumbent, color="#9f4a39", linestyle="--", linewidth=1.3,
               label="Best observed CV accuracy")
    ax.axvline(best["C"], color="#404040", linestyle=":", linewidth=1.2)
    ax.set_ylabel("Predicted segment accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    ax_ei.plot(grid, 100 * ei, color="#ab6428", linewidth=2)
    ax_ei.axvline(best["C"], color="#404040", linestyle=":", linewidth=1.2,
                  label="Best observed C")
    ax_ei.set_xscale("log")
    ax_ei.set_xlabel("SVM C (log scale)")
    ax_ei.set_ylabel("EI, xi=0.01 (pp)")
    ax_ei.legend(frameon=False, fontsize=8)
    ax_ei.grid(alpha=0.2)
    fig.suptitle("Final BO surrogate: conditional C slice", fontsize=13)
    fig.text(0.12, 0.035, _caption(best, prior, seed), fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.22, hspace=0.08)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_two_dimensional(optimizer, history, best, prior, seed, output,
                         x_name, y_name, x_grid, y_grid, x_label, y_label,
                         log_axes=False, contours=False):
    x, y = np.meshgrid(x_grid, y_grid)
    mean, sd = _predict(optimizer, best, prior, {
        x_name: x.ravel(), y_name: y.ravel(),
    })
    incumbent = float(history["score"].max())
    arrays = [
        (100 * mean.reshape(x.shape), "Predicted CV accuracy (%)", "viridis"),
        (100 * sd.reshape(x.shape), "GP SD (percentage points)", "magma"),
        (100 * _expected_improvement(mean, sd, incumbent).reshape(x.shape),
         "Expected improvement (pp; xi=0.01)", "cividis"),
    ]
    show_search = contours and x_name == "C" and y_name == "gamma"
    peak_index = int(np.argmax(mean)) if show_search else None
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7), sharex=True, sharey=True)
    for panel_index, (ax, (values, title, cmap)) in enumerate(zip(axes, arrays)):
        if contours:
            image = ax.contourf(x, y, values, levels=15, cmap=cmap)
            levels = np.linspace(float(values.min()), float(values.max()), 7)[1:-1]
            ax.contour(x, y, values, levels=levels, colors="white",
                       linewidths=0.65, alpha=0.8)
        else:
            image = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap)
        if show_search and panel_index == 0:
            ax.scatter(history[x_name], history[y_name], marker="o", s=11,
                       facecolors="none", edgecolors="#333333", alpha=0.55,
                       linewidths=0.75, label="50 evaluations (2D projection)")
        if show_search:
            ax.scatter(best[x_name], best[y_name], marker="*", s=155,
                       color="white", edgecolors="#222222", linewidths=0.9,
                       zorder=5,
                       label=(f"Best CV: C={best['C']:.2f}, gamma={best['gamma']:.4f}"
                              if panel_index == 0 else None))
            if panel_index == 0:
                ax.scatter(x.ravel()[peak_index], y.ravel()[peak_index], marker="D",
                           s=78, color="#d1492e", edgecolors="white",
                           linewidths=1.1, zorder=5,
                           label=(f"GP slice peak: C={x.ravel()[peak_index]:.2f}, "
                                  f"gamma={y.ravel()[peak_index]:.3f}"))
        else:
            ax.scatter(best[x_name], best[y_name], marker="x", s=75,
                       color="white", linewidths=2, label="Best observed setting")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel(y_label)
    axes[0].legend(frameon=True, fontsize=7,
                   loc="lower left" if show_search else "lower right")
    kind = "contours" if contours else "slice"
    fig.suptitle(f"Final BO surrogate: conditional {x_name}-{y_name} {kind}", fontsize=13)
    caption = _caption(best, prior, seed)
    if show_search:
        caption += "\nSmall circles project five-dimensional evaluations; the GP peak is conditional, not the final selected setting."
    fig.text(0.07, 0.035, caption, fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.85, bottom=0.22, wspace=0.24)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_filter_window_surfaces(optimizer, best, prior, seed, output):
    lows, highs = np.meshgrid(np.linspace(0.5, 5.0, 45),
                              np.linspace(30.0, 60.0, 45))
    windows = [1.0, best["window_size"], 5.0]
    surfaces = []
    for window in windows:
        mean, _ = _predict(optimizer, best, prior, {
            "f_low": lows.ravel(),
            "f_high": highs.ravel(),
            "window_size": np.full(lows.size, window),
        })
        surfaces.append(100 * mean.reshape(lows.shape))

    minimum = min(float(surface.min()) for surface in surfaces)
    maximum = max(float(surface.max()) for surface in surfaces)
    fig = plt.figure(figsize=(15.0, 5.7))
    for index, (window, surface) in enumerate(zip(windows, surfaces), start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.plot_surface(lows, highs, surface, cmap="viridis", vmin=minimum,
                        vmax=maximum, linewidth=0, antialiased=True, alpha=0.95)
        ax.set_title(f"window_size = {window:.2f} s", fontsize=10)
        ax.set_xlabel("f_low (Hz)", labelpad=5)
        ax.set_ylabel("f_high (Hz)", labelpad=5)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(30.0, 60.0)
        ax.set_zlim(minimum, maximum)
        ax.view_init(elev=25, azim=-55)
        ax.set_box_aspect((1, 1, 0.7))
    colorbar_axis = fig.add_axes([0.93, 0.29, 0.014, 0.42])
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(minimum, maximum),
                                  cmap="viridis"), cax=colorbar_axis,
                 label="Predicted CV accuracy (%)")
    fig.suptitle("Final GP surrogate: filter cutoffs at three window sizes", fontsize=13)
    fig.text(0.06, 0.035,
             f"Seed {seed} | {prior} | C and SVM gamma fixed at the best observed configuration.\n"
             "Height and colour show conditional GP prediction, not test accuracy. See the companion 3D scatter for evaluated points.",
             fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.86, bottom=0.20, wspace=0.08)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_filter_window_evaluations(history, best, prior, seed, output):
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(history["f_low"], history["f_high"], history["window_size"],
                        c=100 * history["score"], cmap="viridis", s=38,
                        edgecolor="#333333", linewidth=0.4, depthshade=False)
    ax.scatter(best["f_low"], best["f_high"], best["window_size"],
               marker="*", s=220, color="#d1492e", edgecolor="white",
               depthshade=False, label="Best observed CV")
    ax.set_xlabel("f_low (Hz)", labelpad=8)
    ax.set_ylabel("f_high (Hz)", labelpad=8)
    ax.set_zlabel("window_size (s)", labelpad=8)
    ax.set_xlim(0.5, 5.0)
    ax.set_ylim(30.0, 60.0)
    ax.set_zlim(1.0, 5.0)
    ax.view_init(elev=23, azim=-54)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(points, ax=ax, shrink=0.65, pad=0.10,
                 label="Observed grouped-CV accuracy (%)")
    fig.suptitle("BO evaluations in preprocessing-parameter space", fontsize=13)
    fig.text(0.06, 0.035,
             f"Seed {seed} | {prior} | Points are 50 five-dimensional BO evaluations projected onto three preprocessing axes.\n"
             "C and SVM gamma differ between points; colour is observed training-CV score, not test accuracy.",
             fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.02, right=0.93, top=0.90, bottom=0.14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior", choices=SUPPORTED_PRIORS, default="uniform")
    parser.add_argument("--legacy", action="store_true",
                        help="Plot archived pre-fixed-fold results separately")
    args = parser.parse_args()

    run_dir = _run_dir(args.seed, args.prior, args.legacy)
    optimizer, history, best = _refit(run_dir, args.prior, args.seed, args.legacy)
    output_dir = (
        ROOT / "results" / "08_surrogate_visualization" / f"seed_{args.seed}_{args.prior}"
        if args.legacy else FIXED_FOLD_DIR / "visualizations" / f"seed_{args.seed}_{args.prior}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    one_dimensional = output_dir / "surrogate_C_slice.png"
    filter_slice = output_dir / "surrogate_filter_slice.png"
    svm_slice = output_dir / "surrogate_C_gamma_slice.png"
    svm_contours = output_dir / "surrogate_C_gamma_contours.png"
    filter_window_surfaces = output_dir / "surrogate_filter_window_3d_surfaces.png"
    filter_window_evaluations = output_dir / "bo_filter_window_3d_evaluations.png"
    plot_one_dimensional(optimizer, history, best, args.prior, args.seed, one_dimensional)
    plot_two_dimensional(
        optimizer, history, best, args.prior, args.seed, filter_slice,
        "f_low", "f_high", np.linspace(0.5, 5.0, 55),
        np.linspace(30.0, 60.0, 55), "Low cutoff (Hz)", "High cutoff (Hz)",
    )
    plot_two_dimensional(
        optimizer, history, best, args.prior, args.seed, svm_slice,
        "C", "gamma", np.geomspace(0.1, 100.0, 55),
        np.geomspace(1e-4, 1.0, 55), "SVM C", "SVM kernel gamma",
        log_axes=True,
    )
    plot_two_dimensional(
        optimizer, history, best, args.prior, args.seed, svm_contours,
        "C", "gamma", np.geomspace(0.1, 100.0, 75),
        np.geomspace(1e-4, 1.0, 75), "SVM C", "SVM kernel gamma",
        log_axes=True, contours=True,
    )
    plot_filter_window_surfaces(
        optimizer, best, args.prior, args.seed, filter_window_surfaces,
    )
    plot_filter_window_evaluations(
        history, best, args.prior, args.seed, filter_window_evaluations,
    )
    print(one_dimensional)
    print(filter_slice)
    print(svm_slice)
    print(svm_contours)
    print(filter_window_surfaces)
    print(filter_window_evaluations)


if __name__ == "__main__":
    main()

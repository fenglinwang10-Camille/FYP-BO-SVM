"""Summarize completed grouped BO parameter-prior comparisons across seeds."""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import t

from compare_grouped_priors import compare
from grouped_bo_optimizer import FIXED_FOLD_DIR, PROJECT_DIR
from parameter_priors import SUPPORTED_PRIORS


METRICS = (
    "segment_accuracy",
    "segment_macro_f1",
    "segment_recall_N",
    "segment_recall_F",
    "segment_recall_S",
)


def _summary(values, bounds):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"n": n, "mean": mean, "sd": np.nan,
                "ci95_lower": np.nan, "ci95_upper": np.nan,
                "ci_method": "descriptive_t_clipped"}
    sd = float(np.std(values, ddof=1))
    margin = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_lower": max(bounds[0], mean - margin),
            "ci95_upper": min(bounds[1], mean + margin),
            "ci_method": "descriptive_t_clipped"}


def summarize(seeds, objective_mode="segment_accuracy"):
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    tables = []
    for seed in seeds:
        table = compare(seed=seed, mode=objective_mode)
        if set(table["prior"]) != set(SUPPORTED_PRIORS) or not table["status"].eq("complete").all():
            raise ValueError(f"Seed {seed} does not have four complete runs")
        tables.append(table.assign(seed=seed))
    per_seed = pd.concat(tables, ignore_index=True)
    output = (
        FIXED_FOLD_DIR / "repeated" if objective_mode == "segment_accuracy"
        else PROJECT_DIR / "results" / "06_grouped_prior_comparison" / f"{objective_mode}_repeated"
    )
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)

    summary_rows = []
    for prior, group in per_seed.groupby("prior"):
        for metric in METRICS:
            summary_rows.append({"prior": prior, "metric": metric,
                                 **_summary(group[metric], (0.0, 1.0))})
    pd.DataFrame(summary_rows).to_csv(output / "summary_by_prior.csv", index=False)

    paired_rows = []
    for seed, group in per_seed.groupby("seed"):
        by_prior = group.set_index("prior")
        for prior in SUPPORTED_PRIORS:
            if prior == "uniform":
                continue
            paired_rows.append({"seed": seed, "prior": prior,
                                **{metric: float(by_prior.loc[prior, metric]
                                                 - by_prior.loc["uniform", metric])
                                   for metric in METRICS}})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_differences.csv", index=False)
    paired_summary = []
    for prior, group in paired.groupby("prior"):
        for metric in METRICS:
            paired_summary.append({"prior": prior,
                                   "difference": f"{prior} - uniform",
                                   "metric": metric, **_summary(group[metric], (-1.0, 1.0))})
    pd.DataFrame(paired_summary).to_csv(output / "paired_summary.csv", index=False)
    print(f"Saved {len(seeds)} paired outer splits in {output}")
    print("95% t intervals are descriptive: repeated outer splits overlap and are not independent external cohorts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--objective-mode", default="segment_accuracy",
                        choices=("nf_focused", "macro_f1", "segment_accuracy"))
    arguments = parser.parse_args()
    summarize(arguments.seeds, arguments.objective_mode)

"""Summarize paired weighted-vs-segment-accuracy BO results across splits."""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import t

from compare_grouped_segment_objective import compare
from grouped_bo_optimizer import SEGMENT_OBJECTIVE_DIR


METRICS = (
    "test_segment_accuracy",
    "test_segment_macro_f1",
    "test_recall_N",
    "test_recall_F",
    "test_recall_S",
)


def _summary(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return {"n": n, "mean": mean, "sd": np.nan,
                "ci95_lower": np.nan, "ci95_upper": np.nan}
    sd = float(np.std(values, ddof=1))
    margin = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd,
            "ci95_lower": mean - margin, "ci95_upper": mean + margin}


def summarize(seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    tables = []
    for seed in seeds:
        table = compare(seed)
        if table is None:
            raise ValueError(f"Seed {seed} is incomplete")
        tables.append(table)
    per_seed = pd.concat(tables, ignore_index=True)
    output = SEGMENT_OBJECTIVE_DIR
    per_seed.to_csv(output / "paired_per_seed.csv", index=False)

    summary_rows = []
    difference_rows = []
    paired = []
    for seed, group in per_seed.groupby("seed"):
        by_mode = group.set_index("objective")
        if set(by_mode.index) != {"nf_focused", "segment_accuracy"}:
            raise ValueError(f"Missing or duplicate objective for seed {seed}")
        paired.append({"seed": seed, **{
            metric: float(by_mode.loc["segment_accuracy", metric]
                          - by_mode.loc["nf_focused", metric])
            for metric in METRICS
        }})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(output / "paired_differences.csv", index=False)

    for objective, group in per_seed.groupby("objective"):
        for metric in METRICS:
            summary_rows.append({"objective": objective, "metric": metric,
                                 **_summary(group[metric])})
    for metric in METRICS:
        difference_rows.append({"metric": metric,
                                "difference": "segment_accuracy - nf_focused",
                                **_summary(paired_df[metric])})
    pd.DataFrame(summary_rows).to_csv(output / "summary_by_objective.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(output / "paired_summary.csv", index=False)
    print(f"Saved summaries for {len(paired_df)} paired outer splits in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    arguments = parser.parse_args()
    summarize(arguments.seeds)

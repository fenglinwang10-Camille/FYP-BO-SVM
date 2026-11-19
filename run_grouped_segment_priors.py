"""Run missing parameter-prior experiments with segment accuracy as the BO objective."""

import argparse
import json

from compare_grouped_priors import _directory, compare
from grouped_bo_optimizer import N_CALLS, run_grouped_bo
from parameter_priors import SUPPORTED_PRIORS


def run(seed):
    feature_cache = {}
    for prior in SUPPORTED_PRIORS:
        directory = _directory(seed, "segment_accuracy", prior)
        final_path = directory / "grouped_bo_final_result.json"
        if final_path.exists():
            result = json.loads(final_path.read_text(encoding="utf-8"))
            if (result.get("objective_mode") != "segment_accuracy"
                    or result.get("parameter_prior") != prior
                    or result.get("random_state") != seed
                    or result.get("bo_evaluations_used") != N_CALLS
                    or result.get("cv_split_protocol") != "recording_fixed_v2"):
                raise ValueError(f"Completed result has unexpected settings: {final_path}")
            print(f"Using completed seed {seed}, prior {prior}", flush=True)
            continue
        print(f"Starting seed {seed}, prior {prior}", flush=True)
        run_grouped_bo(
            objective_mode="segment_accuracy",
            parameter_prior=prior,
            random_state=seed,
            feature_cache=feature_cache,
        )
    compare(seed=seed, mode="segment_accuracy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    run(arguments.seed)

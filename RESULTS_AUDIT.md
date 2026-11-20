# Post-hoc results audit

Run `python -B audit_results_feedback.py` to check the 500 paired outer-fold
predictions, compute nominal exact two-sided McNemar statistics, and list folds
where BO and RS selected exactly the same configuration. It does not rerun any
parameter search or modify the frozen nested-experiment source or predictions.
Outputs are under the completed nested run's `posthoc_audit/` directory.

BO alone correctly classified 21 recordings and RS alone correctly classified
7; both were correct on 426 and both incorrect on 46. The pooled nominal exact
two-sided p-value is 0.0125409514. Independence of recording pairs is an
assumption, not an established property of these cross-validated EEG data.
Outer training sets overlap and subject IDs are unavailable. Report the test
as a post-hoc, assumption-dependent analysis, not conclusive evidence of
general superiority. Per-fold p-values are diagnostic and not additional
confirmatory hypotheses. Fold differences and the 2.8 percentage-point pooled
accuracy difference remain central to interpretation.

Folds 2 and 4 selected the same BO/RS configuration, respectively from shared
initial candidates 13 and 3. Their complete predictions are identical. These
folds must remain in all averages and comparisons.

Run `python -B audit_results_feedback.py --reproduce-seed44` to additionally
recompute the historical fixed reference with split seed 44 and CV seed 44.
The original `run_comparison` entry point cannot produce that result as written:
its manual branch is seed-42-only and `run_manual` uses the default CV seed and
output directory. The audit is an explicit, committed alternative generation
path: it saves a reproduced JSON and verifies every original field with absolute
tolerance 1e-12, without overwriting the historical JSON. The verifier reports
failure if any field differs. Numerical reproduction does not establish which
unrecorded command originally produced the historical file. The old source is
retained unchanged because its hash is part of the completed nested protocol.

The original majority vote still breaks ties by smallest class index. Changing
this now would change the experimental protocol. Window count and discarded
tail length depend on T, while feature values also change with window boundaries
and sample count. For 4097 samples at 173.61 Hz, T=5 seconds gives four 868-sample
windows and discards 625 samples (15.25%). Odd window counts do not eliminate
multiclass ties: counts 2-2-1 still tie. These mechanisms can make the objective
irregular but do not establish why BO chose any particular T. A future voting
ablation requires the same changed rule during search and outer evaluation.

The unused experiment_2 branch was removed in cleanup commit f11c8b1. It is not
evidence for the current 55-feature experiment. Smoke tests and empty logs alone
cannot establish completion of its former full run. No result from that branch
should be claimed as a completed experiment in the dissertation.

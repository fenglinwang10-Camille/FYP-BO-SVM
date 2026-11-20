# Supplementary nested experiment: score-based recording tie-breaking

The original 55-feature pipeline, five search bounds, outer split seed 42,
inner split seeds 1729–1733, search seed 42, 50 evaluations per BO/RS search,
and sixteen shared initial candidates are retained. Only recording aggregation
changes: a unique majority winner is retained; when multiple classes tie for
the most window votes, choose the tied class with the highest mean SVC OVR
decision score across all windows. Exact score ties retain the lowest index.
These decision scores are not probabilities. SVC's own window prediction
tie handling is unchanged.

The rule is used in **both inner validation and outer testing**, consistently
for BO, RS, and the fixed reference. It is not applied just to the outer test
results or only to BO. Each outer fold reruns both searches with the new score
definition; old search scores and selected configurations are not reused.
Each method's selected configuration is frozen before its outer evaluation.

This is an exploratory supplementary reanalysis. The choice of this rule was
informed by an earlier inner-CV pilot on outer fold 1's training data. The
dataset had also been inspected in previous development. Consequently the
new results are not an untouched independent confirmation, and a favorable
outcome must not silently replace the earlier experiment. No rule selection
or search-bound adjustment is performed using the new outer-test results.

## Implementation

`nested_score_tiebreak.py` uses a process-local adapter around the existing
`nested_bo_rs.py` runner. The original experiment's source files remain
unchanged so its frozen protocol hashes and results remain valid. The new
protocol includes the adapter's source hash and the pilot-informed design note.
The operating-system run lock, atomic search checkpoints, deterministic
initial points, fixed folds, earliest-best selection, and resume validation are
reused. Do not edit the hashed source files during a run.

Outputs are separate:
`results/13_nested_score_tiebreak/split_42_inner_1729/search_42`.
Expected full work is 420 inner-CV candidate evaluations, 2100 inner SVM fits,
and fifteen outer fits/evaluations. Each candidate also computes decision
scores, so it may cost more than the original hard-voting evaluation. A single
worker and BLAS thread limits control memory use. Computer sleep pauses work.

Run `run_nested_score_tiebreak.ps1` to start hidden in the background or resume
after interruption. The launcher reports the process ID and timestamped log.
Inspect the root `run_status.json`, each fold's `progress.json`, and the log.
Outer results include tie counts and per-recording vote diagnostics. Once all
five folds finish, `summary.json`, `outer_fold_results.csv`,
`outer_recording_predictions.csv`, and `paired_bo_minus_rs.csv` are available.

Only the uniform mapping is included in this supplementary run. The previous
four-mapping experiment remains a separate completed experiment under the
original hard-voting rule; its results should not be combined with the new
uniform run as a controlled mapping comparison.

## Validation

Ten unit checks passed across the original runner and the new adapter. They
cover 2-2-1 vote ties, restricting score selection to tied classes, preserving
unique vote winners, exact score ties, invalid labels, training-fold isolation,
outer feature-transform ordering, shared initialization, and exact restart.
An integration check on the saved real 55-feature training cache reproduced
86.00% inner CV and 67.50% F recall at C=10, gamma=0.1.
One real shared-initial candidate is run before launching the resumable job.
